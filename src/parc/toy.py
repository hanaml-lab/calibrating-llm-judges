"""A CPU toy setting: long runs with weighted rubrics, human labels and an overconfident, lenient judge whose errors
share a cause within a run. It exercises the whole pipeline (calibration, design, audit, estimate) in seconds."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import IsotonicCalibrator
from .design import draw_audit, plan_audit, residual_correlation
from .estimator import estimate

KINDS = ("state", "artifact", "comms", "process")


@dataclass(frozen=True)
class ToyConfig:
    n_tasks: int = 300
    seeds: int = 2
    k_mean: float = 10.0
    kind_mix: tuple[float, float, float, float] = (0.35, 0.2, 0.25, 0.2)
    judge_sep: tuple[float, float, float, float] = (2.75, 2.4, 2.6, 1.7)
    run_err: float = 0.6        # run-shared judge error (logit scale)
    cell_err: float = 0.6
    kappa: float = 1.75         # overconfident read-out
    lenient: float = 0.55
    claim_bias: float = 0.4     # shift on communication criteria when the run claims success
    c0: float = 5.8
    c1: float = 1.4
    n_cal_runs: int = 160
    cal_version_shift: float = -0.6   # the calibration set comes from an earlier, weaker agent version
    n_judged: int = 2000


@dataclass
class Suite:
    run: np.ndarray
    w: np.ndarray
    y: np.ndarray
    kind: np.ndarray
    raw: np.ndarray
    S: np.ndarray

    @property
    def n_runs(self) -> int:
        return len(self.S)

    def take(self, idx: np.ndarray) -> "Suite":
        """The runs idx (with repeats), re-indexed 0..len(idx)-1."""
        K = np.bincount(self.run, minlength=self.n_runs)
        start = np.concatenate([[0], np.cumsum(K)[:-1]])
        k = K[idx]
        src = np.repeat(start[idx], k) + (np.arange(k.sum()) - np.repeat(np.cumsum(k) - k, k))
        run = np.repeat(np.arange(len(idx)), k)
        return Suite(run, self.w[src], self.y[src], self.kind[src], self.raw[src], self.S[idx])


def make_suite(rng: np.random.Generator, cfg: ToyConfig, n_tasks: int | None = None, seeds: int | None = None,
               quality_shift: float = 0.0) -> Suite:
    n_tasks = cfg.n_tasks if n_tasks is None else n_tasks
    seeds = cfg.seeds if seeds is None else seeds
    run, w, y, kind, raw = [], [], [], [], []
    r = 0
    for _ in range(n_tasks):
        K = int(np.clip(rng.poisson(cfg.k_mean - 1) + 1, 3, 20))
        kinds = rng.choice(4, size=K, p=cfg.kind_mix)
        wt = rng.integers(1, 4, size=K).astype(float)
        wt /= wt.sum()
        ease = rng.normal(0, 0.9, K) + np.where(kinds >= 2, 0.4, 0.0)
        u = rng.normal(0, 0.85)
        for _ in range(seeds):
            z = u + quality_shift + rng.normal(0, 0.9)
            yy = (rng.random(K) < 1 / (1 + np.exp(-(1.4 * z + ease)))).astype(int)
            claim = rng.random() < 1 / (1 + np.exp(-(3.0 * ((wt * yy).sum() - 0.35) + 0.45)))
            sep = np.asarray(cfg.judge_sep)[kinds]
            logit = sep / 2 * (2 * yy - 1) + rng.normal(0, cfg.run_err) + rng.normal(0, cfg.cell_err, K)
            logit += cfg.claim_bias * claim * (kinds == 2)
            rr = 1 / (1 + np.exp(-(cfg.kappa * logit / 1.5 + cfg.lenient)))
            run += [r] * K
            w += wt.tolist()
            y += yy.tolist()
            kind += kinds.tolist()
            raw += rr.tolist()
            r += 1
    run_a = np.asarray(run)
    w_a = np.asarray(w)
    y_a = np.asarray(y)
    return Suite(run_a, w_a, y_a, np.asarray(kind), np.asarray(raw), np.bincount(run_a, weights=w_a * y_a))


def compare(cfg: ToyConfig = ToyConfig(), budget_minutes: float = 480, trials: int = 200, seed: int = 0) -> dict:
    """Full-run grading, the calibrated judge alone, and PARC at the same expected budget."""
    rng = np.random.default_rng(seed)
    pool = make_suite(rng, cfg)
    cal_pool = make_suite(rng, cfg, n_tasks=cfg.n_cal_runs // cfg.seeds, quality_shift=cfg.cal_version_shift)
    cal = IsotonicCalibrator().fit(cal_pool.raw, cal_pool.y, cal_pool.kind)
    rho = residual_correlation(cal_pool.y - cal.predict(cal_pool.raw, cal_pool.kind), cal_pool.w, cal_pool.run)
    truth = pool.S.mean()
    rows = {"full_runs": [], "judge_only": [], "parc": []}
    for _ in range(trials):
        s = pool.take(rng.integers(0, pool.n_runs, cfg.n_judged))
        f = cal.predict(s.raw, s.kind)
        one = np.ones(len(f))
        K = np.bincount(s.run).astype(float)
        plan = plan_audit(one * 0.5, s.w, s.run, budget_minutes, cfg.c0, cfg.c1, rho, active=False, m=K)
        lab, pi = draw_audit(plan, s.run, rng)
        e = estimate(one, s.w, s.run, lab, s.y, pi)
        rows["full_runs"].append((e.value, e.half_width, (plan.p * plan.cost).sum()))
        credit = np.bincount(s.run, weights=s.w * f)
        hw = 1.96 * credit.std(ddof=1) / np.sqrt(len(credit))
        rows["judge_only"].append((credit.mean(), hw, 0.0))
        plan = plan_audit(f, s.w, s.run, budget_minutes, cfg.c0, cfg.c1, rho)
        lab, pi = draw_audit(plan, s.run, rng)
        e = estimate(f, s.w, s.run, lab, s.y, pi)
        rows["parc"].append((e.value, e.half_width, (plan.p * plan.cost).sum()))
    out = {"truth": float(truth), "rho": rho}
    for k, v in rows.items():
        a = np.asarray(v)
        out[k] = dict(estimate=float(a[:, 0].mean()), half_width=float(a[:, 1].mean()),
                      coverage=float((np.abs(a[:, 0] - truth) <= a[:, 1]).mean()), minutes=float(a[:, 2].mean()))
    return out
