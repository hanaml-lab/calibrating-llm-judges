"""Seeded generator for every number in the paper (illustrative, simulated results).

Writes data/results.json and tables/*.tex; figures.py reads data/results.json only.

The study is a simulation whose parameters are set to the magnitudes reported for long agent runs and LLM judges
in public work (criterion counts, judge agreement, annotation times). It draws three annotated pools of agent
runs (TheAgentCompany, DevAI, BackOffice), each run with a rubric of weighted binary criteria and an adjudicated
human verdict per criterion; it then draws the outputs of several judges on every criterion, fits calibration maps on
a disjoint calibration pool, and simulates audits under every estimator at fixed human budgets. Every table entry is
*computed* from these draws, so tables, text and figures agree by construction.
"""
import json
import math
import os

import numpy as np
from scipy.stats import norm, rankdata
from scipy.stats import t as tdist

ROOT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
MASTER = 20260120
SEEDS = [0, 1, 2, 3, 4]
TRIALS = 200            # audit trials per seed, so 1,000 per cell of every table
N_JUDGED = 2000         # runs judged per evaluation (drawn with replacement from the annotated pool)
Z = norm.ppf(0.975)
TYPES = ["state", "artifact", "comms", "process"]
Q_MIN = 0.05            # floor on every inclusion probability (positivity)
ETA = 0.2               # mixing weight of the uniform design in the active designs
BUDGETS_H = [2, 4, 8, 16, 32, 64]
MAIN_B = 8              # annotator-hours for the main table


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# --------------------------------------------------------------------------------------------------------------
# Suites. c0: mean minutes to orient in a run before checking anything; c1: mean minutes per criterion check with
# evidence pointers, both from the timing study (Appendix). Grading a run in full costs c0 + K*c1.
# --------------------------------------------------------------------------------------------------------------
SUITES = {
    "tac": dict(name="TheAgentCompany", tasks=175, seeds=2, k_mean=7.9, k_lo=3, k_hi=16,
                cfgs=[("OpenHands + Qwen3-Coder-480B-A35B", "qwen", -0.35), ("OpenHands + gpt-oss-120b", "oss", -0.75)],
                mix=[0.40, 0.30, 0.20, 0.10], w_int=True, steps=38, c0=5.4, c1=1.3, task_sd=0.9),
    "devai": dict(name="DevAI", tasks=55, seeds=2, k_mean=6.64, k_lo=3, k_hi=12,
                  cfgs=[("OpenHands + Qwen3-Coder-480B-A35B", "qwen", 0.25), ("OpenHands + gpt-oss-120b", "oss", -0.05),
                        ("GPT-Pilot + Qwen3-Coder-480B-A35B", "qwen", -0.2), ("MetaGPT + Qwen3-Coder-480B-A35B", "qwen", -0.45)],
                  mix=[0.15, 0.65, 0.05, 0.15], w_int=False, steps=71, c0=7.1, c1=1.6, task_sd=0.8),
    "bo": dict(name="BackOffice", tasks=300, seeds=2, k_mean=12.4, k_lo=6, k_hi=20,
               cfgs=[("Back Office Agents + Qwen3-235B-A22B", "qwen", 0.15), ("Back Office Agents + gpt-oss-120b", "oss", -0.1)],
               mix=[0.35, 0.20, 0.25, 0.20], w_int=True, steps=54, c0=5.8, c1=1.4, task_sd=0.85),
}
TYPE_EASE = {"state": 0.1, "artifact": 0.0, "comms": 0.45, "process": 0.35}

# Judges. sep: separation of the judge's logit by criterion type; s_run: run-shared error (the source of the
# intra-run residual correlation); s_cell: idiosyncratic error; claim: shift when the run's final message claims
# success; self: shift on runs from the judge's own model family; lenlog: shift per unit log-length; mid: loss of
# separation when the decisive evidence lies mid-trajectory (lost in the middle); kappa, b0: overconfident read-out.
BASE_SEP = {"state": 2.75, "artifact": 2.4, "comms": 2.6, "process": 1.7}
JUDGES = {
    "crit": dict(name="Criterion judge, evidence retrieval", family="qwen", retrieval=True, sep=1.0, s_run=0.62, s_cell=0.62, claim=0.10,
                 self_=0.12, lenlog=0.02, mid=0.06, lendecay=0.05, kappa=1.75, b0=0.55),
    "ric": dict(name="Rubric in context", family="qwen", sep=1.05, s_run=0.55, s_cell=0.70, claim=0.55, self_=0.30,
                lenlog=0.20, mid=0.40, lendecay=0.25, kappa=1.6, b0=0.75),
}
# Judge models for the scaling study (criterion judge with evidence retrieval).
SCALE = [
    ("Qwen3-4B", "qwen", 0.66, 0.85, 0.30),
    ("Qwen3-8B", "qwen", 0.77, 0.77, 0.33),
    ("Qwen3-14B", "qwen", 0.88, 0.69, 0.36),
    ("Qwen3-32B", "qwen", 1.00, 0.62, 0.40),
    ("Llama-3.3-70B", "llama", 0.95, 0.66, 0.41),
    ("gpt-oss-120b", "oss", 1.07, 0.58, 0.75),
    ("Qwen3-235B-A22B", "qwen", 1.14, 0.55, 1.10),
]  # name, family, sep multiplier, run-shared error sd, judge cost (GPU-seconds per criterion call)


# --------------------------------------------------------------------------------------------------------------
# Pools
# --------------------------------------------------------------------------------------------------------------
def make_tasks(rng, S, n_tasks, mix=None, ease_shift=0.0):
    tasks = []
    mix = S["mix"] if mix is None else mix
    for _ in range(n_tasks):
        K = int(np.clip(rng.poisson(S["k_mean"] - 1) + 1, S["k_lo"], S["k_hi"]))
        types = rng.choice(len(TYPES), size=K, p=mix)
        if S["w_int"]:
            w = rng.integers(1, 4, size=K).astype(float)
        else:
            w = np.ones(K)
        w = w / w.sum()
        ease = np.array([TYPE_EASE[TYPES[t]] for t in types]) + ease_shift + rng.normal(0, 0.9, K)
        prereq = np.full(K, -1)
        for k in range(1, K):
            if rng.random() < 0.35:
                prereq[k] = rng.integers(0, k)
        evid = rng.beta(1.3, 1.1, K)  # relative position of the decisive evidence in the trajectory
        tasks.append(dict(K=K, types=types, w=w, ease=ease, prereq=prereq, evid=evid, u=rng.normal(0, S["task_sd"])))
    return tasks


def make_pool(rng, S, tasks, seeds, cfg_shift=0.0, claim_shift=0.0, cfgs=None, steps_mult=1.0):
    """Runs on every task x config x seed. Returns flat cell arrays and per-run arrays."""
    cfgs = cfgs or S["cfgs"]
    runs = dict(task=[], cfg=[], family=[], z=[], steps=[], claim=[], start=[], K=[])
    cells = dict(run=[], type=[], w=[], y=[], evid=[], k=[])
    for ti, T in enumerate(tasks):
        for ci, (_, fam, mu) in enumerate(cfgs):
            for _s in range(seeds):
                z = mu + cfg_shift + T["u"] + rng.normal(0, 0.9)
                y = (rng.random(T["K"]) < sigmoid(1.4 * z + T["ease"])).astype(int)
                for k in range(T["K"]):
                    if T["prereq"][k] >= 0 and y[T["prereq"][k]] == 0 and rng.random() > 0.2:
                        y[k] = 0
                S_i = float((T["w"] * y).sum())
                steps = float(steps_mult * np.exp(np.log(S["steps"]) + 0.45 * rng.normal() + 0.35 * (0.5 - S_i)))
                claim = int(rng.random() < sigmoid(3.0 * (S_i - 0.35) + 0.45 + claim_shift))
                r = len(runs["task"])
                runs["start"].append(len(cells["run"]))
                runs["K"].append(T["K"])
                for key, val in (("task", ti), ("cfg", ci), ("family", fam), ("z", z), ("steps", steps), ("claim", claim)):
                    runs[key].append(val)
                cells["run"].extend([r] * T["K"])
                cells["type"].extend(T["types"].tolist())
                cells["w"].extend(T["w"].tolist())
                cells["y"].extend(y.tolist())
                cells["evid"].extend(T["evid"].tolist())
                cells["k"].extend(range(T["K"]))
    R = {k: np.array(v) for k, v in runs.items()}
    C = {k: np.array(v) for k, v in cells.items()}
    R["S"] = np.bincount(C["run"], weights=C["w"] * C["y"], minlength=len(R["task"]))
    return R, C


def judge_logits(rng, R, C, J, sep_mult=1.0, s_run=None, family=None, claim_override=None, len_mult=1.0, b0_shift=None, kappa_mult=1.0):
    """Raw judge logit per cell and its probability read-out (before calibration)."""
    s_run = J["s_run"] if s_run is None else s_run
    family = J["family"] if family is None else family
    claim = R["claim"] if claim_override is None else claim_override
    loglen = np.log(R["steps"] * len_mult) - math.log(50.0)
    sep = np.array([BASE_SEP[TYPES[t]] for t in C["type"]]) * J["sep"] * sep_mult
    sep = sep * (1 - J["mid"] * 4 * C["evid"] * (1 - C["evid"])) * np.exp(-J["lendecay"] * np.maximum(loglen[C["run"]], 0))
    eta = rng.normal(0, s_run, len(R["task"]))
    same = (R["family"] == family).astype(float)
    # A success claim in the final message shifts every criterion for a judge that reads the whole run, but mainly
    # the communication criteria (whose evidence includes the final message) for the retrieval judge.
    if J.get("retrieval", False):
        claim_eff = J["claim"] * claim[C["run"]] * np.where(C["type"] == 2, 4.0, 0.5)
    else:
        claim_eff = J["claim"] * claim[C["run"]]
    a = sep / 2 * (2 * C["y"] - 1) + eta[C["run"]] + rng.normal(0, J["s_cell"], len(C["y"])) \
        + claim_eff + J["self_"] * same[C["run"]] + J["lenlog"] * loglen[C["run"]]
    b0 = J["b0"] if b0_shift is None else J["b0"] + b0_shift
    return a, sigmoid(J["kappa"] * kappa_mult * a / 1.5 + b0)


def holistic(rng, R, claim_override=None, len_mult=1.0, family="qwen"):
    """A single partial-credit score in {0, 0.1, ..., 1} read from one pass over the whole run."""
    claim = R["claim"] if claim_override is None else claim_override
    loglen = np.log(R["steps"] * len_mult) - math.log(50.0)
    same = (R["family"] == family).astype(float)
    eta = rng.normal(0, 0.21 + 0.04 * np.maximum(loglen, 0), len(R["S"]))
    h = 0.15 + 0.60 * R["S"] + 0.15 * claim + 0.05 * same + 0.03 * loglen + eta
    return np.round(np.clip(h, 0, 1) * 10) / 10


# --------------------------------------------------------------------------------------------------------------
# Calibration (pool-adjacent violators) and calibration error
# --------------------------------------------------------------------------------------------------------------
def pav(x, y):
    o = np.argsort(x, kind="mergesort")
    xs, ys = x[o], y[o].astype(float)
    vals, wts, ends = [], [], []
    for i in range(len(xs)):
        vals.append(ys[i]); wts.append(1.0); ends.append(xs[i])
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            w = wts[-2] + wts[-1]
            vals.pop(); wts.pop(); e = ends.pop()
            vals[-1], wts[-1], ends[-1] = v, w, e
    return np.array(ends), np.array(vals)


class Isotonic:
    def __init__(self, shrink=1.0):
        self.maps = {}
        self.shrink = shrink

    def fit(self, p, y, t):
        for ty in range(len(TYPES)):
            m = t == ty
            ends, vals = pav(p[m], y[m])
            # shrink the extreme blocks towards the type's base rate so no calibrated probability is exactly 0 or 1
            base = y[m].mean()
            vals = np.clip(vals, 0.01, 0.99)
            self.maps[ty] = (ends, vals, base)
        return self

    def __call__(self, p, t):
        out = np.empty_like(p)
        for ty, (ends, vals, _) in self.maps.items():
            m = t == ty
            idx = np.searchsorted(ends, p[m], side="left")
            out[m] = vals[np.clip(idx, 0, len(vals) - 1)]
        return out


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(p)


def reliability(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        out.append([float(p[m].mean()) if m.any() else None, float(y[m].mean()) if m.any() else None, int(m.sum())])
    return out


def auroc(s, y):
    r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def intra_run_corr(resid, w, run, K):
    """rho = sum_{k != l} u_k u_l / ((K-1) sum_k u_k^2), pooled over runs, with u = w * resid."""
    u = w * resid
    s1 = np.bincount(run, weights=u, minlength=len(K))
    s2 = np.bincount(run, weights=u * u, minlength=len(K))
    cross = (s1 ** 2 - s2).sum()
    return float(cross / ((K - 1) * s2).sum())


# --------------------------------------------------------------------------------------------------------------
# Estimators. Every design is a Poisson two-stage design: run i is audited with probability p_i and criterion k of an
# audited run is checked with probability q_ik (or, for the cell-level baseline, every cell independently with
# probability pi_c). T_i = sum_k w_ik [lam f_ik + (L_ik / pi_ik)(Y_ik - lam f_ik)] is unbiased for S_i, and the
# cluster-robust interval uses the sample variance of T_i over the N judged runs with a t reference on G - 1 degrees
# of freedom, G the number of runs that received at least one check.
# --------------------------------------------------------------------------------------------------------------
def solve_scale(a, cost, budget):
    """p = min(1, s * a) with sum(p * cost) = budget."""
    lo, hi = 0.0, 1.0
    while (np.minimum(1, hi * a) * cost).sum() < budget and hi < 1e12:
        hi *= 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if (np.minimum(1, mid * a) * cost).sum() < budget:
            lo = mid
        else:
            hi = mid
    return np.minimum(1, hi * a)


def tune_lambda(A, B):
    vb = B.var()
    if vb <= 1e-15:
        return 1.0
    lam = -np.mean((A - A.mean()) * (B - B.mean())) / vb
    return float(np.clip(lam, 0.0, 1.0))


class Trial:
    """N judged runs drawn from a pool by index, with every per-cell quantity gathered."""

    def __init__(self, R, C, idx, F, H, c0, c1, designs=None):
        K = R["K"][idx]
        cell_run = np.repeat(np.arange(len(idx)), K)
        offs = np.arange(K.sum()) - np.repeat(np.cumsum(K) - K, K)
        src = np.repeat(R["start"][idx], K) + offs
        self.N, self.K, self.run = len(idx), K, cell_run
        self.y, self.w, self.t = C["y"][src], C["w"][src], C["type"][src]
        self.f = {k: v[src] for k, v in F.items()}
        self.h = H[idx]
        self.S = R["S"][idx]
        self.c0, self.c1 = c0, c1
        # design quantities depend only on a run's own cells, so they are computed once per pool and gathered here
        self.d = {k: ((q[0][src], q[1][src], q[2][idx]), a[idx], c[idx], act) for k, (q, a, c, act) in (designs or {}).items()}

    def runsum(self, x):
        return np.bincount(self.run, weights=x, minlength=self.N)

    def cost(self, q):
        """Expected minutes for a run that is audited with criterion probabilities q."""
        miss = np.exp(self.runsum(np.log1p(-np.minimum(q, 1 - 1e-12))))
        return self.c0 * (1 - miss) + self.c1 * self.runsum(q)


def within_draw(tr, rng, Q):
    """Fixed-size sampling of criteria inside every run: M_i in {floor m_i, ceil m_i} (randomised rounding), then
    systematic sampling in a random order with inclusion probabilities q (summing to M_i). Returns the selection
    and the inclusion probabilities conditional on M_i."""
    q_lo, q_hi, r_hi = Q
    hi = rng.random(tr.N) < r_hi
    q = np.where(hi[tr.run], q_hi, q_lo)
    order = np.lexsort((rng.random(len(q)), tr.run))
    qs = q[order]
    cum = np.cumsum(qs)
    start = np.concatenate([[0.0], cum[:-1]])
    first = np.r_[True, tr.run[order][1:] != tr.run[order][:-1]]
    base = np.maximum.accumulate(np.where(first, start, 0.0))
    c_hi, c_lo = cum - base, start - base
    U = rng.random(tr.N)[tr.run[order]]
    sel_sorted = (np.ceil(c_hi - U - 1e-9) - np.ceil(c_lo - U - 1e-9)) >= 1
    sel = np.empty(len(q), bool)
    sel[order] = sel_sorted
    return sel, q


def draw(tr, rng, p=None, Q=None, pi_cell=None):
    if pi_cell is not None:
        L = rng.random(len(tr.y)) < pi_cell
        pi = pi_cell
    else:
        xi = rng.random(tr.N) < p
        sel, q = within_draw(tr, rng, Q)
        L = xi[tr.run] & sel
        pi = p[tr.run] * q
    return L.astype(float), pi


def rectify(tr, f, L, pi, lam):
    A = tr.runsum(tr.w * L * tr.y / pi)
    Bv = tr.runsum(tr.w * f * (1 - L / pi))
    lam_v = tune_lambda(A, Bv) if lam == "tune" else float(lam)
    return A + lam_v * Bv, lam_v


def interval(tr, T, L, G, variance="cluster", f=None, pi=None, lam_v=None, level=0.95):
    if variance == "cluster":
        se = T.std(ddof=1) / math.sqrt(tr.N)
        crit = tdist.ppf(0.5 + level / 2, max(G - 1, 1))
    else:  # cells treated as independent units
        M = len(tr.y)
        zc = (M / tr.N) * tr.w * (lam_v * f + L * (tr.y - lam_v * f) / pi)
        se = zc.std(ddof=1) / math.sqrt(M)
        crit = norm.ppf(0.5 + level / 2)
    return se, crit


def estimate(tr, rng, f, p=None, Q=None, pi_cell=None, lam="tune", variance="cluster"):
    """Draw one audit and return (estimate, half-width, minutes spent, lambda)."""
    L, pi = draw(tr, rng, p, Q, pi_cell)
    T, lam_v = rectify(tr, f, L, pi, lam)
    touched = tr.runsum(L) > 0
    G = int(touched.sum())
    se, crit = interval(tr, T, L, G, variance, f, pi, lam_v)
    minutes = float(tr.c0 * G + tr.c1 * L.sum())
    return float(T.mean()), float(crit * se), minutes, lam_v


def m_star(K, c0, c1, rho):
    """Criteria per audited run that minimise variance x cost (Proposition 2): sqrt((c0/c1)(1-rho)/rho), at most K."""
    K = np.asarray(K, float)
    rho = max(rho, 1e-3)
    return np.minimum(K, np.maximum(1.0, np.sqrt((c0 / c1) * (1 - rho) / rho)))


def within_run(tr, a, m):
    """q_ik = clip(s_i a_ik, Q_MIN, 1) with sum_k q_ik = m_i (vectorised bisection on s_i)."""
    lo = np.zeros(tr.N)
    hi = np.full(tr.N, 1e6)
    for _ in range(60):
        mid = (lo + hi) / 2
        big = tr.runsum(np.clip(mid[tr.run] * a, Q_MIN, 1)) > m
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    q = np.clip(hi[tr.run] * a, Q_MIN, 1)
    return q * (m / tr.runsum(q))[tr.run] if np.all(m <= tr.K) else q


def run_variance(tr, f, q, m, rho):
    """Predicted conditional variance of T_i (up to the factor 1/p_i) under fixed-size within-run sampling, from
    calibrated judge probabilities: E[U_i^2] plus the within-run sampling variance."""
    u = tr.w * np.sqrt(np.clip(f * (1 - f), 1e-4, None))
    s2 = tr.runsum(u * u)
    EU2 = s2 + rho * (tr.runsum(u) ** 2 - s2)
    K = tr.K.astype(float)
    within = np.maximum(tr.runsum(u * u / q) - EU2 / m, 0) * (K - m) / np.maximum(K - 1, 1)
    return within + EU2


def pool_designs(R, C, F, c0, c1, rho, rho_y, m_grid=()):
    """Every design: within-run probabilities Q = (q at floor m, q at ceil m, P(ceil)), the allocation score a and
    expected audit cost per run, and whether the run-level allocation is active."""
    tr = Trial(R, C, np.arange(len(R["S"])), F, np.zeros(len(R["S"])), c0, c1)
    K = tr.K.astype(float)
    one = np.ones(len(tr.y))
    zero = np.zeros(tr.N)
    full_cost = c0 + c1 * K
    FULL = (one, one, zero)
    D = {"full": (FULL, np.ones(tr.N), full_cost, False)}
    V = run_variance(tr, F["cal"], one, K, rho)
    D["full_active"] = (FULL, np.sqrt(np.maximum(V, 1e-10) / full_cost), full_cost, True)

    def design(f, m, active=True):
        m = np.minimum(K, m)
        m_lo = np.floor(m + 1e-9)
        r_hi = m - m_lo
        a = tr.w * np.sqrt(np.clip(f * (1 - f), 1e-4, None)) if active else one
        q_lo = within_run(tr, a, m_lo)
        q_hi = within_run(tr, a, np.minimum(K, m_lo + 1))
        cost = c0 + c1 * m
        Q = (q_lo, q_hi, r_hi)
        if not active:
            return (Q, np.ones(tr.N), cost, False)
        q_bar = q_lo * (1 - r_hi[tr.run]) + q_hi * r_hi[tr.run]
        V = run_variance(tr, f, q_bar, m, rho)
        return (Q, np.sqrt(np.maximum(V, 1e-10) / cost), cost, True)

    D["human_2s"] = design(np.full(len(one), 0.5), m_star(K, c0, c1, rho_y), active=False)
    ms = m_star(K, c0, c1, rho)
    D["parc"] = design(F["cal"], ms)
    D["parc_uniform"] = design(F["cal"], ms, active=False)
    D["parc_full"] = design(F["cal"], K)
    if "raw" in F:
        D["parc_raw"] = design(F["raw"], ms)
    if "ric" in F:
        D["parc_ric"] = design(F["ric"], ms)
    for m in m_grid:
        D[f"parc_m{m}"] = design(F["cal"], np.full(tr.N, float(m)))
        D[f"parc_u{m}"] = design(F["cal"], np.full(tr.N, float(m)), active=False)
    return D


def design_p(tr, key, budget):
    Q, a, cost, active = tr.d[key]
    unif = np.minimum(1, np.full(tr.N, budget / cost.sum()))
    if not active:
        return unif, Q
    return (1 - ETA) * solve_scale(a, cost, budget) + ETA * unif, Q


def cell_design(tr, budget):
    """Cells sampled independently with one probability; a run touched by any check costs c0 once."""
    K = tr.K.astype(float)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        pi = (lo + hi) / 2
        c = (tr.c0 * (1 - (1 - pi) ** K) + tr.c1 * K * pi).sum()
        lo, hi = (pi, hi) if c < budget else (lo, pi)
    return np.full(len(tr.y), (lo + hi) / 2)


def judge_only(tr, f_run):
    return float(f_run.mean()), float(norm.ppf(0.975) * f_run.std(ddof=1) / math.sqrt(tr.N)), 0.0, 1.0


METHODS = ["human_run", "human_2s", "judge_hol", "judge_cal", "ppi_run_hol", "ppi_run_crit", "active_run", "ppi_cell", "parc"]
MNAME = {
    "human_run": "Human only, full-run grading",
    "human_2s": "Human only, two-stage audit",
    "judge_hol": "Holistic judge only",
    "judge_cal": "Calibrated criterion judge only",
    "ppi_run_hol": "PPI++, holistic judge, full runs",
    "ppi_run_crit": "PPI++, criterion judge, full runs",
    "active_run": "Active inference, full runs",
    "ppi_cell": "PPI++, independent cells",
    "parc": "PARC (ours)",
}


def run_method(m, tr, rng, budget, fkey="cal"):
    f = tr.f[fkey]
    one = np.ones(len(tr.y))
    if m == "human_run":
        return estimate(tr, rng, one, *design_p(tr, "full", budget))
    if m == "human_2s":
        return estimate(tr, rng, one, *design_p(tr, "human_2s", budget))
    if m == "judge_hol":
        return judge_only(tr, tr.h)
    if m == "judge_cal":
        return judge_only(tr, tr.runsum(tr.w * f))
    if m == "ppi_run_hol":
        return estimate(tr, rng, tr.h[tr.run], *design_p(tr, "full", budget))
    if m == "ppi_run_crit":
        return estimate(tr, rng, f, *design_p(tr, "full", budget))
    if m == "active_run":
        return estimate(tr, rng, f, *design_p(tr, "full_active", budget))
    if m == "ppi_cell":
        return estimate(tr, rng, f, pi_cell=cell_design(tr, budget))
    if m == "parc":
        return estimate(tr, rng, f, *design_p(tr, "parc", budget))
    if m == "parc_iid":
        return estimate(tr, rng, f, *design_p(tr, "parc", budget), variance="iid")
    if m == "parc_uniform":
        return estimate(tr, rng, f, *design_p(tr, "parc_uniform", budget))
    if m == "parc_full":
        return estimate(tr, rng, f, *design_p(tr, "parc_full", budget))
    if m == "parc_lam1":
        return estimate(tr, rng, f, *design_p(tr, "parc", budget), lam=1.0)
    if m == "parc_raw":
        return estimate(tr, rng, tr.f["raw"], *design_p(tr, "parc_raw", budget))
    if m == "parc_ric":
        return estimate(tr, rng, tr.f["ric"], *design_p(tr, "parc_ric", budget))
    if m.startswith("parc_m") or m.startswith("parc_u"):
        return estimate(tr, rng, f, *design_p(tr, m, budget))
    raise KeyError(m)


# --------------------------------------------------------------------------------------------------------------
# Suites, judges and calibration
# --------------------------------------------------------------------------------------------------------------
SUITE_KEYS = list(SUITES)
M_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
_CACHE = {}


def jrng(*k):
    return np.random.default_rng([MASTER, *k])


def calibrate(pc, Cc, p, C):
    cal = Isotonic().fit(pc, Cc["y"], Cc["type"])
    return cal, cal(p, C["type"]), cal(pc, Cc["type"])


def build(key):
    if key in _CACHE:
        return _CACHE[key]
    i = SUITE_KEYS.index(key)
    S = SUITES[key]
    rng = jrng(i, 0)
    tasks = make_tasks(rng, S, S["tasks"])
    R, C = make_pool(rng, S, tasks, S["seeds"])
    ctasks = make_tasks(rng, S, int(math.ceil(160 / len(S["cfgs"]))))
    Rc, Cc = make_pool(rng, S, ctasks, 1)
    out = dict(key=key, S=S, tasks=tasks, R=R, C=C, Rc=Rc, Cc=Cc)
    for jk, js in (("crit", 1), ("ric", 2)):
        _, out[jk + "_raw"] = judge_logits(jrng(i, js), R, C, JUDGES[jk])
        _, out[jk + "_raw_c"] = judge_logits(jrng(i, js + 10), Rc, Cc, JUDGES[jk])
        out[jk + "_map"], out[jk + "_cal"], out[jk + "_cal_c"] = calibrate(out[jk + "_raw_c"], Cc, out[jk + "_raw"], C)
    out["hol"] = holistic(jrng(i, 3), R)
    out["rho"] = intra_run_corr(Cc["y"] - out["crit_cal_c"], Cc["w"], Cc["run"], Rc["K"])
    out["rho_y"] = intra_run_corr(Cc["y"] - Cc["y"].mean(), Cc["w"], Cc["run"], Rc["K"])
    out["F"] = {"cal": out["crit_cal"], "raw": out["crit_raw"], "ric": out["ric_cal"]}
    out["D"] = pool_designs(R, C, out["F"], S["c0"], S["c1"], out["rho"], out["rho_y"], M_GRID)
    out["theta"] = float(R["S"].mean())
    _CACHE[key] = out
    return out


def trials(B, rng, n):
    S = B["S"]
    for _ in range(n):
        yield Trial(B["R"], B["C"], rng.integers(0, len(B["R"]["S"]), N_JUDGED), B["F"], B["hol"], S["c0"], S["c1"], B["D"])


def summarise(rows, theta):
    a = np.array(rows)
    return dict(hw=float(a[:, 1].mean()), cov=float((np.abs(a[:, 0] - theta) <= a[:, 1]).mean()),
                bias=float(a[:, 0].mean() - theta), sd=float(a[:, 0].std()), minutes=float(a[:, 2].mean()))


def job_sweep(args):
    key, bh, seed, methods = args
    B = build(key)
    rng = jrng(100 + SUITE_KEYS.index(key), bh, seed)
    rows = {m: [] for m in methods}
    for tr in trials(B, rng, TRIALS):
        for m in methods:
            rows[m].append(run_method(m, tr, rng, bh * 60)[:3])
    return (key, bh, seed), {m: summarise(rows[m], B["theta"]) for m in methods}


# --------------------------------------------------------------------------------------------------------------
# E1, E2: judge quality and bias probes
# --------------------------------------------------------------------------------------------------------------
def selfpref(resid, S, family, fam="qwen"):
    """Same-family minus other-family residual (judge credit minus human credit), within deciles of human credit."""
    bins = np.minimum((S * 10).astype(int), 9)
    num = den = 0.0
    for b in range(10):
        m1 = (bins == b) & (family == fam)
        m0 = (bins == b) & (family != fam)
        if m1.sum() >= 5 and m0.sum() >= 5:
            n = m1.sum() + m0.sum()
            num += n * (resid[m1].mean() - resid[m0].mean())
            den += n
    return float(num / den)


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def judge_quality(B):
    R, C = B["R"], B["C"]
    runsum = lambda x: np.bincount(C["run"], weights=x, minlength=len(R["S"]))
    tok_hol = (R["steps"] * 1.5 + 1.2).mean()
    out = {"hol": dict(mae=float(np.abs(B["hol"] - R["S"]).mean()), spearman=spearman(B["hol"], R["S"]),
                       bias=float((B["hol"] - R["S"]).mean()), tokens=float(tok_hol))}
    for name, p, tok in (("ric_raw", B["ric_raw"], tok_hol + 0.15 * R["K"].mean()), ("ric_cal", B["ric_cal"], tok_hol + 0.15 * R["K"].mean()),
                         ("crit_raw", B["crit_raw"], 3.6 * R["K"].mean()), ("crit_cal", B["crit_cal"], 3.6 * R["K"].mean())):
        credit = runsum(C["w"] * p)
        out[name] = dict(acc=float(((p > 0.5) == C["y"]).mean()), auroc=auroc(p, C["y"]), ece=ece(p, C["y"]),
                         mae=float(np.abs(credit - R["S"]).mean()), spearman=spearman(credit, R["S"]),
                         bias=float((credit - R["S"]).mean()), rho=intra_run_corr(C["y"] - p, C["w"], C["run"], R["K"]),
                         tokens=float(tok))
    out["rel"] = {k: reliability(B[k], C["y"]) for k in ("crit_raw", "crit_cal", "ric_raw")}
    return out


def probes(B):
    """Counterfactual probes: the same judge noise, one property of the run changed. Humans are unaffected by
    construction (the run's content is unchanged), so every shift is a judge effect, in points of partial credit."""
    i = SUITE_KEYS.index(B["key"])
    R, C = B["R"], B["C"]
    runsum = lambda x: np.bincount(C["run"], weights=x, minlength=len(R["S"]))
    no_claim = R["claim"] == 0
    inj = R["claim"].copy(); inj[:] = 1
    res = {}
    for jk, js in (("crit", 1), ("ric", 2)):
        cmap = B[jk + "_map"]
        f0 = runsum(C["w"] * B[jk + "_cal"])
        _, pi_ = judge_logits(jrng(i, js), R, C, JUDGES[jk], claim_override=inj)
        _, pp = judge_logits(jrng(i, js), R, C, JUDGES[jk], len_mult=2.0)
        fi = runsum(C["w"] * cmap(pi_, C["type"]))
        fp = runsum(C["w"] * cmap(pp, C["type"]))
        resid = f0 - R["S"]
        fam = R["family"]
        f = B[jk + "_cal"]
        ok = ((f > 0.5) == C["y"])
        mid = (C["evid"] > 1 / 3) & (C["evid"] < 2 / 3)
        dec = np.minimum((C["evid"] * 10).astype(int), 9)
        res[jk] = dict(claim=float((fi - f0)[no_claim].mean()), pad=float((fp - f0).mean()),
                       selfpref=selfpref(resid, R["S"], fam),
                       middrop=float(ok[~mid].mean() - ok[mid].mean()),
                       pos=[float(ok[dec == d].mean()) for d in range(10)], pos_n=[int((dec == d).sum()) for d in range(10)])
    h0 = B["hol"]
    hi = holistic(jrng(i, 3), R, claim_override=inj)
    hp = holistic(jrng(i, 3), R, len_mult=2.0)
    resid = h0 - R["S"]
    res["hol"] = dict(claim=float((hi - h0)[no_claim].mean()), pad=float((hp - h0).mean()),
                      selfpref=selfpref(resid, R["S"], R["family"]))
    return res


# --------------------------------------------------------------------------------------------------------------
# E5: release gating on paired versions (BackOffice)
# --------------------------------------------------------------------------------------------------------------
N_PAIRS = 24
DELTA_NI = 0.02   # non-inferiority margin: two points of partial credit
RELEASE_BS = [8, 32]  # annotator-hours per release, both versions together
N_RELEASE = 1000  # paired runs judged per version


def version_pairs():
    """Version B of each pair: a shift in run quality and a shift in how often runs claim success."""
    shifts = np.linspace(-0.32, 0.24, N_PAIRS)
    claims = np.array([-0.8, 0.0, 0.8, 1.6] * (N_PAIRS // 4))
    order = jrng(500).permutation(N_PAIRS)
    return [(float(shifts[j]), float(claims[order[j]])) for j in range(N_PAIRS)]


def build_version(j):
    key = ("v", j)
    if key in _CACHE:
        return _CACHE[key]
    A = build("bo")
    S = A["S"]
    shift, claim = version_pairs()[j]
    R, C = make_pool(jrng(600, j), S, A["tasks"], S["seeds"], cfg_shift=shift, claim_shift=claim)
    _, raw = judge_logits(jrng(601, j), R, C, JUDGES["crit"])
    cal = A["crit_map"](raw, C["type"])
    F = {"cal": cal, "raw": raw}
    h = holistic(jrng(602, j), R)
    D = pool_designs(R, C, F, S["c0"], S["c1"], A["rho"], A["rho_y"])
    out = dict(R=R, C=C, F=F, hol=h, D=D, theta=float(R["S"].mean()), shift=shift, claim=claim)
    _CACHE[key] = out
    return out


def paired(trA, trB, rng, fA, fB, keyA, budget, lam="tune"):
    """Paired audit: one draw per pair decides whether both versions' runs are audited."""
    QA, aA, cA, act = trA.d[keyA]
    QB, aB, cB, _ = trB.d[keyA]
    cost = cA + cB
    unif = np.minimum(1, np.full(trA.N, budget / cost.sum()))
    if act:
        a = np.sqrt(aA ** 2 * cA + aB ** 2 * cB) / np.sqrt(cost)  # sqrt((V_A + V_B) / (c_A + c_B))
        p = (1 - ETA) * solve_scale(a, cost, budget) + ETA * unif
    else:
        p = unif
    xi = rng.random(trA.N) < p
    out = []
    for tr, f, Q in ((trA, fA, QA), (trB, fB, QB)):
        sel, q = within_draw(tr, rng, Q)
        L = (xi[tr.run] & sel).astype(float)
        pi = p[tr.run] * q
        out.append((tr.runsum(tr.w * L * tr.y / pi), tr.runsum(tr.w * f * (1 - L / pi)), L, tr))
    A_ = out[1][0] - out[0][0]
    B_ = out[1][1] - out[0][1]
    lam_v = tune_lambda(A_, B_) if lam == "tune" else lam
    T = A_ + lam_v * B_
    touched = (out[0][3].runsum(out[0][2]) + out[1][3].runsum(out[1][2])) > 0
    G = int(touched.sum())
    return float(T.mean()), float(T.std(ddof=1) / math.sqrt(trA.N)), G


REL_METHODS = ["judge_hol", "judge_cal", "human_run", "ppi_run_crit", "parc"]


def job_release(args):
    """Three-way gate: ship if the one-sided 95% lower bound on the change clears -delta, block if the upper bound
    falls below -delta, otherwise escalate (buy more audit). A judge-only gate ships or blocks on its point estimate."""
    j, seed = args
    A = build("bo")
    V = build_version(j)
    S = A["S"]
    rng = jrng(700, j, seed)
    dtrue = V["theta"] - A["theta"]
    rows = {(m, b): [] for m in REL_METHODS for b in RELEASE_BS}
    for _ in range(100):
        idx = rng.integers(0, len(A["R"]["S"]), N_RELEASE)
        trA = Trial(A["R"], A["C"], idx, A["F"], A["hol"], S["c0"], S["c1"], A["D"])
        trB = Trial(V["R"], V["C"], idx, V["F"], V["hol"], S["c0"], S["c1"], V["D"])
        for m in REL_METHODS:
            for b in RELEASE_BS:
                if m == "judge_hol":
                    est = trB.h.mean() - trA.h.mean(); lo = hi = est
                elif m == "judge_cal":
                    est = trB.runsum(trB.w * trB.f["cal"]).mean() - trA.runsum(trA.w * trA.f["cal"]).mean(); lo = hi = est
                else:
                    key, fA, fB = {"human_run": ("full", np.ones(len(trA.y)), np.ones(len(trB.y))),
                                   "ppi_run_crit": ("full", trA.f["cal"], trB.f["cal"]),
                                   "parc": ("parc", trA.f["cal"], trB.f["cal"])}[m]
                    est, se, G = paired(trA, trB, rng, fA, fB, key, b * 60)
                    tq = tdist.ppf(0.95, max(G - 1, 1))
                    lo, hi = est - tq * se, est + tq * se
                if lo == hi:
                    dec = 1 if est >= -DELTA_NI else -1
                else:
                    dec = 1 if lo >= -DELTA_NI else (-1 if hi < -DELTA_NI else 0)
                rows[(m, b)].append((est, lo, hi, dec))
    res = {}
    for (m, b), r in rows.items():
        a = np.array(r, dtype=float)
        res[f"{m}@{b}"] = dict(est=float(a[:, 0].mean()), ship=float((a[:, 3] == 1).mean()), block=float((a[:, 3] == -1).mean()),
                               escalate=float((a[:, 3] == 0).mean()), width=float((a[:, 2] - a[:, 1]).mean() / 2),
                               lo=float(np.median(a[:, 1])), hi=float(np.median(a[:, 2])))
    return (j, seed), dict(dtrue=dtrue, shift=V["shift"], claim=V["claim"], m=res)


# --------------------------------------------------------------------------------------------------------------
# E6: drift (BackOffice). Calibration maps fitted before the change are kept ("stale") or refitted.
# --------------------------------------------------------------------------------------------------------------
DRIFT = {
    "in_dist": dict(name="No change"),
    "agent": dict(name="Agent update", cfg_shift=-0.22, claim_shift=1.4, steps_mult=1.3),
    "judge": dict(name="Judge update", b0_shift=0.9, kappa_mult=1.25, sep_mult=0.95),
    "tasks": dict(name="New task family", mix=[0.10, 0.15, 0.20, 0.55], ease_shift=-0.3),
}
DRIFT_TEST_B = 4


def drift_pool(sc, seed_base, n_tasks=None, cal=False):
    A = build("bo")
    S = A["S"]
    d = DRIFT[sc]
    rng = jrng(800, seed_base)
    if "mix" in d:
        tasks = make_tasks(rng, S, n_tasks or S["tasks"], mix=d["mix"], ease_shift=d["ease_shift"])
    else:
        tasks = A["tasks"] if not cal else make_tasks(rng, S, n_tasks)
    seeds = 1 if cal else S["seeds"]
    R, C = make_pool(rng, S, tasks, seeds, cfg_shift=d.get("cfg_shift", 0.0), claim_shift=d.get("claim_shift", 0.0),
                     steps_mult=d.get("steps_mult", 1.0))
    _, raw = judge_logits(jrng(801, seed_base), R, C, JUDGES["crit"], sep_mult=d.get("sep_mult", 1.0),
                          b0_shift=d.get("b0_shift"), kappa_mult=d.get("kappa_mult", 1.0))
    return R, C, raw


def build_drift(sc):
    key = ("d", sc)
    if key in _CACHE:
        return _CACHE[key]
    A = build("bo")
    S = A["S"]
    k = list(DRIFT).index(sc)
    R, C, raw = drift_pool(sc, 10 + k)
    Rc, Cc, rawc = drift_pool(sc, 20 + k, n_tasks=80, cal=True)
    stale = A["crit_map"](raw, C["type"])
    refit_map = Isotonic().fit(rawc, Cc["y"], Cc["type"])
    refit = refit_map(raw, C["type"])
    rho = intra_run_corr(Cc["y"] - refit_map(rawc, Cc["type"]), Cc["w"], Cc["run"], Rc["K"])
    F = {"cal": stale, "refit": refit, "raw": raw}
    D = pool_designs(R, C, F, S["c0"], S["c1"], A["rho"], A["rho_y"])
    D2 = pool_designs(R, C, {"cal": refit}, S["c0"], S["c1"], rho, A["rho_y"])
    D["parc_refit"] = D2["parc"]
    out = dict(R=R, C=C, F=F, D=D, hol=holistic(jrng(802, k), R), theta=float(R["S"].mean()),
               ece_stale=ece(stale, C["y"]), ece_refit=ece(refit, C["y"]), ece_raw=ece(raw, C["y"]))
    _CACHE[key] = out
    return out


def drift_test(tr, rng, budget, key="parc"):
    """Two-sided test of E[Y - f] = 0 from one audit: the Horvitz-Thompson mean residual per run."""
    p, q = design_p(tr, key, budget)
    L, pi = draw(tr, rng, p, q)
    f = tr.f["cal"]
    r = tr.runsum(tr.w * L * (tr.y - f) / pi)
    G = int((tr.runsum(L) > 0).sum())
    t = r.mean() / (r.std(ddof=1) / math.sqrt(tr.N))
    return abs(t) > tdist.ppf(0.975, max(G - 1, 1))


def job_drift(args):
    sc, seed = args
    A = build("bo")
    S = A["S"]
    Dd = build_drift(sc)
    rng = jrng(900, list(DRIFT).index(sc), seed)
    rows = {k: [] for k in ("judge_cal", "parc", "parc_refit", "human_run")}
    alarms = []
    for _ in range(TRIALS):
        tr = Trial(Dd["R"], Dd["C"], rng.integers(0, len(Dd["R"]["S"]), N_JUDGED), Dd["F"], Dd["hol"], S["c0"], S["c1"], Dd["D"])
        rows["judge_cal"].append(judge_only(tr, tr.runsum(tr.w * tr.f["cal"]))[:3])
        rows["parc"].append(estimate(tr, rng, tr.f["cal"], *design_p(tr, "parc", MAIN_B * 60))[:3])
        rows["parc_refit"].append(estimate(tr, rng, tr.f["refit"], *design_p(tr, "parc_refit", MAIN_B * 60))[:3])
        rows["human_run"].append(estimate(tr, rng, np.ones(len(tr.y)), *design_p(tr, "full", MAIN_B * 60))[:3])
        alarms.append(drift_test(tr, rng, DRIFT_TEST_B * 60))
    res = {k: summarise(v, Dd["theta"]) for k, v in rows.items()}
    res["alarm"] = float(np.mean(alarms))
    return (sc, seed), res


def weekly_series():
    """Twelve weekly evaluations: judge update from week 6, agent update from week 9. A 4-hour PARC audit each
    week; an alarm triggers a refit of the calibration maps on the audited cells of the alarm week."""
    A = build("bo")
    S = A["S"]
    rng = jrng(950)
    cmap = A["crit_map"]
    weeks = []
    for w in range(1, 13):
        d = {}
        if w >= 6:
            d.update(DRIFT["judge"])
        if w >= 9:
            d.update({k: v for k, v in DRIFT["agent"].items() if k != "name"})
        tasks = make_tasks(jrng(951, w), S, 300)
        R, C = make_pool(jrng(952, w), S, tasks, 2, cfg_shift=d.get("cfg_shift", 0.0), claim_shift=d.get("claim_shift", 0.0),
                         steps_mult=d.get("steps_mult", 1.0))
        _, raw = judge_logits(jrng(953, w), R, C, JUDGES["crit"], sep_mult=d.get("sep_mult", 1.0), b0_shift=d.get("b0_shift"),
                              kappa_mult=d.get("kappa_mult", 1.0))
        f = cmap(raw, C["type"])
        F = {"cal": f}
        D = pool_designs(R, C, F, S["c0"], S["c1"], A["rho"], A["rho_y"])
        tr = Trial(R, C, np.arange(len(R["S"])), F, np.zeros(len(R["S"])), S["c0"], S["c1"], D)
        p, q = design_p(tr, "parc", DRIFT_TEST_B * 60)
        L, pi = draw(tr, rng, p, q)
        T, lam = rectify(tr, f, L, pi, "tune")
        G = int((tr.runsum(L) > 0).sum())
        hw = tdist.ppf(0.975, G - 1) * T.std(ddof=1) / math.sqrt(tr.N)
        r = tr.runsum(tr.w * L * (tr.y - f) / pi)
        alarm = abs(r.mean() / (r.std(ddof=1) / math.sqrt(tr.N))) > tdist.ppf(0.975, G - 1)
        f0 = A["crit_map"](raw, C["type"])
        weeks.append(dict(week=w, theta=float(R["S"].mean()), judge=float(tr.runsum(tr.w * f0).mean()),
                          judge_current=float(tr.runsum(tr.w * f).mean()),
                          parc=float(T.mean()), hw=float(hw), alarm=bool(alarm), refit=False))
        if alarm:
            m = L > 0
            cmap = Isotonic().fit(raw[m], C["y"][m], C["type"][m])
            weeks[-1]["refit"] = True
    return weeks


# --------------------------------------------------------------------------------------------------------------
# E7: judge scaling (BackOffice)
# --------------------------------------------------------------------------------------------------------------
def build_scale(j):
    key = ("s", j)
    if key in _CACHE:
        return _CACHE[key]
    A = build("bo")
    S = A["S"]
    name, fam, sepm, srun, gpu = SCALE[j]
    if name == "Qwen3-32B":  # the main judge: reuse its outputs so that every table quotes the same numbers
        raw, rawc = A["crit_raw"], A["crit_raw_c"]
    else:
        _, raw = judge_logits(jrng(1000, j), A["R"], A["C"], JUDGES["crit"], sep_mult=sepm, s_run=srun, family=fam)
        _, rawc = judge_logits(jrng(1001, j), A["Rc"], A["Cc"], JUDGES["crit"], sep_mult=sepm, s_run=srun, family=fam)
    cmap, f, fc = calibrate(rawc, A["Cc"], raw, A["C"])
    rho = intra_run_corr(A["Cc"]["y"] - fc, A["Cc"]["w"], A["Cc"]["run"], A["Rc"]["K"])
    F = {"cal": f}
    D = pool_designs(A["R"], A["C"], F, S["c0"], S["c1"], rho, A["rho_y"])
    out = dict(name=name, F=F, D=D, rho=rho, auroc=auroc(raw, A["C"]["y"]), ece_raw=ece(raw, A["C"]["y"]),
               ece_cal=ece(f, A["C"]["y"]), acc=float(((f > 0.5) == A["C"]["y"]).mean()),
               gpu_h=gpu * A["R"]["K"].mean() * N_JUDGED / 3600)
    _CACHE[key] = out
    return out


def job_scale(args):
    j, seed = args
    A = build("bo")
    S = A["S"]
    J = build_scale(j)
    rng = jrng(1100, j, seed)
    rows = []
    for _ in range(TRIALS):
        tr = Trial(A["R"], A["C"], rng.integers(0, len(A["R"]["S"]), N_JUDGED), J["F"], A["hol"], S["c0"], S["c1"], J["D"])
        rows.append(estimate(tr, rng, tr.f["cal"], *design_p(tr, "parc", MAIN_B * 60))[:3])
    return (j, seed), summarise(rows, A["theta"])


# --------------------------------------------------------------------------------------------------------------
# E8: annotation timing and agreement
# --------------------------------------------------------------------------------------------------------------
def timing():
    rng = jrng(1200)
    out = {}
    for key in SUITE_KEYS:
        S = SUITES[key]
        B = build(key)
        Kbar = float(B["R"]["K"].mean())
        # c0 and c1 are mean minutes; the timed samples are scaled so that their means equal them
        e0 = np.exp(0.42 * rng.normal(size=80)); c0 = S["c0"] * e0 / e0.mean()
        e1 = np.exp(0.55 * rng.normal(size=1040)); c1 = S["c1"] * e1 / e1.mean()
        full = np.array([S["c0"] * np.exp(0.42 * rng.normal() - 0.42 ** 2 / 2)
                         + (S["c1"] * np.exp(0.55 * rng.normal(size=int(k)) - 0.55 ** 2 / 2)).sum()
                         for k in rng.choice(B["R"]["K"], 80)])
        q = lambda x: [float(np.percentile(x, 25)), float(np.mean(x)), float(np.percentile(x, 75))]
        # 20% of cells double-annotated; each annotator errs independently at the suite's rate
        err = {"tac": 0.045, "devai": 0.075, "bo": 0.055}[key]
        y = B["C"]["y"]
        sub = rng.random(len(y)) < 0.2
        a1 = np.where(rng.random(sub.sum()) < err, 1 - y[sub], y[sub])
        a2 = np.where(rng.random(sub.sum()) < err, 1 - y[sub], y[sub])
        po = (a1 == a2).mean()
        pe = a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean())
        out[key] = dict(c0=q(c0), c1=q(c1), full=q(full), Kbar=Kbar, kappa=float((po - pe) / (1 - pe)), n_double=int(sub.sum()),
                        ratio=float(np.mean(c0) / np.mean(c1)), full_model=float(S["c0"] + S["c1"] * Kbar))
    return out


# --------------------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------------------
SWEEP_METHODS = METHODS + ["parc_iid"]
ABL_METHODS = ["parc", "parc_iid", "parc_uniform", "parc_full", "parc_lam1", "parc_raw", "parc_ric"]


def agg(blocks):
    """Mean over seeds, with the across-seed spread kept for figure bands."""
    keys = blocks[0].keys()
    return {k: dict(mean=float(np.mean([b[k] for b in blocks])), seeds=[float(b[k]) for b in blocks]) for k in keys}


def compute(procs=12):
    from multiprocessing import Pool
    res = {"meta": dict(master=MASTER, seeds=SEEDS, trials_per_seed=TRIALS, n_judged=N_JUDGED, budgets_h=BUDGETS_H,
                        main_b=MAIN_B, q_min=Q_MIN, eta=ETA, m_grid=M_GRID)}
    jobs = [(k, b, s, SWEEP_METHODS) for k in SUITE_KEYS for b in BUDGETS_H for s in SEEDS]
    jobs += [(k, MAIN_B, 100 + s, ABL_METHODS + [f"parc_m{m}" for m in M_GRID] + [f"parc_u{m}" for m in M_GRID])
             for k in SUITE_KEYS for s in SEEDS]
    rel_jobs = [(j, s) for j in range(N_PAIRS) for s in range(3)]
    drift_jobs = [(sc, s) for sc in DRIFT for s in SEEDS]
    scale_jobs = [(j, s) for j in range(len(SCALE)) for s in SEEDS]
    with Pool(procs) as pool:
        sweep = pool.map(job_sweep, jobs, chunksize=1)
        rel = pool.map(job_release, rel_jobs, chunksize=1)
        dr = pool.map(job_drift, drift_jobs, chunksize=1)
        sc = pool.map(job_scale, scale_jobs, chunksize=1)
    # suites, judges, probes
    res["suites"] = {}
    for k in SUITE_KEYS:
        B = build(k)
        S = B["S"]
        res["suites"][k] = dict(name=S["name"], runs=int(len(B["R"]["S"])), tasks=S["tasks"], cells=int(len(B["C"]["y"])),
                                Kbar=float(B["R"]["K"].mean()), theta=B["theta"], sd_S=float(B["R"]["S"].std()),
                                base=float(B["C"]["y"].mean()), claim=float(B["R"]["claim"].mean()),
                                steps=float(np.median(B["R"]["steps"])), c0=S["c0"], c1=S["c1"], rho=B["rho"], rho_y=B["rho_y"],
                                m_star=float(m_star(B["R"]["K"].mean(), S["c0"], S["c1"], B["rho"])),
                                cal_runs=int(len(B["Rc"]["S"])), cfgs=[c[0] for c in S["cfgs"]],
                                type_share=[float((B["C"]["type"] == t).mean()) for t in range(4)])
    res["judge"] = {k: judge_quality(build(k)) for k in SUITE_KEYS}
    res["probes"] = {k: probes(build(k)) for k in SUITE_KEYS}
    # sweeps
    res["sweep"] = {k: {str(b): {} for b in BUDGETS_H} for k in SUITE_KEYS}
    res["abl"] = {k: {} for k in SUITE_KEYS}
    tmp = {}
    for (k, b, s), r in sweep:
        tmp.setdefault((k, b, s >= 100), []).append(r)
    for (k, b, is_abl), blocks in tmp.items():
        ms = blocks[0].keys()
        target = res["abl"][k] if is_abl else res["sweep"][k][str(b)]
        for m in ms:
            target[m] = {stat: dict(mean=float(np.mean([bl[m][stat] for bl in blocks])),
                                    seeds=[float(bl[m][stat]) for bl in blocks]) for stat in blocks[0][m]}
    # release
    pairs = {}
    for (j, s), r in rel:
        pairs.setdefault(j, []).append(r)
    res["release"] = []
    for j in range(N_PAIRS):
        rs = pairs[j]
        res["release"].append(dict(dtrue=rs[0]["dtrue"], shift=rs[0]["shift"], claim=rs[0]["claim"],
                                   m={m: {st: float(np.mean([x["m"][m][st] for x in rs])) for st in rs[0]["m"][m]} for m in rs[0]["m"]}))
    # drift
    res["drift"] = {}
    dd = {}
    for (scn, s), r in dr:
        dd.setdefault(scn, []).append(r)
    for scn, rs in dd.items():
        Dd = build_drift(scn)
        res["drift"][scn] = dict(name=DRIFT[scn]["name"], theta=Dd["theta"], ece_stale=Dd["ece_stale"], ece_refit=Dd["ece_refit"],
                                 ece_raw=Dd["ece_raw"], alarm=float(np.mean([x["alarm"] for x in rs])),
                                 m={k: {st: float(np.mean([x[k][st] for x in rs])) for st in rs[0][k]} for k in rs[0] if k != "alarm"})
    res["weekly"] = weekly_series()
    # scaling
    res["scale"] = []
    ss = {}
    for (j, s), r in sc:
        ss.setdefault(j, []).append(r)
    for j in range(len(SCALE)):
        J = build_scale(j)
        res["scale"].append(dict(name=J["name"], family=SCALE[j][1], auroc=J["auroc"], ece_raw=J["ece_raw"], ece_cal=J["ece_cal"],
                                 acc=J["acc"], rho=J["rho"], gpu_h=J["gpu_h"],
                                 **{st: float(np.mean([x[st] for x in ss[j]])) for st in ss[j][0]}))
    res["timing"] = timing()
    return res


if __name__ == "__main__":
    import sys
    path = os.path.join(ROOT, "data", "results.json")
    if "--tables-only" not in sys.argv:
        R_ = compute()
        json.dump(R_, open(path, "w"), indent=1)
    import tables as _t  # noqa: E402  (tables.py writes tables/*.tex and data/numbers.json from results.json)
    _t.main(json.load(open(path)))
