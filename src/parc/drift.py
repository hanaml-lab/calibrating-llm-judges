"""A residual test for drift in the judge's error against people.

On an audit, r_i = sum_k w_ik L_ik (Y_ik - f_ik) / pi_ik has mean E[Y - f] summed over a run's weights, which is zero
when the calibration map still fits. A two-sided t test of zero flags a stale map. A flag costs efficiency, not
validity: PARC's rectifier uses the audit's own labels, so its interval stays valid with a stale map, only wider.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import t as student_t

from .design import Plan, draw_audit


@dataclass(frozen=True)
class DriftTest:
    mean_residual: float
    t_stat: float
    alarm: bool
    audited_runs: int


def residual_test(f, w, run, y, plan: Plan, rng: np.random.Generator, *, alpha: float = 0.05, audit=None) -> DriftTest:
    """Test E[Y - f] = 0 from one audit. Pass `audit=(labelled, pi)` to reuse an audit already drawn."""
    f = np.asarray(f, float)
    w = np.asarray(w, float)
    run = np.asarray(run, int)
    y = np.asarray(y, float)
    n = len(plan.p)
    labelled, pi = draw_audit(plan, run, rng) if audit is None else audit
    L = np.asarray(labelled, float)
    r = np.bincount(run, weights=w * L * (np.where(labelled, y, 0.0) - f) / pi, minlength=n)
    G = int((np.bincount(run, weights=L, minlength=n) > 0).sum())
    se = r.std(ddof=1) / np.sqrt(n)
    t_stat = float(r.mean() / se) if se > 0 else 0.0
    alarm = abs(t_stat) > student_t.ppf(1 - alpha / 2, max(G - 1, 1))
    return DriftTest(float(r.mean()), t_stat, bool(alarm), G)
