"""The PARC estimator: a Horvitz-Thompson prediction-powered estimate of mean partial credit with a run-level
(cluster-robust) interval.

For run i with criteria k, weights w_ik, calibrated judge probabilities f_ik, human labels Y_ik on the audited cells
(L_ik = 1) and inclusion probabilities pi_ik = p_i q_ik,

    T_i(lam) = sum_k w_ik [ lam f_ik + (L_ik / pi_ik) (Y_ik - lam f_ik) ],

which is unbiased for the run's partial credit S_i for every lam. The estimate is the mean of T_i over the N judged
runs, and the interval uses the sample variance of T_i over runs, so everything that is correlated within a run stays
inside T_i (Proposition 1 of the paper).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t


@dataclass(frozen=True)
class Estimate:
    value: float
    half_width: float
    lower: float
    upper: float
    se: float
    lam: float
    audited_runs: int


def _components(f, w, run, labelled, y, pi, n):
    L = labelled.astype(float)
    yl = np.where(labelled, y, 0.0)
    A = np.bincount(run, weights=w * L * yl / pi, minlength=n)
    D = np.bincount(run, weights=w * f * (1.0 - L / pi), minlength=n)
    return A, D, L, yl


def tune_lambda(A: np.ndarray, D: np.ndarray) -> float:
    """PPI++ power tuning: the lam minimising Var(A + lam D), clipped to [0, 1]."""
    vd = D.var()
    if vd <= 1e-15:
        return 1.0
    lam = -np.mean((A - A.mean()) * (D - D.mean())) / vd
    return float(np.clip(lam, 0.0, 1.0))


def estimate(f, w, run, labelled, y, pi, *, alpha: float = 0.05, lam="tune", variance: str = "run",
             n_runs: int | None = None) -> Estimate:
    """Estimate mean partial credit from judge probabilities on every cell and human labels on the audited ones.

    variance="run" is PARC's interval: sample variance of T_i over runs and a t reference on G - 1 degrees of
    freedom (G audited runs). variance="cells" treats every criterion as an independent unit; it is provided only to
    show what that mistake costs.
    """
    f = np.asarray(f, float)
    w = np.asarray(w, float)
    run = np.asarray(run, int)
    labelled = np.asarray(labelled, bool)
    y = np.asarray(y, float)
    pi = np.broadcast_to(np.asarray(pi, float), f.shape)
    n = int(run.max()) + 1 if n_runs is None else n_runs
    A, D, L, yl = _components(f, w, run, labelled, y, pi, n)
    lam_v = tune_lambda(A, D) if lam == "tune" else float(lam)
    T = A + lam_v * D
    value = float(T.mean())
    G = int((np.bincount(run, weights=L, minlength=n) > 0).sum())
    if variance == "run":
        se = float(T.std(ddof=1) / np.sqrt(n))
        crit = float(student_t.ppf(1 - alpha / 2, max(G - 1, 1)))
    elif variance == "cells":
        M = len(f)
        z = (M / n) * w * (lam_v * f + L * (yl - lam_v * f) / pi)
        se = float(z.std(ddof=1) / np.sqrt(M))
        crit = float(norm.ppf(1 - alpha / 2))
    else:
        raise ValueError(f"unknown variance {variance!r}")
    hw = crit * se
    return Estimate(value, hw, value - hw, value + hw, se, lam_v, G)
