"""The audit design: how many criteria to check in an audited run, which runs and which criteria to audit, and the
draw itself.

Costs: understanding a run once costs c0 minutes and each criterion checked costs c1 more. A run is audited with
probability p_i; in an audited run a fixed number M_i of criteria (floor or ceil of m_i, with the probabilities that
make its mean m_i) is drawn by systematic sampling in a random order, with inclusion probabilities q_ik.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def optimal_criteria_per_run(c0: float, c1: float, rho: float, K=None):
    """Proposition 2: m* = sqrt((c0 / c1) (1 - rho) / rho), at least one and at most K.

    rho is the intra-run correlation of the judge's residuals Y - lam f, not of the labels."""
    if rho <= 0:
        m = np.inf
    else:
        m = np.sqrt((c0 / c1) * (1 - rho) / rho)
    m = max(1.0, m)
    if K is None:
        return float(m)
    return np.minimum(np.asarray(K, float), m) if np.ndim(K) else float(min(K, m))


def residual_correlation(resid, w, run) -> float:
    """rho = sum_{k != l} u_k u_l / ((K - 1) sum_k u_k^2), pooled over runs, with u = w * resid."""
    run = np.asarray(run, int)
    u = np.asarray(w, float) * np.asarray(resid, float)
    K = np.bincount(run).astype(float)
    s1 = np.bincount(run, weights=u)
    s2 = np.bincount(run, weights=u * u)
    return float((s1 ** 2 - s2).sum() / ((K - 1) * s2).sum())


@dataclass(frozen=True)
class Plan:
    p: np.ndarray       # per run: probability that the run is audited
    q_lo: np.ndarray    # per cell: inclusion probability when floor(m_i) criteria are drawn
    q_hi: np.ndarray    # per cell: inclusion probability when floor(m_i) + 1 criteria are drawn
    r_hi: np.ndarray    # per run: probability of drawing floor(m_i) + 1
    m: np.ndarray       # per run: expected number of criteria checked when audited
    cost: np.ndarray    # per run: expected minutes when audited
    v: np.ndarray       # per run: predicted conditional variance of T_i (up to the factor 1/p_i)
    eta: float = 0.2


def _within(run, n, a, m, q_min):
    """q = clip(s_i a, q_min, 1) with sum_k q = m_i, by bisection on s_i (vectorised over runs)."""
    lo, hi = np.zeros(n), np.full(n, 1e6)
    for _ in range(80):
        mid = (lo + hi) / 2
        big = np.bincount(run, weights=np.clip(mid[run] * a, q_min, 1), minlength=n) > m
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    q = np.clip(hi[run] * a, q_min, 1)
    tot = np.bincount(run, weights=q, minlength=n)
    return np.minimum(q * (m / tot)[run], 1.0)


def _scale_to_budget(score, cost, budget):
    """p = min(1, s * score) with sum p * cost = budget."""
    lo, hi = 0.0, 1.0
    while (np.minimum(1, hi * score) * cost).sum() < budget and hi < 1e12:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (np.minimum(1, mid * score) * cost).sum() < budget else (lo, mid)
    return np.minimum(1, hi * score)


def plan_audit(f, w, run, budget: float, c0: float, c1: float, rho: float, *, eta: float = 0.2, q_min: float = 0.05,
               active: bool = True, m=None, n_runs: int | None = None) -> Plan:
    """Plan an audit that spends `budget` minutes in expectation.

    Within a run, q is proportional to w * sqrt(f (1 - f)) (Neyman allocation for a calibrated judge); across runs,
    p is proportional to sqrt(V_i / C_i), mixed with the uniform design (weight eta). active=False gives the uniform
    design. These choices affect efficiency only; the estimator is valid for any of them.
    """
    f = np.asarray(f, float)
    w = np.asarray(w, float)
    run = np.asarray(run, int)
    n = int(run.max()) + 1 if n_runs is None else n_runs
    K = np.bincount(run, minlength=n).astype(float)
    if m is None:
        m_run = optimal_criteria_per_run(c0, c1, rho, K)
    else:
        m_run = np.minimum(K, np.broadcast_to(np.asarray(m, float), (n,)))
    m_run = np.maximum(np.minimum(K, m_run), np.minimum(K, 1.0))
    m_lo = np.floor(m_run + 1e-9)
    r_hi = m_run - m_lo
    sig = np.sqrt(np.clip(f * (1 - f), 1e-4, None))
    a = w * sig if active else np.ones_like(f)
    q_lo = _within(run, n, a, m_lo, q_min)
    q_hi = _within(run, n, a, np.minimum(K, m_lo + 1), q_min)
    cost = c0 + c1 * m_run
    uniform = np.minimum(1, np.full(n, budget / cost.sum()))
    q_bar = q_lo * (1 - r_hi[run]) + q_hi * r_hi[run]
    u = w * np.sqrt(np.clip(f * (1 - f), 1e-4, None))
    s2 = np.bincount(run, weights=u * u, minlength=n)
    EU2 = s2 + rho * (np.bincount(run, weights=u, minlength=n) ** 2 - s2)
    within = np.maximum(np.bincount(run, weights=u * u / q_bar, minlength=n) - EU2 / m_run, 0)
    within *= (K - m_run) / np.maximum(K - 1, 1)
    V = within + EU2
    if not active:
        p = uniform
    else:
        p = (1 - eta) * _scale_to_budget(np.sqrt(np.maximum(V, 1e-12) / cost), cost, budget) + eta * uniform
    return Plan(p=p, q_lo=q_lo, q_hi=q_hi, r_hi=r_hi, m=m_run, cost=cost, v=V, eta=eta)


def plan_paired(plan_a: Plan, plan_b: Plan, budget: float) -> np.ndarray:
    """Audit probability per pair of runs (same task and seed under versions A and B), audited together:
    p proportional to sqrt((V_A + V_B) / (C_A + C_B)), mixed with the uniform design, spending `budget`."""
    cost = plan_a.cost + plan_b.cost
    score = np.sqrt(np.maximum(plan_a.v + plan_b.v, 1e-12) / cost)
    uniform = np.minimum(1, np.full(len(cost), budget / cost.sum()))
    eta = plan_a.eta
    return (1 - eta) * _scale_to_budget(score, cost, budget) + eta * uniform


def draw_criteria(plan: Plan, run, rng: np.random.Generator):
    """Which criteria would be checked in each run if it were audited, and their inclusion probabilities
    conditional on the number drawn (systematic sampling in a random order)."""
    run = np.asarray(run, int)
    n = len(plan.p)
    hi = rng.random(n) < plan.r_hi
    q = np.where(hi[run], plan.q_hi, plan.q_lo)
    order = np.lexsort((rng.random(len(q)), run))
    qs = q[order]
    cum = np.cumsum(qs)
    start = np.concatenate([[0.0], cum[:-1]])
    first = np.r_[True, run[order][1:] != run[order][:-1]]
    base = np.maximum.accumulate(np.where(first, start, 0.0))
    upper, lower = cum - base, start - base
    U = rng.random(n)[run[order]]
    chosen = (np.ceil(upper - U - 1e-9) - np.ceil(lower - U - 1e-9)) >= 1
    sel = np.empty(len(q), bool)
    sel[order] = chosen
    return sel, q


def draw_audit(plan: Plan, run, rng: np.random.Generator, p=None):
    """Draw which runs are audited and which criteria are checked in each; returns (labelled, pi) per cell, where pi
    is the inclusion probability conditional on the number of criteria drawn in the run. `p` overrides plan.p (as
    for paired audits)."""
    run = np.asarray(run, int)
    p = plan.p if p is None else np.asarray(p, float)
    xi = rng.random(len(p)) < p
    sel, q = draw_criteria(plan, run, rng)
    return xi[run] & sel, p[run] * q
