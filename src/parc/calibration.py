"""Per-criterion-type isotonic calibration of judge scores and binned expected calibration error."""
from __future__ import annotations

import numpy as np


def _pav(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: right ends of the fitted blocks and their values."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order], y[order].astype(float)
    vals: list[float] = []
    wts: list[float] = []
    ends: list[float] = []
    for xi, yi in zip(xs, ys):
        vals.append(yi)
        wts.append(1.0)
        ends.append(xi)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals.pop(); wts.pop(); e = ends.pop()
            vals[-1], wts[-1], ends[-1] = v, w, e
    return np.asarray(ends), np.asarray(vals)


class IsotonicCalibrator:
    """Monotone map from a judge's raw score to P(criterion met), fitted separately for each criterion type.

    Calibration is not needed for the validity of PARC's interval; it makes the judge a better control variate and
    makes f(1 - f) a usable guide to where people should look.
    """

    def __init__(self, clip: tuple[float, float] = (0.01, 0.99)):
        self.clip = clip
        self.maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, scores, labels, types=None) -> "IsotonicCalibrator":
        s = np.asarray(scores, float)
        y = np.asarray(labels)
        t = np.zeros(len(s), int) if types is None else np.asarray(types)
        for ty in np.unique(t):
            m = t == ty
            ends, vals = _pav(s[m], y[m])
            self.maps[int(ty)] = (ends, np.clip(vals, *self.clip))
        return self

    def predict(self, scores, types=None) -> np.ndarray:
        s = np.asarray(scores, float)
        t = np.zeros(len(s), int) if types is None else np.asarray(types)
        out = np.empty_like(s)
        for ty in np.unique(t):
            if int(ty) not in self.maps:
                raise KeyError(f"no calibration map for criterion type {ty}")
            ends, vals = self.maps[int(ty)]
            m = t == ty
            idx = np.searchsorted(ends, s[m], side="left")
            out[m] = vals[np.clip(idx, 0, len(vals) - 1)]
        return out


def expected_calibration_error(probs, labels, bins: int = 10) -> float:
    """Binned ECE with equal-width bins: sum_b (n_b / n) |mean p_b - mean y_b|."""
    p = np.asarray(probs, float)
    y = np.asarray(labels, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(p))
