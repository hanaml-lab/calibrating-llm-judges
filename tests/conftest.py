import numpy as np
import pytest


def population(rng, n_runs=800, k=8, run_err=0.8, judge_shift=0.0):
    """Runs with k equally weighted criteria; labels driven by run quality; a judge whose logit error has a
    run-shared part (so its residuals are correlated within a run) and an optional constant shift."""
    run = np.repeat(np.arange(n_runs), k)
    z = rng.normal(0, 1, n_runs)
    ease = rng.normal(0, 1, n_runs * k)
    y = (rng.random(n_runs * k) < 1 / (1 + np.exp(-(1.5 * z[run] + ease)))).astype(float)
    logit = 2.4 * (2 * y - 1) / 2 + rng.normal(0, run_err, n_runs)[run] + rng.normal(0, 0.7, n_runs * k) + judge_shift
    f = 1 / (1 + np.exp(-logit))
    w = np.full(n_runs * k, 1 / k)
    return dict(run=run, y=y, f=f, w=w, n=n_runs, S=np.bincount(run, weights=w * y))


@pytest.fixture
def make_population():
    return population
