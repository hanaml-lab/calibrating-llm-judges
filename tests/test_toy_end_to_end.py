import time

import numpy as np

from parc.toy import ToyConfig, compare, make_suite


def test_toy_suite_has_rubrics_labels_and_judge_scores():
    suite = make_suite(np.random.default_rng(0), ToyConfig(n_tasks=50))
    assert len(suite.y) == len(suite.raw) == len(suite.run)
    assert set(np.unique(suite.y)) <= {0, 1}
    np.testing.assert_allclose(np.bincount(suite.run, weights=suite.w), 1.0)


def test_parc_beats_full_run_grading_at_equal_budget_and_keeps_coverage():
    start = time.time()
    res = compare(ToyConfig(), budget_minutes=480, trials=150, seed=0)
    assert time.time() - start < 60
    assert res["parc"]["half_width"] < 0.6 * res["full_runs"]["half_width"]
    assert res["parc"]["coverage"] >= 0.88
    assert res["judge_only"]["coverage"] < 0.5
