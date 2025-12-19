import numpy as np

from parc.calibration import IsotonicCalibrator
from parc.design import plan_audit, plan_paired
from parc.drift import residual_test
from parc.estimator import paired_estimate


def test_paired_difference_is_unbiased_and_covers(make_population):
    rng = np.random.default_rng(0)
    cover, vals = [], []
    for _ in range(300):
        a = make_population(rng, n_runs=400)
        b = make_population(rng, n_runs=400, judge_shift=0.6)  # version B narrates more confidently
        truth = 0.0  # both versions have E[S] = 1/2
        plan_a = plan_audit(a["f"], a["w"], a["run"], 150.0, 6.0, 1.5, 0.3)
        plan_b = plan_audit(b["f"], b["w"], b["run"], 150.0, 6.0, 1.5, 0.3)
        p = plan_paired(plan_a, plan_b, budget=300.0)
        est = paired_estimate((a["f"], a["w"], a["run"], a["y"], plan_a), (b["f"], b["w"], b["run"], b["y"], plan_b), p, rng)
        vals.append(est.value)
        cover.append(est.lower <= truth <= est.upper)
    assert abs(np.mean(vals)) < 3 * np.std(vals) / np.sqrt(len(vals))
    assert np.mean(cover) >= 0.91


def test_residual_test_fires_on_a_shifted_judge_and_rarely_on_a_calibrated_one(make_population):
    rng = np.random.default_rng(1)
    ref = make_population(rng, n_runs=20000)
    cal = IsotonicCalibrator().fit(ref["f"], ref["y"])
    alarms_shift, alarms_ok = [], []
    for _ in range(200):
        pop = make_population(rng, n_runs=600)
        calibrated = cal.predict(pop["f"])
        shifted = np.clip(calibrated + 0.15, 0, 1)
        for f, sink in ((calibrated, alarms_ok), (shifted, alarms_shift)):
            plan = plan_audit(f, pop["w"], pop["run"], 480.0, 6.0, 1.5, 0.3)
            sink.append(residual_test(f, pop["w"], pop["run"], pop["y"], plan, rng).alarm)
    assert np.mean(alarms_shift) > 0.8
    assert np.mean(alarms_ok) < 0.12
