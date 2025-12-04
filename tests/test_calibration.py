import numpy as np

from parc.calibration import IsotonicCalibrator, expected_calibration_error


def test_pool_adjacent_violators_worked_example():
    # Scores in order with labels 0,1,0,1: the middle pair violates monotonicity and is pooled to 0.5.
    cal = IsotonicCalibrator(clip=(0.0, 1.0)).fit(np.array([0.1, 0.2, 0.3, 0.4]), np.array([0, 1, 0, 1]))
    np.testing.assert_allclose(cal.predict(np.array([0.1, 0.2, 0.3, 0.4])), [0.0, 0.5, 0.5, 1.0])


def test_maps_are_fitted_per_criterion_type():
    # Type 0 is overconfident, type 1 underconfident; one map per type recovers both.
    rng = np.random.default_rng(0)
    s = rng.random(20000)
    t = rng.integers(0, 2, 20000)
    true_p = np.where(t == 0, s ** 3, np.sqrt(s))
    y = (rng.random(20000) < true_p).astype(int)
    cal = IsotonicCalibrator().fit(s, y, t)
    f = cal.predict(s, t)
    for ty in (0, 1):
        m = t == ty
        assert expected_calibration_error(s[m], y[m]) > 0.10
        assert expected_calibration_error(f[m], y[m]) < 0.02


def test_calibrated_output_is_monotone_in_the_score_within_a_type():
    rng = np.random.default_rng(1)
    s = rng.random(5000)
    y = (rng.random(5000) < s ** 2).astype(int)
    cal = IsotonicCalibrator().fit(s, y)
    grid = np.linspace(0, 1, 101)
    assert np.all(np.diff(cal.predict(grid)) >= 0)


def test_calibrated_probabilities_stay_inside_the_clip_range():
    cal = IsotonicCalibrator(clip=(0.01, 0.99)).fit(np.array([0.1, 0.9]), np.array([0, 1]))
    out = cal.predict(np.array([0.0, 0.1, 0.9, 1.0]))
    assert out.min() >= 0.01 and out.max() <= 0.99


def test_ece_of_a_perfectly_calibrated_predictor_is_near_zero():
    rng = np.random.default_rng(2)
    p = rng.random(200000)
    y = (rng.random(200000) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.01
