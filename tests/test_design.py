import numpy as np
import pytest

from parc.design import draw_audit, optimal_criteria_per_run, plan_audit


def test_optimal_criteria_per_run_worked_example():
    # c0/c1 = 4 and (1 - rho)/rho = 4 give m* = sqrt(16) = 4.
    assert optimal_criteria_per_run(6.0, 1.5, 0.2) == pytest.approx(4.0)


def test_optimal_criteria_per_run_respects_the_rubric_size_and_checks_at_least_one():
    assert optimal_criteria_per_run(6.0, 1.5, 0.2, K=3) == pytest.approx(3.0)
    assert optimal_criteria_per_run(6.0, 1.5, 0.95) == pytest.approx(1.0)


def test_independent_errors_make_it_cheapest_to_check_everything_in_an_opened_run():
    assert optimal_criteria_per_run(6.0, 1.5, 0.0, K=12) == pytest.approx(12.0)


def test_plan_spends_the_budget_in_expectation(make_population):
    pop = make_population(np.random.default_rng(0), n_runs=500)
    plan = plan_audit(pop["f"], pop["w"], pop["run"], budget=240.0, c0=6.0, c1=1.5, rho=0.3)
    assert (plan.p * plan.cost).sum() == pytest.approx(240.0, rel=1e-6)


def test_within_run_probabilities_sum_to_the_planned_number_of_checks(make_population):
    pop = make_population(np.random.default_rng(1), n_runs=200)
    plan = plan_audit(pop["f"], pop["w"], pop["run"], budget=120.0, c0=6.0, c1=1.5, rho=0.3, q_min=0.05)
    lo = np.bincount(pop["run"], weights=plan.q_lo)
    hi = np.bincount(pop["run"], weights=plan.q_hi)
    np.testing.assert_allclose(lo, np.floor(plan.m + 1e-9), atol=1e-6)
    np.testing.assert_allclose(hi, np.minimum(np.floor(plan.m + 1e-9) + 1, 8), atol=1e-6)
    assert plan.q_lo.min() >= 0.05 - 1e-12 and plan.q_hi.max() <= 1 + 1e-12


def test_active_plan_checks_uncertain_criteria_more_often():
    f = np.array([0.5, 0.98, 0.02, 0.6])
    plan = plan_audit(f, np.full(4, 0.25), np.zeros(4, int), budget=10.0, c0=6.0, c1=1.5, rho=0.3, m=2.0)
    assert plan.q_lo[0] > plan.q_lo[1] and plan.q_lo[3] > plan.q_lo[2]


def test_drawn_audits_have_the_planned_inclusion_probabilities(make_population):
    rng = np.random.default_rng(2)
    pop = make_population(rng, n_runs=40)
    plan = plan_audit(pop["f"], pop["w"], pop["run"], budget=200.0, c0=6.0, c1=1.5, rho=0.3)
    hits = np.zeros(len(pop["y"]))
    pis = np.zeros(len(pop["y"]))
    sizes = set()
    for _ in range(6000):
        lab, pi = draw_audit(plan, pop["run"], rng)
        hits += lab
        pis += pi
        per_run = np.bincount(pop["run"], weights=lab)
        sizes.update(per_run[per_run > 0].astype(int).tolist())
    assert np.abs(hits / 6000 - pis / 6000).max() < 0.03
    assert sizes <= {int(np.floor(plan.m.min())), int(np.ceil(plan.m.max()))}
