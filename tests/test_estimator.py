import numpy as np

from parc.estimator import estimate


def two_stage_bernoulli(rng, run, n, p, q):
    xi = rng.random(n) < p
    labelled = xi[run] & (rng.random(len(run)) < q)
    return labelled, np.full(len(run), p * q)


def test_with_every_criterion_labelled_the_estimate_is_the_mean_partial_credit(make_population):
    pop = make_population(np.random.default_rng(0), n_runs=50)
    est = estimate(pop["f"], pop["w"], pop["run"], np.ones(len(pop["y"]), bool), pop["y"], np.ones(len(pop["y"])))
    assert abs(est.value - pop["S"].mean()) < 1e-12


def test_estimate_is_unbiased_even_when_the_judge_is_biased(make_population):
    rng = np.random.default_rng(1)
    pop = make_population(rng, n_runs=400, judge_shift=1.0)
    truth = pop["S"].mean()
    judge_only = np.bincount(pop["run"], weights=pop["w"] * pop["f"]).mean()
    assert judge_only - truth > 0.05  # the judge alone is clearly biased upwards
    vals = []
    for _ in range(3000):
        lab, pi = two_stage_bernoulli(rng, pop["run"], pop["n"], 0.15, 0.5)
        vals.append(estimate(pop["f"], pop["w"], pop["run"], lab, pop["y"], pi, lam=1.0).value)
    vals = np.array(vals)
    assert abs(vals.mean() - truth) < 3 * vals.std() / np.sqrt(len(vals))


def test_run_level_interval_covers_and_the_independent_cells_interval_does_not(make_population):
    rng = np.random.default_rng(2)
    cover_run, cover_cells = [], []
    for _ in range(400):
        pop = make_population(rng, n_runs=600, run_err=1.2)
        lab, pi = two_stage_bernoulli(rng, pop["run"], pop["n"], 0.08, 0.6)
        truth = 0.5  # labels are symmetric around a logit of zero, so E[S] = 1/2
        a = estimate(pop["f"], pop["w"], pop["run"], lab, pop["y"], pi)
        b = estimate(pop["f"], pop["w"], pop["run"], lab, pop["y"], pi, variance="cells")
        cover_run.append(a.lower <= truth <= a.upper)
        cover_cells.append(b.lower <= truth <= b.upper)
    assert 0.91 <= np.mean(cover_run) <= 0.99
    assert np.mean(cover_cells) < 0.90


def test_power_tuning_is_never_wider_than_a_fixed_weight_and_trusts_a_good_judge(make_population):
    rng = np.random.default_rng(3)
    pop = make_population(rng, n_runs=3000)
    lab, pi = two_stage_bernoulli(rng, pop["run"], pop["n"], 0.2, 0.5)
    for f in (rng.random(len(pop["y"])), pop["f"]):  # a judge that is pure noise, and a good one
        tuned = estimate(f, pop["w"], pop["run"], lab, pop["y"], pi)
        for lam in (0.0, 0.5, 1.0):
            assert tuned.half_width <= estimate(f, pop["w"], pop["run"], lab, pop["y"], pi, lam=lam).half_width + 1e-12
    assert estimate(pop["f"], pop["w"], pop["run"], lab, pop["y"], pi).lam > 0.6


def test_reports_how_many_runs_were_audited(make_population):
    pop = make_population(np.random.default_rng(4), n_runs=10, k=3)
    lab = np.zeros(30, bool)
    lab[[0, 1, 7]] = True  # runs 0 and 2
    est = estimate(pop["f"], pop["w"], pop["run"], lab, pop["y"], np.full(30, 0.3))
    assert est.audited_runs == 2
