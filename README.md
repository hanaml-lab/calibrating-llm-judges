# parc: Prediction-Powered Audits of Rubric Criteria

Reference implementation of the hanaML report **Judging Long Agent Runs with Valid Error Bars: Prediction-Powered Audits of Rubric Criteria** (Agarwal, 2026).

Long agent runs are graded against rubrics, usually by an LLM judge. PARC turns the judge's per-criterion verdicts and a few hours of human checking into an estimate of mean partial credit with a confidence interval that is valid whatever the judge's quality:

1. **Criterion judge.** One call per criterion, on evidence retrieved for it, read out as P(yes) and calibrated per criterion type with isotonic regression (`parc.evidence`, `parc.judge`, `parc.calibration`).
2. **Two-stage audit.** People check a few criteria in each of a few runs. The number per audited run is `m* = sqrt((c0/c1)(1 - rho)/rho)`, where `c0` is the time to understand a run, `c1` the time per check and `rho` the intra-run correlation of the judge's residuals. Runs and criteria are chosen by Neyman allocation on the calibrated judge's uncertainty (`parc.design`).
3. **Estimator.** A Horvitz-Thompson prediction-powered estimator with PPI++ power tuning and a run-level (cluster-robust) interval with a t reference (`parc.estimator`).
4. **Gates and drift.** Paired comparison of two agent versions for a three-way release gate, and a residual test that flags a stale calibration map (`parc.estimator.paired_estimate`, `parc.drift`).

| Module | What it does |
| --- | --- |
| `parc.calibration` | Per-type isotonic calibration (pool-adjacent violators) and binned ECE. |
| `parc.design` | `optimal_criteria_per_run`, `plan_audit` (active two-stage plan that spends a budget in expectation), `plan_paired`, `draw_audit` (systematic sampling of a fixed number of criteria per run). |
| `parc.estimator` | `estimate` (run-level interval; `variance="cells"` shows what treating criteria as independent costs), `paired_estimate`. |
| `parc.drift` | `residual_test`: is the judge's error against people still zero on average? |
| `parc.evidence` | BM25 plus locator retrieval of the steps that bear on one criterion. |
| `parc.judge` | Criterion prompt and P(yes) read-out for any OpenAI-compatible endpoint (for example vLLM). |
| `parc.toy` | A CPU toy suite that runs the whole pipeline in seconds. |

## Install

```bash
pip install -e ".[dev]"           # numpy, scipy, pytest
pip install -e ".[judge,paper]"   # optional: OpenAI-compatible client, matplotlib
```

## Quickstart

```bash
python scripts/run_toy.py          # calibrate, plan, audit and compare at 8 annotator-hours
pytest -q                          # 26 tests
bash scripts/reproduce_paper.sh    # regenerate every table and figure in the report
```

On the toy suite, at 480 annotator-minutes, PARC estimates mean partial credit to within ±5.0 points with 94.5% coverage, against ±13.1 points (92.0%) for grading whole runs; the calibrated judge alone is off by 4.6 points and never covers.

```python
import numpy as np
from parc.calibration import IsotonicCalibrator
from parc.design import plan_audit, draw_audit, residual_correlation
from parc.estimator import estimate

# raw: judge P(yes) per (run, criterion) cell; kind: criterion type; w: weights summing to 1 per run; run: run index
cal = IsotonicCalibrator().fit(raw_cal, y_cal, kind_cal)               # a calibration set graded by people
rho = residual_correlation(y_cal - cal.predict(raw_cal, kind_cal), w_cal, run_cal)
f = cal.predict(raw, kind)
plan = plan_audit(f, w, run, budget=480, c0=5.8, c1=1.4, rho=rho)
labelled, pi = draw_audit(plan, run, np.random.default_rng(0))          # send these cells to people
est = estimate(f, w, run, labelled, y, pi)                              # y: their verdicts on the labelled cells
print(f"{est.value:.3f} ± {est.half_width:.3f}")
```

## Results (from the report)

Mean partial credit at 8 annotator-hours; coverage of nominal 95% intervals and half-width in points; × is the human time full-run grading needs to match, divided by 8. Results come from a seeded simulation study calibrated to public figures and are illustrative.

| Estimator | TAC cov. | TAC ± | DevAI cov. | DevAI ± | BackOffice cov. | BackOffice ± | × |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full-run grading | 92.7 | 11.5 | 93.6 | 12.8 | 92.7 | 13.9 | 1.0 |
| Holistic judge only | 0.0 | 1.3 | 0.0 | 1.3 | 0.0 | 1.2 | – |
| PPI++, criterion judge, full runs | 95.0 | 6.3 | 94.8 | 7.2 | 94.8 | 7.0 | 3.3 |
| Active inference, full runs | 93.8 | 5.6 | 95.2 | 6.5 | 96.4 | 6.4 | 4.1 |
| **PARC** | **94.9** | **4.9** | **94.9** | **5.8** | **95.4** | **4.8** | **6.0** |

Treating PARC's audit as independent criteria covers at only 81–88%. Release gates built on PARC ship 2.1% of regressions at 8 hours, against 24–33% for judge-only gates.

## Citation

```bibtex
@techreport{hanaml2026calibratingllmjudges,
  title       = {Judging Long Agent Runs with Valid Error Bars: Prediction-Powered Audits of Rubric Criteria},
  author      = {Manav Agarwal},
  institution = {hanaML},
  type        = {Technical report},
  year        = {2026},
  month       = jan,
  url         = {https://hanaml.com/research/papers/calibrating-llm-judges}
}
```

Apache-2.0. See `LICENSE`.
