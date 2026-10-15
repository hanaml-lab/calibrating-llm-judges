# Toy run (real output)

A real run of the reference implementation on its CPU toy setting. Unlike the report's tables, which come from a
seeded simulation and are illustrative, every number below was printed by the code. The toy is small by design: it
checks that the method works end to end on a problem with a known answer; it does not reproduce the report's
experiments.

- Command: `python -m parc.demo` (settings in `configs/toy.yaml`)
- Code: commit `dfba582`
- Environment: Python 3.12.13, numpy 2.5.4, scipy 1.18.1, PyYAML 6.0.3; Linux x86_64, Intel(R) Core(TM) i9-14900KF
- Wall time: 32 s on a shared machine under heavy load (1-minute load average 155 on 32 hardware threads); CPU time 34 s

## The toy setting

`parc.toy` makes a suite of long runs graded against weighted rubrics of 3 to 20 criteria of four types. The judge
is overconfident and lenient, its errors share a cause within a run, and it favours communication criteria when a
run claims success. It is calibrated, one isotonic map per criterion type, on 160 runs of an earlier and weaker
agent version that people checked in full. Each of 200 audits draws 2,000 judged runs and spends 480 annotator
minutes in expectation (5.8 minutes to understand a run, 1.4 per criterion checked) three ways: people grading
whole runs, the calibrated judge alone with a naive interval, and PARC's two-stage audit and prediction-powered
estimate. Coverage is the share of the 200 intervals that contain the true mean partial credit.

## Output

```
true mean partial credit 54.0 points; residual correlation rho = 0.23
200 audits of 2000 judged runs, 480 annotator minutes each
method                   estimate    +/-  coverage  minutes
full-run grading             53.6   13.1     92.0%      480
calibrated judge alone       49.4    1.2      0.0%        0
PARC                         53.8    5.0     94.5%      480
done in 26.3s
```

## What it shows

- PARC's interval is about 2.6 times narrower than grading whole runs at the same budget (plus or minus 5.0 points
  against 13.1) and contains the true value in 94.5% of audits, close to the nominal 95%; full-run grading covers
  92.0%.
- The calibrated judge alone is narrow (plus or minus 1.2) and wrong: it estimates 49.4 points against a true 54.0,
  so its interval never covers. Its calibration set comes from an earlier, weaker agent version.
- The judge's residual errors are correlated within a run (rho = 0.23), and PARC's interval is computed over runs
  rather than over criteria.
