"""Audit your own judged agent runs with PARC: plan which criteria people check, then estimate with an interval.

    python examples/audit.py plan examples/data/judged.csv examples/data/calibration.csv --budget 240 --out plan.csv
    python examples/audit.py estimate examples/data/judged.csv examples/data/calibration.csv examples/data/audited.csv

``judged.csv`` has one row per (run, criterion) cell of the suite you want to score: ``run``, ``criterion``,
``kind`` (the criterion type, which gets its own calibration map), ``weight`` (summing to 1 within a run) and
``judge_p``, the judge's raw P(yes). ``calibration.csv`` has the same columns plus ``human``, people's verdicts on an
earlier, fully checked set of runs; it fits the per-type isotonic calibration and the residual correlation rho.

``plan`` spends a budget of annotator minutes in expectation (``--c0`` minutes to understand a run, ``--c1`` per
criterion checked) and writes the cells to send to people, with their inclusion probabilities. ``estimate`` reads
those cells back with a ``human`` column and prints mean partial credit with a run-level confidence interval, next to
the calibrated judge alone. In the bundled sample, which comes from the toy setting (``parc.toy``), the human
verdicts in ``audited.csv`` are the toy's ground truth for the cells the plan drew.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np

from parc.calibration import IsotonicCalibrator
from parc.design import draw_audit, plan_audit, residual_correlation
from parc.estimator import estimate


def read(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def cells(rows: list[dict]):
    runs = {r: i for i, r in enumerate(dict.fromkeys(row["run"] for row in rows))}
    run = np.array([runs[row["run"]] for row in rows])
    kinds = sorted({row["kind"] for row in rows})
    kind = np.array([kinds.index(row["kind"]) for row in rows])
    w = np.array([float(row["weight"]) for row in rows])
    raw = np.array([float(row["judge_p"]) for row in rows])
    return run, kind, w, raw, kinds


def calibrate(judged: list[dict], calibration: list[dict]):
    run_c, _, w_c, raw_c, _ = cells(calibration)
    kinds = sorted({row["kind"] for row in judged} | {row["kind"] for row in calibration})
    kind_c = np.array([kinds.index(row["kind"]) for row in calibration])
    y_c = np.array([int(row["human"]) for row in calibration])
    cal = IsotonicCalibrator().fit(raw_c, y_c, kind_c)
    rho = residual_correlation(y_c - cal.predict(raw_c, kind_c), w_c, run_c)
    run, _, w, raw, _ = cells(judged)
    kind = np.array([kinds.index(row["kind"]) for row in judged])
    return run, w, cal.predict(raw, kind), rho


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    p = sub.add_parser("plan", help="choose the cells people check")
    p.add_argument("judged")
    p.add_argument("calibration")
    p.add_argument("--budget", type=float, default=240, help="annotator minutes, in expectation")
    p.add_argument("--c0", type=float, default=5.8, help="minutes to understand a run")
    p.add_argument("--c1", type=float, default=1.4, help="minutes per criterion checked")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="plan.csv")
    e = sub.add_parser("estimate", help="estimate mean partial credit from the audited cells")
    e.add_argument("judged")
    e.add_argument("calibration")
    e.add_argument("audited", help="the plan's cells with a human column")
    args = ap.parse_args(argv)

    judged, calibration = read(args.judged), read(args.calibration)
    run, w, f, rho = calibrate(judged, calibration)
    if args.step == "plan":
        plan = plan_audit(f, w, run, args.budget, args.c0, args.c1, rho)
        labelled, pi = draw_audit(plan, run, np.random.default_rng(args.seed))
        with open(args.out, "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["run", "criterion", "pi"])
            for i in np.flatnonzero(labelled):
                out.writerow([judged[i]["run"], judged[i]["criterion"], f"{pi[i]:.6f}"])
        n_runs = len(set(run[labelled]))
        minutes = n_runs * args.c0 + labelled.sum() * args.c1
        print(f"rho = {rho:.2f}; {run.max() + 1} runs, {len(run)} cells; audit {labelled.sum()} cells in {n_runs} runs "
              f"({minutes:.0f} annotator minutes; {args.budget:.0f} expected) -> {args.out}")
        return {"labelled": int(labelled.sum()), "runs": n_runs}

    audited = {(row["run"], row["criterion"]): row for row in read(args.audited)}
    keys = [(row["run"], row["criterion"]) for row in judged]
    labelled = np.array([k in audited for k in keys])
    y = np.array([int(audited[k]["human"]) if k in audited else 0 for k in keys])
    pi = np.array([float(audited[k]["pi"]) if k in audited else 1.0 for k in keys])
    est = estimate(f, w, run, labelled, y, pi)
    judge_only = np.bincount(run, weights=w * f).mean()
    print(f"{run.max() + 1} runs; {labelled.sum()} audited cells in {est.audited_runs} runs")
    print(f"PARC estimate of mean partial credit: {100 * est.value:.1f} +/- {100 * est.half_width:.1f} points (95%)")
    print(f"calibrated judge alone: {100 * judge_only:.1f} points, with no valid interval")
    return {"value": est.value, "half_width": est.half_width, "judge_only": float(judge_only)}


if __name__ == "__main__":
    main()
