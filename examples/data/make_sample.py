"""Write the bundled sample for examples/audit.py from the toy setting (parc.toy), and print its true value.

    python examples/data/make_sample.py
    python examples/audit.py plan examples/data/judged.csv examples/data/calibration.csv --out examples/data/plan.csv
    python examples/data/make_sample.py --answer examples/data/plan.csv   # writes audited.csv from the toy's truth

judged.csv holds 60 runs of a current agent version; calibration.csv holds 40 fully checked runs of an earlier
version. The human column of audited.csv plays the people who check the planned cells.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from parc.toy import KINDS, ToyConfig, make_suite

HERE = Path(__file__).resolve().parent


def suites():
    cfg = ToyConfig()
    rng = np.random.default_rng(2026)
    judged = make_suite(rng, cfg, n_tasks=30, seeds=2)
    calibration = make_suite(rng, cfg, n_tasks=20, seeds=2, quality_shift=cfg.cal_version_shift)
    return judged, calibration


def rows(suite, prefix: str, human: bool):
    K = np.bincount(suite.run)
    start = np.concatenate([[0], np.cumsum(K)[:-1]])
    for i in range(len(suite.run)):
        r = suite.run[i]
        row = {"run": f"{prefix}-{r:03d}", "criterion": f"c{i - start[r] + 1:02d}", "kind": KINDS[suite.kind[i]],
               "weight": f"{suite.w[i]:.6f}", "judge_p": f"{suite.raw[i]:.6f}"}
        if human:
            row["human"] = int(suite.y[i])
        yield row


def write(path: Path, data) -> None:
    data = list(data)
    with open(path, "w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(data[0]))
        out.writeheader()
        out.writerows(data)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--answer", help="a plan written by examples/audit.py; writes audited.csv with the toy's verdicts")
    args = ap.parse_args(argv)
    judged, calibration = suites()
    if args.answer:
        truth = {(r["run"], r["criterion"]): r["human"] for r in rows(judged, "run", human=True)}
        with open(args.answer, newline="") as f:
            plan = list(csv.DictReader(f))
        write(HERE / "audited.csv", ({**p, "human": truth[(p["run"], p["criterion"])]} for p in plan))
        return
    write(HERE / "judged.csv", rows(judged, "run", human=False))
    write(HERE / "calibration.csv", rows(calibration, "cal", human=True))
    print(f"true mean partial credit of judged.csv: {100 * judged.S.mean():.1f} points ({len(judged.S)} runs)")


if __name__ == "__main__":
    main()
