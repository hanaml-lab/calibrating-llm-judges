"""Run the CPU toy setting: calibrate the judge, plan and draw audits, and compare PARC with full-run grading and
with the calibrated judge alone at the same expected human budget."""
import argparse
import time

from parc.toy import ToyConfig, compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-minutes", type=float, default=480)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    res = compare(ToyConfig(), args.budget_minutes, args.trials, args.seed)
    print(f"truth {100 * res['truth']:.1f} points; residual correlation rho = {res['rho']:.2f}")
    print(f"{'method':<14}{'estimate':>10}{'+/-':>8}{'coverage':>10}{'minutes':>9}")
    for k in ("full_runs", "judge_only", "parc"):
        r = res[k]
        print(f"{k:<14}{100 * r['estimate']:>10.1f}{100 * r['half_width']:>8.1f}{100 * r['coverage']:>9.1f}%{r['minutes']:>9.0f}")
    print(f"({args.trials} trials in {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
