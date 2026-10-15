"""PARC on the CPU toy setting: calibrate the judge, plan and draw audits, and compare PARC with full-run grading and
with the calibrated judge alone at the same expected human budget.

    python -m parc.demo [--config configs/toy.yaml]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import yaml

from .toy import ToyConfig, compare

NAMES = {"full_runs": "full-run grading", "judge_only": "calibrated judge alone", "parc": "PARC"}


def config_path(name: str) -> Path:
    """``name`` from the working directory, else from the checkout this package is installed from."""
    path = Path(name)
    if not path.exists() and (Path(__file__).resolve().parents[2] / name).exists():
        return Path(__file__).resolve().parents[2] / name
    return path


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/toy.yaml")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(config_path(args.config).read_text())
    toy = ToyConfig(**{f.name: cfg[f.name] for f in fields(ToyConfig) if f.name in cfg})
    t0 = time.time()
    res = compare(toy, cfg["budget_minutes"], cfg["trials"], cfg["seed"])
    print(f"true mean partial credit {100 * res['truth']:.1f} points; residual correlation rho = {res['rho']:.2f}")
    print(f"{cfg['trials']} audits of {toy.n_judged} judged runs, {cfg['budget_minutes']:.0f} annotator minutes each")
    print(f"{'method':<24}{'estimate':>9}{'+/-':>7}{'coverage':>10}{'minutes':>9}")
    for key in ("full_runs", "judge_only", "parc"):
        r = res[key]
        print(f"{NAMES[key]:<24}{100 * r['estimate']:>9.1f}{100 * r['half_width']:>7.1f}{100 * r['coverage']:>9.1f}%{r['minutes']:>9.0f}")
    print(f"done in {time.time() - t0:.1f}s")
    return res


if __name__ == "__main__":
    main()
