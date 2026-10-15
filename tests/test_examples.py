import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "data"


def _example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_plans_within_budget_and_estimates_with_an_interval(tmp_path):
    audit = _example("audit")
    plan = audit.main(["plan", str(DATA / "judged.csv"), str(DATA / "calibration.csv"), "--budget", "240",
                       "--out", str(tmp_path / "plan.csv")])
    assert 0 < plan["labelled"] and 0 < plan["runs"] <= 60
    # the bundled verdicts answer exactly the cells this plan draws
    assert (tmp_path / "plan.csv").read_text().splitlines()[1:] == [
        line.rsplit(",", 1)[0] for line in (DATA / "audited.csv").read_text().splitlines()[1:]]
    est = audit.main(["estimate", str(DATA / "judged.csv"), str(DATA / "calibration.csv"), str(DATA / "audited.csv")])
    assert 0.0 < est["half_width"] < 0.2
    assert abs(est["value"] - 0.497) < est["half_width"]  # the sample's true mean partial credit is 49.7 points
