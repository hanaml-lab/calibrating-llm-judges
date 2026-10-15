from pathlib import Path

import yaml

from parc.demo import main

ROOT = Path(__file__).resolve().parents[1]


def test_demo_reads_its_config(tmp_path, capsys):
    cfg = yaml.safe_load((ROOT / "configs" / "toy.yaml").read_text())
    cfg.update(trials=10, n_tasks=60, n_judged=300)
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump(cfg))
    res = main(["--config", str(path)])
    assert set(res) >= {"truth", "rho", "full_runs", "judge_only", "parc"}
    assert "PARC" in capsys.readouterr().out
