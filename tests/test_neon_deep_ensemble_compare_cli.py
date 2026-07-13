from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "neon_deep_ensemble_compare.py"
    spec = importlib.util.spec_from_file_location("neon_deep_ensemble_compare", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_de_compare_dry_run_is_parameterized_and_torch_free(tmp_path, capsys):
    script = _load_script()
    rc = script.main(
        [
            "--config",
            "/cfg.yaml",
            "--stage2-checkpoint",
            "/stage2.pt",
            "--stage1-bundle",
            "/bundle.json",
            "--output-dir",
            str(tmp_path),
            "--m-eval",
            "16",
            "--k-neon",
            "50",
            "--k-de",
            "50",
            "--dry-run",
        ]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["m_eval"] == 16
    assert plan["k_neon"] == 50
    assert plan["k_de"] == 50
    assert plan["common_aleatory_latent_bank"] is True
    assert plan["physical_space"] is True


def test_de_compare_rejects_mismatched_aleatory_bank_sizes(tmp_path):
    script = _load_script()
    with pytest.raises(ValueError, match="identical aleatory bank sizes"):
        script.main(
            [
                "--config",
                "/cfg.yaml",
                "--stage2-checkpoint",
                "/stage2.pt",
                "--stage1-bundle",
                "/bundle.json",
                "--output-dir",
                str(tmp_path),
                "--k-neon",
                "50",
                "--k-de",
                "20",
                "--dry-run",
            ]
        )


def test_ensemble_mean_absolute_error_uses_each_method_prediction():
    torch = pytest.importorskip("torch")
    script = _load_script()
    reference = torch.tensor([[[[[2.0]]], [[[4.0]]]]])
    prediction = torch.tensor([[[[[1.0]]], [[[3.0]]]]])
    error = script.ensemble_mean_absolute_error(
        prediction, reference, ensemble_dims=(1,)
    )
    assert torch.equal(error, torch.tensor([[[[1.0]]]]))
