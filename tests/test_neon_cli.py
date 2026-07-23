"""TDD tests for NEON Gap 4: CLI plan resolution + entrypoint dry-run."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    package = sys.modules.setdefault(name, types.ModuleType(name))
    package_path = str(REPO_ROOT / name.replace(".", "/"))
    existing_paths = list(getattr(package, "__path__", []))
    if package_path not in existing_paths:
        existing_paths.append(package_path)
    package.__path__ = existing_paths


def _load_module(name: str, rel_path: str, pkgs):
    for pkg in pkgs:
        _ensure_pkg(pkg)
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon_config = _load_module(
    "neuralop.flood.neon_config", "neuralop/flood/neon_config.py", ("neuralop", "neuralop.flood")
)
cli = _load_module(
    "neuralop.flood.cli.train_neon_stage2",
    "neuralop/flood/cli/train_neon_stage2.py",
    ("neuralop", "neuralop.flood", "neuralop.flood.cli"),
)

NEONStage2Config = neon_config.NEONStage2Config
load_neon_config = neon_config.load_neon_config
resolve_training_plan = cli.resolve_training_plan
main = cli.main


# ---------------------------------------------------------------------------
# Config lead_time_dim knob
# ---------------------------------------------------------------------------


def test_config_lead_time_dim_defaults_off_and_validates():
    assert NEONStage2Config().lead_time_dim == 0
    NEONStage2Config(lead_time_dim=8).validate()
    with pytest.raises(neon_config.NEONConfigError, match="lead_time_dim"):
        NEONStage2Config(lead_time_dim=-1).validate()


def test_load_config_maps_lead_time_dim():
    cfg = load_neon_config({"neon": {"enabled": True, "lead_time_dim": 12}})
    assert cfg.lead_time_dim == 12


# ---------------------------------------------------------------------------
# Training plan resolution
# ---------------------------------------------------------------------------


def test_resolve_training_plan_surfaces_key_settings():
    cfg = NEONStage2Config(
        enabled=True,
        d_e=16,
        m_train=4,
        k_train=8,
        m_eval=16,
        k_eval=50,
        n_epochs=30,
        feature_source="decoder_pre_projection",
        dependency="za_dependent",
        objective="per_epistemic_fcrps",
        prior_scale="auto_0p10_base_rmse",
        alpha=None,
        lead_time_dim=6,
    )
    plan = resolve_training_plan(cfg)
    assert plan["feature_source"] == "decoder_pre_projection"
    assert plan["dependency"] == "za_dependent"
    assert plan["objective"] == "per_epistemic_fcrps"
    assert plan["m_train"] == 4 and plan["k_train"] == 8
    assert plan["m_eval"] == 16 and plan["k_eval"] == 50
    assert plan["n_epochs"] == 30
    assert plan["lead_time_dim"] == 6
    assert plan["prior_scale_mode"] == "auto"
    assert plan["prior_scale_fraction"] == pytest.approx(0.10)
    assert plan["loss_weights"] == {
        "rpf": pytest.approx(0.0),
        "smooth": pytest.approx(0.0),
        "time": pytest.approx(0.0),
        "pos": pytest.approx(0.0),
        "mag": pytest.approx(0.0),
    }
    assert plan["optimizer"]["learning_rate"] == pytest.approx(1e-4)


def test_resolve_training_plan_explicit_alpha_mode():
    cfg = NEONStage2Config(alpha=0.12, prior_scale="auto_0p10_base_rmse")
    plan = resolve_training_plan(cfg)
    assert plan["prior_scale_mode"] == "explicit"
    assert plan["alpha"] == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# CLI dry-run entrypoint
# ---------------------------------------------------------------------------


def test_main_dry_run_prints_plan_and_returns_zero(tmp_path, capsys):
    yaml = pytest.importorskip("yaml")
    cfg_path = tmp_path / "neon.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "neon": {
                    "enabled": True,
                    "stage1_checkpoint_dir": "/tmp/fgno",
                    "stage2_checkpoint_dir": str(tmp_path / "out"),
                    "d_e": 8,
                    "M_train": 2,
                    "K_train": 4,
                    "n_epochs": 5,
                }
            }
        )
    )
    rc = main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    printed = capsys.readouterr().out
    assert "decoder_pre_projection" in printed
    assert "n_epochs" in printed
    assert "5" in printed


def test_main_requires_config():
    with pytest.raises(SystemExit):
        main([])


def test_bundle_fallback_uses_process_local_manifest(tmp_path):
    import json
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    manifest = tmp_path / "coastal_fgn_bundle.json"
    manifest.write_text(
        json.dumps(
            {"dt_seconds": 1200, "geometry_path": "domain/geometry.npy"}
        )
    )
    temporary_paths = []
    lock = threading.Lock()

    def fake_load_model_bundle(path):
        candidate = Path(path)
        if candidate == manifest:
            raise ValueError("serving dt metadata drift")
        with lock:
            temporary_paths.append(candidate)
        time.sleep(0.01)
        return json.loads(candidate.read_text())

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: cli._load_bundle_with_dt_fallback(
                    manifest, fake_load_model_bundle
                ),
                range(8),
            )
        )

    assert all(result["dt_seconds"] == 900 for result in results)
    assert len(set(temporary_paths)) == 8
    assert all(path.parent == manifest.parent for path in temporary_paths)
    assert all(not path.exists() for path in temporary_paths)
    assert json.loads(manifest.read_text())["dt_seconds"] == 1200


def test_eval_cli_dry_run_prints_torch_free_plan(capsys):
    import json as _json

    eval_cli = _load_module(
        "neuralop.flood.cli.eval_neon_stage2",
        "neuralop/flood/cli/eval_neon_stage2.py",
        ("neuralop", "neuralop.flood", "neuralop.flood.cli"),
    )
    rc = eval_cli.main([
        "--config", "/tmp/flood.yaml",
        "--stage2-checkpoint", "/tmp/neon_stage2_best.pt",
        "--stage1-bundle", "/tmp/bundle.json",
        "--output-dir", "/tmp/out",
        "--allow-single-reference",
        "--dry-run",
    ])
    assert rc == 0
    plan = _json.loads(capsys.readouterr().out)
    assert plan["m_eval"] == 32
    assert plan["k_eval"] == 50
    assert plan["families"] == "val"
    assert plan["thresholds"] == [0.1, 0.3, 0.5]
    assert plan["cache_dir"] is None
    assert plan["k_chunk"] == 16
    assert plan["impact_members"] == 60
    assert plan["allow_single_reference"] is True


def test_eval_cli_rejects_single_reference_without_explicit_policy():
    eval_cli = _load_module(
        "neuralop.flood.cli.eval_neon_stage2_policy",
        "neuralop/flood/cli/eval_neon_stage2.py",
        ("neuralop", "neuralop.flood", "neuralop.flood.cli"),
    )

    class Family:
        reference = type("Reference", (), {"shape": (1, 2, 3, 1)})()

    with pytest.raises(ValueError, match="allow-single-reference"):
        eval_cli.validate_reference_member_policy(
            [Family()], allow_single_reference=False
        )
    policy = eval_cli.validate_reference_member_policy(
        [Family()], allow_single_reference=True
    )
    assert policy["single_reference_family_count"] == 1
    assert policy["single_reference_policy"] == (
        "forecast_crps_without_reference_self_distance"
    )
