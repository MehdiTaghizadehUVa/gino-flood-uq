from __future__ import annotations

import json
from pathlib import Path

import yaml

from neuralop.flood.cli.coastal_fgn_hpo import (
    _load_manifest,
    _load_registry,
    export_ranking,
    init_study,
    promote_stage,
    render_trial_config,
    suggest_trials,
    tell_result,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_STUDY_SPEC = REPO_ROOT / "config/flood/coastal/coastal_fgn_hpo_study.yaml"


def _write_study_spec(tmp_path: Path, *, stage_a_trials: int = 4, stage_b_top_k: int = 2, stage_c_top_k: int = 1) -> Path:
    payload = yaml.safe_load(BASE_STUDY_SPEC.read_text())
    study = payload["study"]
    study["name"] = "coastal_hpo_test"
    study["stages"]["stage_a"]["n_trials"] = stage_a_trials
    study["stages"]["stage_b"]["promote_top_k"] = stage_b_top_k
    study["stages"]["stage_c"]["promote_top_k"] = stage_c_top_k
    spec_path = tmp_path / "study.yaml"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return spec_path


def _load_trial_specs(study_root: Path, stage: str) -> list[dict]:
    manifest = _load_manifest(study_root, stage)
    return [json.loads(Path(path).read_text()) for path in manifest["trial_specs"]]


def _write_summary(trial_spec: dict, *, status: str, best_test_crps: float | None, best_epoch_test_l2: float | None) -> None:
    summary = {
        "status": status,
        "study_id": trial_spec["study_id"],
        "stage": trial_spec["stage"],
        "trial_number": trial_spec["trial_number"],
        "run_tag": trial_spec["run_tag"],
        "config_path": trial_spec["config_path"],
        "checkpoint_dir": trial_spec["checkpoint_dir"],
        "log_path": trial_spec["log_path"],
        "hyperparameters": trial_spec["hyperparameters"],
        "best_epoch": 1 if best_test_crps is not None else None,
        "best_test_crps": best_test_crps,
        "best_epoch_test_l2": best_epoch_test_l2,
        "final_epoch": 1 if best_test_crps is not None else None,
        "final_metrics": {"test_crps": best_test_crps, "test_l2": best_epoch_test_l2},
        "epochs_seen": 1 if best_test_crps is not None else 0,
    }
    Path(trial_spec["summary_path"]).write_text(json.dumps(summary, indent=2, sort_keys=True))


def test_suggest_trials_is_deterministic(tmp_path: Path) -> None:
    spec_path = _write_study_spec(tmp_path, stage_a_trials=4)
    study_root_a = tmp_path / "study_a"
    study_root_b = tmp_path / "study_b"

    init_study(study_spec_path=spec_path, study_root=study_root_a)
    init_study(study_spec_path=spec_path, study_root=study_root_b)
    suggest_trials(study_root=study_root_a, stage="stage_a")
    suggest_trials(study_root=study_root_b, stage="stage_a")

    specs_a = _load_trial_specs(study_root_a, "stage_a")
    specs_b = _load_trial_specs(study_root_b, "stage_a")
    params_a = [spec["hyperparameters"] for spec in specs_a]
    params_b = [spec["hyperparameters"] for spec in specs_b]
    assert params_a == params_b


def test_render_trial_config_injects_budget_and_shared_normalizers(tmp_path: Path) -> None:
    spec_path = _write_study_spec(tmp_path, stage_a_trials=1)
    study_root = tmp_path / "study"
    init_study(study_spec_path=spec_path, study_root=study_root)
    suggest_trials(study_root=study_root, stage="stage_a")
    trial_spec = _load_trial_specs(study_root, "stage_a")[0]

    render_trial_config(trial_spec_path=Path(trial_spec["summary_path"]).parent / "trial_spec.json")
    payload = yaml.safe_load(Path(trial_spec["config_path"]).read_text())
    config = payload["flood"]

    assert config["data"]["batch_size"] == 16
    assert config["data"]["n_samples_max"] == 4096
    assert config["data"]["force_load_normalizers"] is True
    assert config["data"]["normalizer_root"] == str(study_root / "shared_normalizers")
    assert config["opt"]["n_epochs"] == 2
    assert config["checkpoint"]["save_best_metric"] == "test_crps"
    assert config["wandb"]["group"].startswith("coastal_fgn_hpo_")
    assert "stage_a" in config["wandb"]["tags"]


def test_promote_stage_uses_metric_then_tie_breaker_and_skips_failures(tmp_path: Path) -> None:
    spec_path = _write_study_spec(tmp_path, stage_a_trials=4, stage_b_top_k=2)
    study_root = tmp_path / "study"
    init_study(study_spec_path=spec_path, study_root=study_root)
    suggest_trials(study_root=study_root, stage="stage_a")
    trial_specs = _load_trial_specs(study_root, "stage_a")

    _write_summary(trial_specs[0], status="completed", best_test_crps=0.50, best_epoch_test_l2=0.20)
    _write_summary(trial_specs[1], status="completed", best_test_crps=0.50, best_epoch_test_l2=0.10)
    _write_summary(trial_specs[2], status="completed", best_test_crps=0.40, best_epoch_test_l2=0.90)
    _write_summary(trial_specs[3], status="failed", best_test_crps=None, best_epoch_test_l2=None)

    tell_result(study_root=study_root, stage="stage_a", mark_missing_failed=True)
    promote_stage(study_root=study_root, from_stage="stage_a", to_stage="stage_b")
    promoted_specs = _load_trial_specs(study_root, "stage_b")
    promoted_numbers = [spec["trial_number"] for spec in promoted_specs]

    assert promoted_numbers == [trial_specs[2]["trial_number"], trial_specs[1]["trial_number"]]


def test_suggest_resume_and_mark_missing_failed(tmp_path: Path) -> None:
    spec_path = _write_study_spec(tmp_path, stage_a_trials=3)
    study_root = tmp_path / "study"
    init_study(study_spec_path=spec_path, study_root=study_root)
    first = suggest_trials(study_root=study_root, stage="stage_a")
    second = suggest_trials(study_root=study_root, stage="stage_a")

    assert first["trial_count"] == 3
    assert second["status"] == "exists"

    tell_result(study_root=study_root, stage="stage_a", mark_missing_failed=True)
    registry = _load_registry(study_root)
    statuses = [entry["stages"]["stage_a"]["status"] for entry in registry["trials"].values()]
    assert statuses == ["failed", "failed", "failed"]


def test_export_ranking_prefers_latest_completed_stage(tmp_path: Path) -> None:
    spec_path = _write_study_spec(tmp_path, stage_a_trials=3, stage_b_top_k=2)
    study_root = tmp_path / "study"
    init_study(study_spec_path=spec_path, study_root=study_root)
    suggest_trials(study_root=study_root, stage="stage_a")
    stage_a_specs = _load_trial_specs(study_root, "stage_a")

    _write_summary(stage_a_specs[0], status="completed", best_test_crps=0.40, best_epoch_test_l2=0.10)
    _write_summary(stage_a_specs[1], status="completed", best_test_crps=0.50, best_epoch_test_l2=0.20)
    _write_summary(stage_a_specs[2], status="failed", best_test_crps=None, best_epoch_test_l2=None)
    tell_result(study_root=study_root, stage="stage_a", mark_missing_failed=True)

    promote_stage(study_root=study_root, from_stage="stage_a", to_stage="stage_b")
    stage_b_specs = _load_trial_specs(study_root, "stage_b")
    _write_summary(stage_b_specs[0], status="completed", best_test_crps=0.35, best_epoch_test_l2=0.08)
    _write_summary(stage_b_specs[1], status="completed", best_test_crps=0.55, best_epoch_test_l2=0.30)
    tell_result(study_root=study_root, stage="stage_b", mark_missing_failed=True)

    output_json = tmp_path / "ranking.json"
    output_csv = tmp_path / "ranking.csv"
    export_ranking(study_root=study_root, output_json=output_json, output_csv=output_csv)
    rows = json.loads(output_json.read_text())

    assert rows[0]["stage"] == "stage_b"
    assert rows[0]["best_test_crps"] == 0.35
    assert rows[0]["trial_number"] == stage_b_specs[0]["trial_number"]
    assert output_csv.exists()
