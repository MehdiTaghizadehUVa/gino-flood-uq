from __future__ import annotations

import argparse
import csv
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from neuralop.flood.cli.coastal_fgn_gridsearch import (
    _load_and_apply_common_overrides,
    parse_training_log,
    precompute_normalizers,
)

DEFAULT_STAGES = ("stage_a", "stage_b", "stage_c")
TRIAL_STATUS_SUGGESTED = "suggested"
TRIAL_STATUS_COMPLETED = "completed"
TRIAL_STATUS_FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _format_scientific(value: float) -> str:
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _format_radius(value: float) -> str:
    return f"{float(value):.2f}"


def _short_param_tag(params: dict[str, Any]) -> str:
    return (
        f"lr{_format_scientific(float(params['learning_rate']))}_"
        f"wd{_format_scientific(float(params['weight_decay']))}_"
        f"r{_format_radius(float(params['gno_radius']))}_"
        f"h{int(params['fno_hidden_channels'])}_"
        f"z{int(params['fgn_noise_dim'])}"
    )


def _load_optuna():
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.trial import TrialState
        from optuna.storages import JournalFileStorage, JournalStorage
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Optuna is required for coastal_fgn_hpo. Install optuna in the controller environment."
        ) from exc
    return optuna, TPESampler, TrialState, JournalStorage, JournalFileStorage


def _study_root(path: Path) -> Path:
    return path.resolve()


def _study_metadata_path(study_root: Path) -> Path:
    return study_root / "study_metadata.json"


def _study_registry_path(study_root: Path) -> Path:
    return study_root / "trial_registry.json"


def _study_snapshot_path(study_root: Path) -> Path:
    return study_root / "study_spec.snapshot.yaml"


def _study_jobs_path(study_root: Path) -> Path:
    return study_root / "jobs.json"


def _study_ranking_root(study_root: Path) -> Path:
    return study_root / "ranking"


def _journal_path(study_root: Path) -> Path:
    return study_root / "optuna_journal.log"


def _stage_root(study_root: Path, stage: str) -> Path:
    return study_root / "stages" / stage


def _stage_manifest_path(study_root: Path, stage: str) -> Path:
    return _stage_root(study_root, stage) / "manifest.json"


def _stage_trial_list_path(study_root: Path, stage: str) -> Path:
    return _stage_root(study_root, stage) / "trial_specs.txt"


def _shared_normalizer_root(study_root: Path) -> Path:
    return study_root / "shared_normalizers"


def _normalize_study_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "study" in payload:
        payload = payload["study"]
    if not isinstance(payload, dict):
        raise ValueError("Study spec must be a mapping or contain a top-level 'study' mapping.")
    return payload


def _load_spec(path: Path) -> dict[str, Any]:
    return _normalize_study_payload(_read_yaml(path))


def _load_snapshot(study_root: Path) -> dict[str, Any]:
    return _load_spec(_study_snapshot_path(study_root))


def _load_metadata(study_root: Path) -> dict[str, Any]:
    metadata = _read_json(_study_metadata_path(study_root), default=None)
    if metadata is None:
        raise FileNotFoundError(f"Study metadata not found under {study_root}")
    return metadata


def _load_registry(study_root: Path) -> dict[str, Any]:
    registry = _read_json(_study_registry_path(study_root), default=None)
    if registry is None:
        registry = {"trials": {}, "stage_order": list(DEFAULT_STAGES)}
    return registry


def _save_registry(study_root: Path, registry: dict[str, Any]) -> None:
    _write_json(_study_registry_path(study_root), registry)


def _stage_order(spec: dict[str, Any]) -> list[str]:
    stages = spec.get("stages", {})
    ordered = [stage for stage in DEFAULT_STAGES if stage in stages]
    if not ordered:
        raise ValueError("Study spec must define at least one stage.")
    return ordered


def _stage_cfg(spec: dict[str, Any], stage: str) -> dict[str, Any]:
    try:
        cfg = spec["stages"][stage]
    except KeyError as exc:
        raise ValueError(f"Unknown stage {stage!r}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"Stage {stage!r} config must be a mapping.")
    return cfg


def _get_study(storage_root: Path, spec: dict[str, Any], metadata: dict[str, Any]):
    optuna, TPESampler, _, JournalStorage, JournalFileStorage = _load_optuna()
    sampler_seed = int(spec.get("optuna", {}).get("seed", spec.get("fixed_training", {}).get("seed", 123)))
    sampler = TPESampler(seed=sampler_seed)
    storage = JournalStorage(JournalFileStorage(str(_journal_path(storage_root))))
    study = optuna.create_study(
        study_name=str(metadata["study_id"]),
        storage=storage,
        sampler=sampler,
        direction=str(spec.get("objective", {}).get("direction", "minimize")),
        load_if_exists=True,
    )
    return study


def _stage_trial_count(spec: dict[str, Any], stage: str) -> int:
    cfg = _stage_cfg(spec, stage)
    if "n_trials" in cfg:
        return int(cfg["n_trials"])
    if "promote_top_k" in cfg:
        return int(cfg["promote_top_k"])
    raise ValueError(f"Stage {stage!r} requires n_trials or promote_top_k.")


def _trial_dir(study_root: Path, stage: str, trial_number: int) -> Path:
    return _stage_root(study_root, stage) / "trials" / f"trial_{int(trial_number):04d}"


def _trial_spec_path(study_root: Path, stage: str, trial_number: int) -> Path:
    return _trial_dir(study_root, stage, trial_number) / "trial_spec.json"


def _trial_stage_record(registry: dict[str, Any], trial_number: int, stage: str) -> dict[str, Any]:
    key = str(int(trial_number))
    trial_entry = registry.setdefault("trials", {}).setdefault(key, {"stages": {}, "optuna_told": False})
    return trial_entry.setdefault("stages", {}).setdefault(stage, {})


def _sample_params(trial, search_space: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, cfg in search_space.items():
        kind = str(cfg["type"]).strip().lower()
        if kind == "log_float":
            params[name] = float(trial.suggest_float(name, float(cfg["low"]), float(cfg["high"]), log=True))
        elif kind == "float":
            params[name] = float(trial.suggest_float(name, float(cfg["low"]), float(cfg["high"])))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(cfg["choices"]))
        else:
            raise ValueError(f"Unsupported search-space type {kind!r} for {name}.")
    return params


def _build_trial_spec(
    *,
    study_root: Path,
    spec: dict[str, Any],
    metadata: dict[str, Any],
    stage: str,
    trial_number: int,
    params: dict[str, Any],
    promoted_from: str | None,
) -> dict[str, Any]:
    stage_cfg = _stage_cfg(spec, stage)
    fixed = spec.get("fixed_training", {})
    short_tag = _short_param_tag(params)
    run_tag = f"coastal_fgn_hpo_{metadata['study_id']}_{stage}_trial{int(trial_number):04d}_{short_tag}"
    trial_dir = _trial_dir(study_root, stage, trial_number)
    normalizer_root = _shared_normalizer_root(study_root)
    n_samples_max = stage_cfg.get("n_samples_max", None)
    force_load = bool(n_samples_max is not None)
    return {
        "study_id": metadata["study_id"],
        "stage": stage,
        "promoted_from": promoted_from,
        "trial_number": int(trial_number),
        "optuna_trial_number": int(trial_number),
        "created_at": _utc_now(),
        "run_tag": run_tag,
        "short_param_tag": short_tag,
        "base_config_path": str(Path(spec["base_config"]).resolve()),
        "data_root": str(spec["data_root"]),
        "clean_boundary_root": str(spec["clean_boundary_root"]),
        "config_path": str((trial_dir / "config.yaml").resolve()),
        "checkpoint_dir": str((trial_dir / "checkpoint").resolve()),
        "summary_path": str((trial_dir / "run_summary.json").resolve()),
        "log_path": str((trial_dir / "checkpoint" / "training.log").resolve()),
        "normalizer_root": str(normalizer_root.resolve()),
        "normalizer_path": str((normalizer_root / "normalizers_depth_only.pt").resolve()),
        "hyperparameters": {
            "learning_rate": float(params["learning_rate"]),
            "weight_decay": float(params["weight_decay"]),
            "gno_radius": float(params["gno_radius"]),
            "fno_hidden_channels": int(params["fno_hidden_channels"]),
            "fgn_noise_dim": int(params["fgn_noise_dim"]),
        },
        "budget": {
            "batch_size": int(fixed.get("batch_size", 16)),
            "n_epochs": int(stage_cfg["n_epochs"]),
            "n_samples_max": None if n_samples_max is None else int(n_samples_max),
            "force_load_normalizers": force_load,
            "amp_autocast": bool(fixed.get("amp_autocast", False)),
            "deterministic": bool(fixed.get("deterministic", False)),
            "seed": int(fixed.get("seed", 123)),
        },
        "wandb": {
            "log": bool(spec.get("wandb", {}).get("log", True)),
            "group": f"coastal_fgn_hpo_{metadata['study_id']}",
            "name": f"{stage}_trial{int(trial_number):04d}_{short_tag}",
            "sweep": False,
            "project": spec.get("wandb", {}).get("project", None),
            "entity": spec.get("wandb", {}).get("entity", None),
            "tags": [metadata["study_id"], stage, f"trial:{int(trial_number)}"],
        },
    }


def _ensure_trial_spec_written(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, payload)


def _rank_stage_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[Any, ...]:
        best_crps = item.get("best_test_crps")
        best_l2 = item.get("best_epoch_test_l2")
        return (
            best_crps is None,
            float("inf") if best_crps is None else float(best_crps),
            best_l2 is None,
            float("inf") if best_l2 is None else float(best_l2),
            int(item["trial_number"]),
        )

    return sorted(records, key=key)


def _slurm_walltime(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        total_seconds = int(value)
        days, rem = divmod(total_seconds, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        if days > 0:
            return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    raise TypeError(f"Unsupported walltime value: {value!r}")


def _manifest_payload(*, study_root: Path, spec: dict[str, Any], metadata: dict[str, Any], stage: str, trial_specs: list[Path], promoted_from: str | None = None) -> dict[str, Any]:
    stage_cfg = _stage_cfg(spec, stage)
    return {
        "study_id": metadata["study_id"],
        "stage": stage,
        "promoted_from": promoted_from,
        "created_at": _utc_now(),
        "stage_root": str(_stage_root(study_root, stage)),
        "trial_spec_list_path": str(_stage_trial_list_path(study_root, stage)),
        "trial_specs": [str(path.resolve()) for path in trial_specs],
        "trial_count": len(trial_specs),
        "budget": {
            "n_epochs": int(stage_cfg["n_epochs"]),
            "n_samples_max": stage_cfg.get("n_samples_max", None),
        },
        "resources": {
            "concurrency": int(stage_cfg["concurrency"]),
            "walltime": _slurm_walltime(stage_cfg["walltime"]),
            "partition": str(stage_cfg["partition"]),
            "account": str(stage_cfg.get("account", "uqgroup")),
            "cpus": int(stage_cfg.get("cpus", 8)),
            "mem": str(stage_cfg.get("mem", "128G")),
            "gres": str(stage_cfg.get("gres", "gpu:1")),
        },
    }


def _write_manifest_and_list(study_root: Path, stage: str, manifest: dict[str, Any]) -> None:
    _write_json(_stage_manifest_path(study_root, stage), manifest)
    list_path = _stage_trial_list_path(study_root, stage)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as handle:
        for item in manifest["trial_specs"]:
            handle.write(f"{item}\n")


def _load_manifest(study_root: Path, stage: str) -> dict[str, Any]:
    manifest = _read_json(_stage_manifest_path(study_root, stage), default=None)
    if manifest is None:
        raise FileNotFoundError(f"Stage manifest missing for {stage!r} under {study_root}")
    return manifest


def _describe_study_shell(study_root: Path, spec: dict[str, Any], metadata: dict[str, Any], stage: str | None = None) -> str:
    payload = {
        "HPO_STUDY_ROOT": str(study_root),
        "HPO_STUDY_ID": metadata["study_id"],
        "HPO_BASE_CONFIG": str(Path(spec["base_config"]).resolve()),
        "HPO_DATA_ROOT": str(spec["data_root"]),
        "HPO_CLEAN_BOUNDARY_ROOT": str(spec["clean_boundary_root"]),
        "HPO_SHARED_NORMALIZER_ROOT": str(_shared_normalizer_root(study_root)),
    }
    precompute = spec.get("precompute", {})
    payload.update(
        {
            "HPO_PRECOMP_PARTITION": str(precompute.get("partition", "gpu-a100-80")),
            "HPO_PRECOMP_ACCOUNT": str(precompute.get("account", "uqgroup")),
            "HPO_PRECOMP_CPUS": int(precompute.get("cpus", 8)),
            "HPO_PRECOMP_MEM": str(precompute.get("mem", "128G")),
            "HPO_PRECOMP_GRES": str(precompute.get("gres", "gpu:1")),
            "HPO_PRECOMP_WALLTIME": _slurm_walltime(precompute.get("walltime", "02:00:00")),
        }
    )
    if stage is not None:
        manifest = _load_manifest(study_root, stage)
        resources = manifest["resources"]
        payload.update(
            {
                "HPO_STAGE": stage,
                "HPO_STAGE_TRIAL_COUNT": int(manifest["trial_count"]),
                "HPO_STAGE_CONCURRENCY": int(resources["concurrency"]),
                "HPO_STAGE_WALLTIME": str(resources["walltime"]),
                "HPO_STAGE_PARTITION": str(resources["partition"]),
                "HPO_STAGE_ACCOUNT": str(resources["account"]),
                "HPO_STAGE_CPUS": int(resources["cpus"]),
                "HPO_STAGE_MEM": str(resources["mem"]),
                "HPO_STAGE_GRES": str(resources["gres"]),
                "HPO_STAGE_TRIAL_LIST": str(_stage_trial_list_path(study_root, stage)),
            }
        )
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in payload.items())


def _describe_trial_shell(trial_spec: dict[str, Any]) -> str:
    payload = {
        "HPO_TRIAL_SPEC_PATH": str(Path(trial_spec["summary_path"]).parent / "trial_spec.json"),
        "HPO_TRIAL_NUMBER": int(trial_spec["trial_number"]),
        "HPO_STAGE": str(trial_spec["stage"]),
        "HPO_RUN_TAG": str(trial_spec["run_tag"]),
        "HPO_CONFIG_PATH": str(trial_spec["config_path"]),
        "HPO_CHECKPOINT_DIR": str(trial_spec["checkpoint_dir"]),
        "HPO_SUMMARY_PATH": str(trial_spec["summary_path"]),
        "HPO_LOG_PATH": str(trial_spec["log_path"]),
        "HPO_NORMALIZER_PATH": str(trial_spec["normalizer_path"]),
        "HPO_WANDB_GROUP": str(trial_spec["wandb"]["group"]),
        "HPO_WANDB_NAME": str(trial_spec["wandb"]["name"]),
    }
    return "\n".join(f"export {key}={shlex.quote(str(value))}" for key, value in payload.items())


def init_study(*, study_spec_path: Path, study_root: Path, overwrite: bool = False) -> dict[str, Any]:
    study_root = _study_root(study_root)
    metadata_path = _study_metadata_path(study_root)
    if metadata_path.exists() and not overwrite:
        metadata = _load_metadata(study_root)
        return {"status": "exists", "study_root": str(study_root), "study_id": metadata["study_id"]}

    spec = _load_spec(study_spec_path)
    study_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = _study_snapshot_path(study_root)
    _write_yaml(snapshot_path, {"study": spec})
    study_id = str(spec.get("study_id") or spec.get("name") or study_root.name)
    metadata = {
        "study_id": study_id,
        "study_root": str(study_root),
        "created_at": _utc_now(),
        "source_spec_path": str(study_spec_path.resolve()),
        "stage_order": _stage_order(spec),
        "objective_metric": str(spec.get("objective", {}).get("metric", "best_test_crps")),
        "objective_direction": str(spec.get("objective", {}).get("direction", "minimize")),
    }
    _write_json(metadata_path, metadata)
    _save_registry(study_root, {"trials": {}, "stage_order": metadata["stage_order"]})
    _write_json(_study_jobs_path(study_root), {"created_at": _utc_now(), "jobs": []})
    _shared_normalizer_root(study_root).mkdir(parents=True, exist_ok=True)
    _study_ranking_root(study_root).mkdir(parents=True, exist_ok=True)
    _get_study(study_root, spec, metadata)
    return {
        "status": "initialized",
        "study_root": str(study_root),
        "study_id": study_id,
        "snapshot_path": str(snapshot_path),
        "journal_path": str(_journal_path(study_root)),
    }


def suggest_trials(*, study_root: Path, stage: str, overwrite: bool = False) -> dict[str, Any]:
    study_root = _study_root(study_root)
    spec = _load_snapshot(study_root)
    metadata = _load_metadata(study_root)
    manifest_path = _stage_manifest_path(study_root, stage)
    if manifest_path.exists() and not overwrite:
        manifest = _load_manifest(study_root, stage)
        return {"status": "exists", "manifest_path": str(manifest_path), "trial_count": manifest["trial_count"]}

    if stage != _stage_order(spec)[0]:
        raise ValueError("suggest-trial only samples the initial search stage; later stages must use promote-stage.")

    registry = _load_registry(study_root)
    study = _get_study(study_root, spec, metadata)
    trial_specs: list[Path] = []
    search_space = spec["search_space"]
    trial_count = _stage_trial_count(spec, stage)
    for _ in range(trial_count):
        trial = study.ask()
        params = _sample_params(trial, search_space)
        payload = _build_trial_spec(
            study_root=study_root,
            spec=spec,
            metadata=metadata,
            stage=stage,
            trial_number=int(trial.number),
            params=params,
            promoted_from=None,
        )
        spec_path = _trial_spec_path(study_root, stage, int(trial.number))
        _ensure_trial_spec_written(spec_path, payload)
        stage_record = _trial_stage_record(registry, int(trial.number), stage)
        trial_entry = registry["trials"][str(int(trial.number))]
        trial_entry.update(
            {
                "trial_number": int(trial.number),
                "optuna_trial_number": int(trial.number),
                "hyperparameters": payload["hyperparameters"],
                "created_at": payload["created_at"],
            }
        )
        stage_record.update(
            {
                "status": TRIAL_STATUS_SUGGESTED,
                "spec_path": str(spec_path),
                "summary_path": payload["summary_path"],
                "checkpoint_dir": payload["checkpoint_dir"],
                "config_path": payload["config_path"],
                "run_tag": payload["run_tag"],
            }
        )
        trial_specs.append(spec_path)
        trial.set_user_attr("study_id", metadata["study_id"])
        trial.set_user_attr("initial_stage", stage)
        trial.set_user_attr("trial_spec_path", str(spec_path))
    _save_registry(study_root, registry)
    manifest = _manifest_payload(
        study_root=study_root,
        spec=spec,
        metadata=metadata,
        stage=stage,
        trial_specs=trial_specs,
        promoted_from=None,
    )
    _write_manifest_and_list(study_root, stage, manifest)
    return {
        "status": "created",
        "manifest_path": str(manifest_path),
        "trial_spec_list_path": str(_stage_trial_list_path(study_root, stage)),
        "trial_count": len(trial_specs),
    }


def promote_stage(*, study_root: Path, from_stage: str, to_stage: str, overwrite: bool = False) -> dict[str, Any]:
    study_root = _study_root(study_root)
    spec = _load_snapshot(study_root)
    metadata = _load_metadata(study_root)
    to_manifest_path = _stage_manifest_path(study_root, to_stage)
    if to_manifest_path.exists() and not overwrite:
        manifest = _load_manifest(study_root, to_stage)
        return {"status": "exists", "manifest_path": str(to_manifest_path), "trial_count": manifest["trial_count"]}

    registry = _load_registry(study_root)
    ranked_records: list[dict[str, Any]] = []
    for trial_key, entry in registry.get("trials", {}).items():
        stage_entry = entry.get("stages", {}).get(from_stage)
        if not stage_entry:
            continue
        if stage_entry.get("status") != TRIAL_STATUS_COMPLETED:
            continue
        if stage_entry.get("best_test_crps") is None:
            continue
        ranked_records.append(
            {
                "trial_number": int(trial_key),
                "best_test_crps": stage_entry.get("best_test_crps"),
                "best_epoch_test_l2": stage_entry.get("best_epoch_test_l2"),
            }
        )
    ranked_records = _rank_stage_records(ranked_records)
    promote_k = _stage_trial_count(spec, to_stage)
    selected = ranked_records[:promote_k]
    trial_specs: list[Path] = []
    for item in selected:
        trial_number = int(item["trial_number"])
        prior = registry["trials"][str(trial_number)]
        payload = _build_trial_spec(
            study_root=study_root,
            spec=spec,
            metadata=metadata,
            stage=to_stage,
            trial_number=trial_number,
            params=prior["hyperparameters"],
            promoted_from=from_stage,
        )
        spec_path = _trial_spec_path(study_root, to_stage, trial_number)
        _ensure_trial_spec_written(spec_path, payload)
        stage_record = _trial_stage_record(registry, trial_number, to_stage)
        stage_record.update(
            {
                "status": TRIAL_STATUS_SUGGESTED,
                "spec_path": str(spec_path),
                "summary_path": payload["summary_path"],
                "checkpoint_dir": payload["checkpoint_dir"],
                "config_path": payload["config_path"],
                "run_tag": payload["run_tag"],
                "promoted_from": from_stage,
            }
        )
        trial_specs.append(spec_path)
    _save_registry(study_root, registry)
    manifest = _manifest_payload(
        study_root=study_root,
        spec=spec,
        metadata=metadata,
        stage=to_stage,
        trial_specs=trial_specs,
        promoted_from=from_stage,
    )
    _write_manifest_and_list(study_root, to_stage, manifest)
    return {
        "status": "created",
        "manifest_path": str(to_manifest_path),
        "trial_spec_list_path": str(_stage_trial_list_path(study_root, to_stage)),
        "trial_count": len(trial_specs),
    }


def render_trial_config(*, trial_spec_path: Path) -> dict[str, Any]:
    trial_spec = _read_json(trial_spec_path)
    payload, config = _load_and_apply_common_overrides(
        base_config_path=Path(trial_spec["base_config_path"]),
        data_root=str(trial_spec["data_root"]),
        clean_boundary_root=str(trial_spec["clean_boundary_root"]),
        seed=int(trial_spec["budget"]["seed"]),
        deterministic=bool(trial_spec["budget"]["deterministic"]),
    )
    config.setdefault("data", {})["normalizer_root"] = str(Path(trial_spec["normalizer_root"]))
    config["data"]["normalizer_path"] = Path(trial_spec["normalizer_path"]).name
    config["data"]["batch_size"] = int(trial_spec["budget"]["batch_size"])
    config["data"]["n_samples_max"] = trial_spec["budget"]["n_samples_max"]
    config["data"]["force_load_normalizers"] = bool(trial_spec["budget"].get("force_load_normalizers", False))

    config.setdefault("checkpoint", {})["save_dir"] = str(Path(trial_spec["checkpoint_dir"]))
    config["checkpoint"]["resume_from_dir"] = None
    config["checkpoint"]["save_every"] = int(trial_spec["budget"]["n_epochs"])
    config["checkpoint"]["save_best_metric"] = "test_crps"

    config.setdefault("opt", {})["n_epochs"] = int(trial_spec["budget"]["n_epochs"])
    hp = trial_spec["hyperparameters"]
    config["opt"]["learning_rate"] = float(hp["learning_rate"])
    config["opt"]["weight_decay"] = float(hp["weight_decay"])
    config["opt"]["amp_autocast"] = bool(trial_spec["budget"].get("amp_autocast", False))

    config.setdefault("gino", {})["gno_radius"] = float(hp["gno_radius"])
    config["gino"]["fno_hidden_channels"] = int(hp["fno_hidden_channels"])
    config["gino"]["fgn_noise_dim"] = int(hp["fgn_noise_dim"])

    config.setdefault("wandb", {})["log"] = bool(trial_spec["wandb"]["log"])
    config["wandb"]["group"] = str(trial_spec["wandb"]["group"])
    config["wandb"]["name"] = str(trial_spec["wandb"]["name"])
    config["wandb"]["tags"] = list(trial_spec["wandb"].get("tags", []))
    config["wandb"]["trial_metadata"] = {
        "study_id": trial_spec["study_id"],
        "stage": trial_spec["stage"],
        "trial_number": int(trial_spec["trial_number"]),
        "promoted_from": trial_spec.get("promoted_from"),
    }
    config["wandb"]["sweep"] = False
    if trial_spec["wandb"].get("project"):
        config["wandb"]["project"] = str(trial_spec["wandb"]["project"])
    if trial_spec["wandb"].get("entity"):
        config["wandb"]["entity"] = str(trial_spec["wandb"]["entity"])

    config["verify_training"] = False
    config.setdefault("rollout", {})["run_after_training"] = False

    config_path = Path(trial_spec["config_path"])
    _write_yaml(config_path, payload)
    rendered = dict(trial_spec)
    rendered["config_path"] = str(config_path)
    return rendered


def summarize_trial(
    *,
    trial_spec_path: Path,
    status: str,
    job_id: str | None,
    array_task_id: str | None,
    git_sha: str | None,
) -> dict[str, Any]:
    trial_spec = _read_json(trial_spec_path)
    parsed = parse_training_log(Path(trial_spec["log_path"]))
    summary = {
        "status": status,
        "job_id": job_id,
        "array_task_id": array_task_id,
        "git_sha": git_sha,
        "study_id": trial_spec["study_id"],
        "stage": trial_spec["stage"],
        "promoted_from": trial_spec.get("promoted_from"),
        "trial_number": int(trial_spec["trial_number"]),
        "optuna_trial_number": int(trial_spec["optuna_trial_number"]),
        "run_tag": trial_spec["run_tag"],
        "config_path": trial_spec["config_path"],
        "checkpoint_dir": trial_spec["checkpoint_dir"],
        "log_path": trial_spec["log_path"],
        "hyperparameters": trial_spec["hyperparameters"],
        "best_epoch": parsed["best_epoch"],
        "best_test_crps": parsed["best_test_crps"],
        "best_epoch_test_l2": parsed["best_epoch_test_l2"],
        "final_epoch": parsed["final_epoch"],
        "final_metrics": parsed["final_metrics"],
        "epochs_seen": len(parsed["epochs"]),
    }
    _write_json(Path(trial_spec["summary_path"]), summary)
    return summary


def tell_result(*, study_root: Path, stage: str, mark_missing_failed: bool = False) -> dict[str, Any]:
    study_root = _study_root(study_root)
    spec = _load_snapshot(study_root)
    metadata = _load_metadata(study_root)
    registry = _load_registry(study_root)
    manifest = _load_manifest(study_root, stage)
    study = _get_study(study_root, spec, metadata)
    _, _, TrialState, _, _ = _load_optuna()

    completed = 0
    failed = 0
    told = 0
    for spec_path_str in manifest["trial_specs"]:
        trial_spec = _read_json(Path(spec_path_str))
        trial_number = int(trial_spec["trial_number"])
        entry = registry["trials"][str(trial_number)]
        stage_record = _trial_stage_record(registry, trial_number, stage)
        summary_path = Path(trial_spec["summary_path"])
        summary = _read_json(summary_path, default=None)
        if summary is None:
            if not mark_missing_failed:
                continue
            summary = {
                "status": TRIAL_STATUS_FAILED,
                "best_test_crps": None,
                "best_epoch_test_l2": None,
                "epochs_seen": 0,
            }
            stage_record["failure_reason"] = "missing_summary"

        stage_record.update(
            {
                "status": str(summary.get("status", TRIAL_STATUS_FAILED)),
                "summary_path": str(summary_path),
                "best_test_crps": summary.get("best_test_crps"),
                "best_epoch_test_l2": summary.get("best_epoch_test_l2"),
                "best_epoch": summary.get("best_epoch"),
                "final_epoch": summary.get("final_epoch"),
                "final_metrics": summary.get("final_metrics", {}),
                "epochs_seen": summary.get("epochs_seen", 0),
                "job_id": summary.get("job_id"),
                "array_task_id": summary.get("array_task_id"),
            }
        )
        if stage_record["status"] == TRIAL_STATUS_COMPLETED and stage_record.get("best_test_crps") is not None:
            completed += 1
            entry["latest_completed_stage"] = stage
            entry["latest_completed_metric"] = stage_record.get("best_test_crps")
        else:
            failed += 1

        if not bool(entry.get("optuna_told", False)):
            try:
                if stage_record["status"] == TRIAL_STATUS_COMPLETED and stage_record.get("best_test_crps") is not None:
                    study.tell(trial_number, float(stage_record["best_test_crps"]))
                else:
                    study.tell(trial_number, state=TrialState.FAIL)
                entry["optuna_told"] = True
                entry["optuna_told_stage"] = stage
                told += 1
            except Exception as exc:
                entry["optuna_tell_error"] = str(exc)
    _save_registry(study_root, registry)
    return {
        "study_root": str(study_root),
        "stage": stage,
        "completed": completed,
        "failed": failed,
        "told": told,
    }


def export_ranking(*, study_root: Path, output_json: Path, output_csv: Path | None = None) -> dict[str, Any]:
    study_root = _study_root(study_root)
    spec = _load_snapshot(study_root)
    registry = _load_registry(study_root)
    ordered_stages = _stage_order(spec)
    rows: list[dict[str, Any]] = []
    for trial_key, entry in registry.get("trials", {}).items():
        chosen_stage = None
        chosen = None
        for stage in reversed(ordered_stages):
            stage_entry = entry.get("stages", {}).get(stage)
            if not stage_entry:
                continue
            if stage_entry.get("status") == TRIAL_STATUS_COMPLETED and stage_entry.get("best_test_crps") is not None:
                chosen_stage = stage
                chosen = stage_entry
                break
        if chosen is None:
            continue
        rows.append(
            {
                "trial_number": int(trial_key),
                "stage": chosen_stage,
                "best_test_crps": chosen.get("best_test_crps"),
                "best_epoch_test_l2": chosen.get("best_epoch_test_l2"),
                "summary_path": chosen.get("summary_path"),
                "run_tag": chosen.get("run_tag"),
                "hyperparameters": entry.get("hyperparameters", {}),
                "objective_note": "best_test_crps from the trainer's current 90/10 one-step holdout split",
            }
        )
    rows = _rank_stage_records(rows)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_json, rows)
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "trial_number",
                    "stage",
                    "best_test_crps",
                    "best_epoch_test_l2",
                    "summary_path",
                    "run_tag",
                    "learning_rate",
                    "weight_decay",
                    "gno_radius",
                    "fno_hidden_channels",
                    "fgn_noise_dim",
                ],
            )
            writer.writeheader()
            for row in rows:
                hp = row["hyperparameters"]
                writer.writerow(
                    {
                        "trial_number": row["trial_number"],
                        "stage": row["stage"],
                        "best_test_crps": row["best_test_crps"],
                        "best_epoch_test_l2": row["best_epoch_test_l2"],
                        "summary_path": row["summary_path"],
                        "run_tag": row["run_tag"],
                        "learning_rate": hp.get("learning_rate"),
                        "weight_decay": hp.get("weight_decay"),
                        "gno_radius": hp.get("gno_radius"),
                        "fno_hidden_channels": hp.get("fno_hidden_channels"),
                        "fgn_noise_dim": hp.get("fgn_noise_dim"),
                    }
                )
    return {
        "n_ranked": len(rows),
        "output_json": str(output_json),
        "output_csv": str(output_csv) if output_csv else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coastal FGN HPO helpers")
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser("init-study")
    init_p.add_argument("--study-spec", type=Path, required=True)
    init_p.add_argument("--study-root", type=Path, required=True)
    init_p.add_argument("--overwrite", action="store_true")

    describe_study = sub.add_parser("describe-study")
    describe_study.add_argument("--study-root", type=Path, required=True)
    describe_study.add_argument("--stage", type=str, default=None)
    describe_study.add_argument("--format", choices=("json", "shell"), default="json")

    suggest_p = sub.add_parser("suggest-trial")
    suggest_p.add_argument("--study-root", type=Path, required=True)
    suggest_p.add_argument("--stage", type=str, required=True)
    suggest_p.add_argument("--overwrite", action="store_true")

    promote_p = sub.add_parser("promote-stage")
    promote_p.add_argument("--study-root", type=Path, required=True)
    promote_p.add_argument("--from-stage", type=str, required=True)
    promote_p.add_argument("--to-stage", type=str, required=True)
    promote_p.add_argument("--overwrite", action="store_true")

    describe_trial = sub.add_parser("describe-trial")
    describe_trial.add_argument("--trial-spec", type=Path, required=True)
    describe_trial.add_argument("--format", choices=("json", "shell"), default="json")

    render_p = sub.add_parser("render-trial-config")
    render_p.add_argument("--trial-spec", type=Path, required=True)

    summarize_p = sub.add_parser("summarize-trial")
    summarize_p.add_argument("--trial-spec", type=Path, required=True)
    summarize_p.add_argument("--status", type=str, default=TRIAL_STATUS_COMPLETED)
    summarize_p.add_argument("--job-id", type=str, default=None)
    summarize_p.add_argument("--array-task-id", type=str, default=None)
    summarize_p.add_argument("--git-sha", type=str, default=None)

    tell_p = sub.add_parser("tell-result")
    tell_p.add_argument("--study-root", type=Path, required=True)
    tell_p.add_argument("--stage", type=str, required=True)
    tell_p.add_argument("--mark-missing-failed", action="store_true")

    export_p = sub.add_parser("export-ranking")
    export_p.add_argument("--study-root", type=Path, required=True)
    export_p.add_argument("--output-json", type=Path, required=True)
    export_p.add_argument("--output-csv", type=Path, default=None)

    precompute_p = sub.add_parser("precompute-study-normalizers")
    precompute_p.add_argument("--study-root", type=Path, required=True)
    precompute_p.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.command == "init-study":
        result = init_study(study_spec_path=args.study_spec, study_root=args.study_root, overwrite=bool(args.overwrite))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "describe-study":
        study_root = _study_root(args.study_root)
        spec = _load_snapshot(study_root)
        metadata = _load_metadata(study_root)
        payload = {
            "study_root": str(study_root),
            "study_id": metadata["study_id"],
            "base_config": str(Path(spec["base_config"]).resolve()),
            "data_root": str(spec["data_root"]),
            "clean_boundary_root": str(spec["clean_boundary_root"]),
            "shared_normalizer_root": str(_shared_normalizer_root(study_root)),
        }
        if args.stage is not None:
            payload["stage"] = args.stage
            payload["manifest"] = _load_manifest(study_root, args.stage)
        if args.format == "json":
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(_describe_study_shell(study_root, spec, metadata, stage=args.stage))
        return 0

    if args.command == "suggest-trial":
        result = suggest_trials(study_root=args.study_root, stage=args.stage, overwrite=bool(args.overwrite))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "promote-stage":
        result = promote_stage(study_root=args.study_root, from_stage=args.from_stage, to_stage=args.to_stage, overwrite=bool(args.overwrite))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "describe-trial":
        trial_spec = _read_json(args.trial_spec)
        if args.format == "json":
            print(json.dumps(trial_spec, indent=2, sort_keys=True))
        else:
            print(_describe_trial_shell(trial_spec))
        return 0

    if args.command == "render-trial-config":
        result = render_trial_config(trial_spec_path=args.trial_spec)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "summarize-trial":
        result = summarize_trial(
            trial_spec_path=args.trial_spec,
            status=args.status,
            job_id=args.job_id,
            array_task_id=args.array_task_id,
            git_sha=args.git_sha,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "tell-result":
        result = tell_result(
            study_root=args.study_root,
            stage=args.stage,
            mark_missing_failed=bool(args.mark_missing_failed),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "export-ranking":
        result = export_ranking(study_root=args.study_root, output_json=args.output_json, output_csv=args.output_csv)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "precompute-study-normalizers":
        study_root = _study_root(args.study_root)
        spec = _load_snapshot(study_root)
        result = precompute_normalizers(
            base_config_path=Path(spec["base_config"]),
            normalizer_root=_shared_normalizer_root(study_root),
            data_root=spec.get("data_root"),
            clean_boundary_root=spec.get("clean_boundary_root"),
            n_samples_max=None,
            seed=int(spec.get("fixed_training", {}).get("seed", 123)),
            chunk_size=spec.get("precompute", {}).get("chunk_size", None),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    raise SystemExit(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
