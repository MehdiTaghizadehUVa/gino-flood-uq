"""Shared immutable runtime for NEON Stage-2 Phase-5 diagnostics."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neuralop.flood.eval.neon_phase5 import (
    verify_checksummed_artifact,
    write_checksummed_artifact,
)
from neuralop.flood.neon import load_neon_stage2_checkpoint
from neuralop.flood.train.neon import FrozenFGNOFeatureBatch, NEONFamilySample


LOG = logging.getLogger("neon_phase5")


@dataclass
class Phase5Context:
    stage2: Any
    stage2_metadata: dict[str, Any]
    train_families: list[NEONFamilySample]
    val_families: list[NEONFamilySample]
    collector: Any
    normalizers: dict[str, Any]
    target_variables: list[str]
    physical_scale_m: float
    device: torch.device
    checkpoint_path: Path
    cache_dir: Path
    git_head: str
    run_metadata: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def require_clean_repository(*, expected_head: str | None = None) -> str:
    head = repository_head()
    if expected_head is not None and head != str(expected_head):
        raise RuntimeError(f"Phase-5 git HEAD mismatch: {head} != {expected_head}.")
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("Phase-5 jobs refuse to run from a dirty repository.")
    return head


def _resolved_existing(path: str | Path, *, kind: str) -> Path:
    value = Path(path)
    if kind == "file" and not value.is_file():
        raise FileNotFoundError(f"missing frozen Phase-5 file: {value}")
    if kind == "dir" and not value.is_dir():
        raise FileNotFoundError(f"missing frozen Phase-5 directory: {value}")
    return value.resolve()


def _verify_frozen_file_mapping(
    payload: dict[str, Any],
    *,
    supplied: dict[str, str | Path],
    label: str,
) -> None:
    paths = payload.get("input_paths")
    hashes = payload.get("input_sha256")
    if not isinstance(paths, dict) or not isinstance(hashes, dict):
        raise ValueError(f"{label} does not contain frozen input paths and SHA-256 values.")
    for key, current_path in supplied.items():
        if key not in paths or key not in hashes:
            raise ValueError(f"{label} is missing frozen input {key!r}.")
        current = _resolved_existing(current_path, kind="file")
        expected = _resolved_existing(paths[key], kind="file")
        if current != expected:
            raise ValueError(
                f"{label} {key} path mismatch: {current} != {expected}."
            )
        observed_sha = file_sha256(current)
        if observed_sha != str(hashes[key]):
            raise ValueError(
                f"{label} {key} SHA-256 mismatch: {observed_sha} != {hashes[key]}."
            )


def verify_phase5_frozen_inputs(
    *,
    phase5_preflight_path: str | Path,
    config_path: str | Path,
    bundle_path: str | Path,
    checkpoint_path: str | Path,
    history_path: str | Path,
    source_preflight_path: str | Path,
    cache_dir: str | Path,
    expected_head: str,
) -> dict[str, Any]:
    """Fail closed if any Phase-5 source changed after its immutable preflight.

    The selected legacy B3 source is pinned directly by the root Phase-5
    preflight. Later pilot checkpoints cannot be known at root-preflight time,
    so they must carry a checksummed ``TRAINING_COMPLETE.json`` beside the
    checkpoint. This preserves one verification interface without weakening
    provenance for derived rungs.
    """

    root_path = _resolved_existing(phase5_preflight_path, kind="file")
    verify_checksummed_artifact(root_path)
    root = json.loads(root_path.read_text(encoding="utf-8"))
    if root.get("schema_version") != "neon_phase5_preflight_v1":
        raise ValueError("unsupported Phase-5 root preflight schema.")
    if str(root.get("analysis_git_head")) != str(expected_head):
        raise ValueError(
            "Phase-5 root preflight Git HEAD does not match the requested analysis HEAD."
        )

    current_cache = _resolved_existing(cache_dir, kind="dir")
    frozen_cache = _resolved_existing(root.get("cache_dir", ""), kind="dir")
    if current_cache != frozen_cache:
        raise ValueError(
            f"Phase-5 cache path mismatch: {current_cache} != {frozen_cache}."
        )
    canonical = _resolved_existing(root.get("canonical_cache_dir", ""), kind="dir")
    if canonical.parent != frozen_cache:
        raise ValueError("canonical Phase-5 cache is not inside the frozen v3 cache.")

    _verify_frozen_file_mapping(
        root,
        supplied={"config": config_path, "bundle": bundle_path},
        label="Phase-5 root preflight",
    )
    source = {
        "checkpoint": checkpoint_path,
        "history": history_path,
        "preflight": source_preflight_path,
    }
    frozen_paths = dict(root.get("input_paths") or {})
    is_root_source = all(
        key in frozen_paths
        and _resolved_existing(source[key], kind="file")
        == _resolved_existing(frozen_paths[key], kind="file")
        for key in source
    )
    if is_root_source:
        _verify_frozen_file_mapping(
            root, supplied=source, label="Phase-5 root preflight"
        )
        source_kind = "legacy_b3_root_preflight"
    else:
        checkpoint = _resolved_existing(checkpoint_path, kind="file")
        completion_path = checkpoint.parent / "TRAINING_COMPLETE.json"
        if not completion_path.is_file():
            raise FileNotFoundError(
                f"derived Phase-5 source is missing {completion_path.name}: {completion_path}"
            )
        verify_checksummed_artifact(completion_path)
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("schema_version") != "neon_stage2_training_complete_v1":
            raise ValueError("unsupported NEON training-completion schema.")
        if str(completion.get("git_head")) != str(expected_head):
            raise ValueError("derived Phase-5 source was trained at a different Git HEAD.")
        _verify_frozen_file_mapping(
            completion, supplied=source, label="NEON training completion"
        )
        source_kind = "derived_training_completion"

    return {
        "phase5_preflight": str(root_path),
        "phase5_preflight_sha256": file_sha256(root_path),
        "source_kind": source_kind,
    }


def write_provenance(
    output_dir: str | Path,
    *,
    head: str,
    checkpoint: str | Path,
    cache_dir: str | Path,
    protocol_sha256: str,
    frozen_inputs: dict[str, Any] | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "neon_phase5_provenance_v1",
        "git_head": str(head),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "cache_dir": str(cache_dir),
        "protocol_sha256": str(protocol_sha256),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if frozen_inputs is not None:
        payload["frozen_inputs"] = dict(frozen_inputs)
    write_checksummed_artifact(destination / "PROVENANCE.json", payload)


def _normalizer_scale(normalizer: Any) -> float:
    scale = getattr(normalizer, "std", None)
    if scale is None:
        scale = getattr(normalizer, "scale", None)
    if scale is None:
        return 1.0
    tensor = torch.as_tensor(scale).detach().double().reshape(-1)
    if tensor.numel() != 1 or float(tensor[0]) <= 0.0:
        raise ValueError("Phase-5 depth evaluation requires one positive affine target scale.")
    return float(tensor[0])


def load_phase5_context(
    *,
    config_path: str | Path,
    bundle_path: str | Path,
    checkpoint_path: str | Path,
    history_path: str | Path,
    preflight_path: str | Path,
    phase5_preflight_path: str | Path,
    cache_dir: str | Path,
    device: str | torch.device,
    expected_head: str | None = None,
) -> Phase5Context:
    """Reconstruct the exact B3 split and immutable v3 feature-cache collector."""

    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.neon_config import NEONStage2Config
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import (
        _frozen_rollout_cache_key,
        make_cached_feature_collector,
        make_feature_collector_from_frozen_model,
    )
    from neuralop.flood.utils.runtime_core import load_config_and_setup, parse_target_variables

    head = require_clean_repository(expected_head=expected_head)
    frozen_inputs = verify_phase5_frozen_inputs(
        phase5_preflight_path=phase5_preflight_path,
        config_path=config_path,
        bundle_path=bundle_path,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        source_preflight_path=preflight_path,
        cache_dir=cache_dir,
        expected_head=head,
    )
    checkpoint_path = Path(checkpoint_path)
    cache_dir = Path(cache_dir)
    if not checkpoint_path.is_file() or not cache_dir.is_dir():
        raise FileNotFoundError("Phase-5 checkpoint or v3 cache is missing.")
    history = json.loads(Path(history_path).read_text(encoding="utf-8"))
    preflight = json.loads(Path(preflight_path).read_text(encoding="utf-8"))
    if str(preflight.get("cache_dir")) != str(cache_dir):
        raise ValueError("preflight cache directory does not match Phase-5 cache.")
    config = NEONStage2Config(**dict(preflight["config"])).validate()

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_phase5", "--config_path", str(config_path)]
        flood_config, _dev, _logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = list(
        parse_target_variables(getattr(flood_config.data, "target_variables", ["wd"]))
    )
    stage1 = _load_frozen_stage1(str(bundle_path))
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared  # type: ignore[attr-defined]
    normalizers = dict(prepared["normalizers"])
    dry_mask = prepared.get("structural_dry_mask")
    train_pool, val_families = build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        LOG,
        structural_dry_artifact=(None if dry_mask is None else {"dry_mask": dry_mask}),
        rollout_length=None,
        max_families=None,
        val_fraction=0.1,
    )
    by_id = {family.family_id: family for family in train_pool}
    ordered_ids = [str(value) for value in history["train_family_ids"]]
    missing = sorted(set(ordered_ids).difference(by_id))
    if missing:
        raise ValueError(f"B3 fit families are missing from rebuilt data: {missing[:5]}.")
    train_families = [by_id[family_id] for family_id in ordered_ids]
    if len(train_families) != 450 or len(val_families) != 50:
        raise ValueError("Phase-5 requires the exact 450-fit/50-validation split.")

    canonical_enabled = bool(config.deterministic_head) and str(
        config.deterministic_head_feature
    ).lower() in {"canonical_aleatory_mean", "fixed_zero_latent"}
    canonical_namespace = hashlib.sha256(
        "|".join(
            [
                "neon_canonical_feature_v1",
                str(bundle_path),
                str(config.feature_source),
                str(bundle.fgn_noise_dim),
                str(bundle.n_history),
                str(config.deterministic_head_canonical_k),
                str(config.deterministic_head_latent_seed),
                str(config.deterministic_head_feature == "fixed_zero_latent"),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    canonical_cache_dir = cache_dir / f"canonical_{canonical_namespace}"
    collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=config.feature_source,
        n_history=int(bundle.n_history),
        latent_dim=int(bundle.fgn_noise_dim),
        generator=torch.Generator().manual_seed(0),
        canonical_k=(int(config.deterministic_head_canonical_k) if canonical_enabled else 0),
        canonical_seed=int(config.deterministic_head_latent_seed),
        canonical_zero_latent=str(config.deterministic_head_feature)
        == "fixed_zero_latent",
        canonical_cache_dir=canonical_cache_dir,
        target_normalizer=normalizers.get("target"),
    )
    cache_key = _frozen_rollout_cache_key(
        stage1_checkpoint=str(bundle_path),
        config=config,
        latent_dim=int(bundle.fgn_noise_dim),
        n_history=int(bundle.n_history),
        structural_dry_enabled=dry_mask is not None,
    )
    collector = make_cached_feature_collector(
        collector,
        cache_device="cpu",
        cache_dir=cache_dir,
        cache_key=cache_key,
        prefetch_workers=int(config.feature_prefetch_workers),
        prefetch_depth=int(config.feature_prefetch_depth),
    )
    stage2, metadata = load_neon_stage2_checkpoint(checkpoint_path, map_location=device)
    stage2 = stage2.to(device).eval()
    return Phase5Context(
        stage2=stage2,
        stage2_metadata=metadata,
        train_families=train_families,
        val_families=val_families,
        collector=collector,
        normalizers=normalizers,
        target_variables=target_variables,
        physical_scale_m=_normalizer_scale(normalizers.get("target")),
        device=torch.device(device),
        checkpoint_path=checkpoint_path,
        cache_dir=cache_dir,
        git_head=head,
        run_metadata={
            "ladder_rung": str(history.get("ladder_rung", "")).upper(),
            "prior_seed": history.get("prior_seed"),
            "val_seed": history.get("val_seed"),
            "subset_replicate": history.get("subset_replicate"),
            "n_train": history.get("n_train"),
            "frozen_inputs": frozen_inputs,
        },
    )


def collect_cached_family(
    context: Phase5Context,
    family: NEONFamilySample,
    *,
    k: int = 8,
    bank: int = 0,
) -> FrozenFGNOFeatureBatch:
    return context.collector(
        family,
        num_aleatory=int(k),
        generator=torch.Generator().manual_seed(0),
        latent_bank_id=int(bank),
    )


def inverse_predictions(context: Phase5Context, prediction: torch.Tensor) -> torch.Tensor:
    normalizer = context.normalizers["target"]
    normalizer.to(prediction.device)
    return normalizer.inverse_transform(prediction)


def inverse_reference(context: Phase5Context, reference: torch.Tensor) -> torch.Tensor:
    normalizer = context.normalizers["dynamic"]
    normalizer.to(reference.device)
    return normalizer.inverse_transform(reference)
