"""Precompute NEON frozen-rollout banks and canonical feature sidecars.

This utility is intentionally training-free. It partitions the complete TR
family set into disjoint deterministic shards, computes canonical mean
features once per family, and fills only missing frozen Stage-1 latent banks.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Sequence


LOG = logging.getLogger("neon_cache_warm")

TR_CONFIG = "/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml"
BUNDLE = (
    "/scratch/jrj6wm/GINO_Model/model_bundles/"
    "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json"
)


def select_shard(
    family_ids: Sequence[str], *, shard_index: int, shard_count: int
) -> list[str]:
    """Return one stable, disjoint strided partition of sorted family IDs."""

    count = int(shard_count)
    index = int(shard_index)
    if count < 1:
        raise ValueError("shard_count must be positive")
    if index < 0 or index >= count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    return sorted(str(value) for value in family_ids)[index::count]


def frozen_bank_cache_path(
    cache_dir: Path,
    *,
    cache_key: str,
    family_id: str,
    latent_bank_id: int,
    num_aleatory: int,
) -> Path:
    """Construct the same disk-cache path used by the training collector."""

    safe_id = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in str(family_id)
    )
    return Path(cache_dir) / (
        f"{cache_key}_{safe_id}_bank{int(latent_bank_id)}_k{int(num_aleatory)}.pt"
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_checksum(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )


def _plan_payload() -> dict:
    return {
        "schema_version": "neon_cache_warm_plan_v1",
        "repository": os.environ.get("NEON_REPO", ""),
        "expected_head": os.environ.get("NEON_EXPECTED_HEAD", ""),
        "cache_dir": os.environ.get("NEON_CACHE_DIR", ""),
        "output_dir": os.environ.get("NEON_WARM_OUTPUT_DIR", ""),
        "ladder_rung": os.environ.get("NEON_LADDER_RUNG", "B2"),
        "shard_count": int(os.environ.get("NEON_WARM_SHARD_COUNT", "1")),
        "canonical_k": 32,
        "canonical_seed": 123,
        "latent_bank_count": 4,
        "k_train": 8,
        "training_enabled": False,
    }


def _load_families_and_stage1():
    import torch

    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.neon import freeze_stage1_model
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.utils.runtime_core import (
        load_config_and_setup,
        parse_target_variables,
    )

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_stage2_warm_cache", "--config_path", TR_CONFIG]
        flood_config, _device, _is_logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(
        getattr(flood_config.data, "target_variables", ["wd"])
    )
    stage1 = _load_frozen_stage1(BUNDLE)
    freeze_stage1_model(stage1)
    bundle = _load_frozen_stage1.last_bundle
    prepared = _load_frozen_stage1.last_prepared
    dry_mask = prepared.get("structural_dry_mask")
    train_families, validation_families = build_families_from_config(
        flood_config,
        prepared["normalizers"],
        target_variables,
        LOG,
        structural_dry_artifact=(
            None if dry_mask is None else {"dry_mask": dry_mask}
        ),
        rollout_length=None,
        max_families=None,
        val_fraction=0.1,
    )
    return (
        stage1,
        bundle,
        prepared,
        sorted(train_families + validation_families, key=lambda item: item.family_id),
        torch,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    output_dir = Path(
        os.environ.get("NEON_WARM_OUTPUT_DIR")
        or "/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/cache_warm"
    )
    if os.environ.get("NEON_WARM_PLAN_ONLY") == "1":
        plan_path = Path(
            os.environ.get("NEON_WARM_PREFLIGHT_PATH")
            or output_dir / "preflight.json"
        )
        payload = _plan_payload()
        _atomic_json(plan_path, payload)
        _write_checksum(plan_path)
        print(plan_path)
        return 0

    from neon_stage2_tr_train import _resolved_ladder_config
    from neuralop.flood.train.neon_runner import (
        _frozen_rollout_cache_key,
        make_cached_feature_collector,
        make_feature_collector_from_frozen_model,
    )

    rung = os.environ.get("NEON_LADDER_RUNG", "B2")
    config = _resolved_ladder_config(
        rung,
        prior_scale=os.environ.get("NEON_PRIOR_SCALE", "auto_0p10_base_rmse"),
        d_e=int(os.environ.get("NEON_D_E", "16")),
        n_epochs=1,
    )
    if not config.deterministic_head:
        raise ValueError("cache warm requires a rung with deterministic_head enabled")
    cache_dir = Path(os.environ["NEON_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_index = int(os.environ.get("NEON_WARM_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("NEON_WARM_SHARD_COUNT", "1"))

    started = time.time()
    stage1, bundle, prepared, families, torch = _load_families_and_stage1()
    selected_ids = set(
        select_shard(
            [family.family_id for family in families],
            shard_index=shard_index,
            shard_count=shard_count,
        )
    )
    selected = [family for family in families if family.family_id in selected_ids]
    latent_dim = int(bundle.fgn_noise_dim)
    n_history = int(bundle.n_history)
    canonical_zero_latent = (
        str(config.deterministic_head_feature).strip().lower()
        == "fixed_zero_latent"
    )
    canonical_namespace = hashlib.sha256(
        "|".join(
            [
                "neon_canonical_feature_v1",
                str(BUNDLE),
                str(config.feature_source),
                str(latent_dim),
                str(n_history),
                str(config.deterministic_head_canonical_k),
                str(config.deterministic_head_latent_seed),
                str(canonical_zero_latent),
            ]
        ).encode("utf-8")
    ).hexdigest()[:16]
    canonical_dir = cache_dir / f"canonical_{canonical_namespace}"
    base_collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=config.feature_source,
        n_history=n_history,
        latent_dim=latent_dim,
        generator=torch.Generator().manual_seed(0),
        canonical_k=int(config.deterministic_head_canonical_k),
        canonical_seed=int(config.deterministic_head_latent_seed),
        canonical_zero_latent=canonical_zero_latent,
        canonical_cache_dir=canonical_dir,
        target_normalizer=prepared["normalizers"].get("target"),
    )
    base_key = _frozen_rollout_cache_key(
        stage1_checkpoint=BUNDLE,
        config=config,
        latent_dim=latent_dim,
        n_history=n_history,
        structural_dry_enabled=any(
            family.structural_dry_mask is not None for family in families
        ),
    )
    collector = make_cached_feature_collector(
        base_collector,
        cache_device="cpu",
        cache_dir=cache_dir,
        cache_key=base_key,
    )
    generator = torch.Generator().manual_seed(0)
    rows = []
    canonical_created = 0
    base_created = 0
    for position, family in enumerate(selected, 1):
        cache_token = hashlib.sha256(
            str(family.family_id).encode("utf-8")
        ).hexdigest()[:20]
        canonical_existed = any(canonical_dir.glob(f"{cache_token}_*.pt"))
        canonical_features, canonical_hash = base_collector.load_canonical_features(
            family
        )
        if canonical_features is None or canonical_hash is None:
            raise RuntimeError(f"canonical features missing for {family.family_id}")
        canonical_path = canonical_dir / f"{cache_token}_{canonical_hash[:16]}.pt"
        if not canonical_path.is_file():
            raise RuntimeError(f"canonical collector did not create {canonical_path}")
        canonical_was_created = not canonical_existed
        canonical_created += int(canonical_was_created)
        created_banks = []
        existing_banks = []
        for bank_id in range(int(config.latent_bank_count)):
            bank_path = frozen_bank_cache_path(
                cache_dir,
                cache_key=base_key,
                family_id=family.family_id,
                latent_bank_id=bank_id,
                num_aleatory=int(config.k_train),
            )
            if bank_path.exists():
                existing_banks.append(bank_id)
                continue
            collector(
                family,
                num_aleatory=int(config.k_train),
                generator=generator,
                latent_bank_id=bank_id,
            )
            if not bank_path.is_file():
                raise RuntimeError(f"cache collector did not create {bank_path}")
            created_banks.append(bank_id)
            base_created += 1
        rows.append(
            {
                "family_id": family.family_id,
                "canonical_created": canonical_was_created,
                "canonical_latent_hash": canonical_hash,
                "base_banks_created": created_banks,
                "base_banks_existing": existing_banks,
            }
        )
        LOG.info(
            "shard %d/%d family %d/%d %s canonical_new=%s base_new=%s",
            shard_index,
            shard_count,
            position,
            len(selected),
            family.family_id,
            canonical_was_created,
            created_banks,
        )
        del canonical_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    collector.close()
    manifest = {
        "schema_version": "neon_cache_warm_manifest_v1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "repository_head": os.environ.get("NEON_EXPECTED_HEAD", ""),
        "ladder_rung": rung.upper(),
        "cache_dir": str(cache_dir),
        "base_cache_key": base_key,
        "canonical_namespace": canonical_namespace,
        "canonical_dir": str(canonical_dir),
        "canonical_k": int(config.deterministic_head_canonical_k),
        "canonical_seed": int(config.deterministic_head_latent_seed),
        "latent_bank_count": int(config.latent_bank_count),
        "k_train": int(config.k_train),
        "all_family_count": len(families),
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_family_count": len(selected),
        "canonical_created": canonical_created,
        "base_banks_created": base_created,
        "elapsed_seconds": time.time() - started,
        "families": rows,
    }
    manifest_path = output_dir / f"shard_{shard_index:03d}.json"
    _atomic_json(manifest_path, manifest)
    _write_checksum(manifest_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
