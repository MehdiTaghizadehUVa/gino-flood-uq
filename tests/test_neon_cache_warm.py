"""Behavior tests for the sharded NEON frozen-feature cache warmer."""

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "neon_stage2_warm_cache.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("neon_stage2_warm_cache", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cache_warm_shards_cover_each_family_exactly_once():
    warm = _load_script()
    family_ids = [f"TR{i:06d}" for i in range(17)]
    shards = [
        warm.select_shard(family_ids, shard_index=index, shard_count=4)
        for index in range(4)
    ]

    assert sorted(item for shard in shards for item in shard) == family_ids
    assert sum(len(set(shard)) for shard in shards) == len(family_ids)
    assert not any(set(left) & set(right) for i, left in enumerate(shards) for right in shards[i + 1 :])


def test_cache_entry_path_matches_training_cache_contract(tmp_path):
    warm = _load_script()
    path = warm.frozen_bank_cache_path(
        tmp_path,
        cache_key="abc123",
        family_id="Flood/coastal TR000001",
        latent_bank_id=3,
        num_aleatory=8,
    )

    assert path == tmp_path / "abc123_Flood_coastal_TR000001_bank3_k8.pt"
