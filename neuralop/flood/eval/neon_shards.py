"""Interruption-safe persistence and exact aggregation for NEON evaluation.

Each family shard contains the sufficient statistics used by the historical
single-process evaluator.  Shards are committed atomically and carry a digest
of every setting that changes scientific results.  The merger refuses partial,
duplicate, or mixed-provenance inputs before publishing an aggregate.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

EVAL_SHARD_SCHEMA_VERSION = 1
EXECUTION_ONLY_PLAN_KEYS = {
    "output_dir",
    "family_index",
    "shard_dir",
    "shard_only",
    "resume",
    "merge_only",
    "expected_families",
}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash an immutable evaluation input without loading it fully into RAM."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(int(chunk_bytes)):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_eval_signature(plan: dict[str, Any]) -> str:
    """Return a stable digest of settings that affect scientific outputs."""
    scientific = {
        key: value for key, value in plan.items() if key not in EXECUTION_ONLY_PLAN_KEYS
    }
    encoded = json.dumps(scientific, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    """Durably replace *path* without exposing a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def evaluation_shard_path(shard_dir: Path, family_index: int) -> Path:
    return Path(shard_dir) / f"{int(family_index):05d}.json"


def write_evaluation_shard(
    shard_dir: Path,
    *,
    plan: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    family_index: int,
    family_id: str,
    row: dict[str, Any],
    pit_rank: dict[str, Any],
    reliability: dict[str, Any],
    impact_metrics: Optional[dict[str, Any]] = None,
) -> Path:
    """Atomically persist all sufficient statistics for one evaluated family."""
    payload = {
        "schema_version": EVAL_SHARD_SCHEMA_VERSION,
        "scientific_signature": scientific_eval_signature(plan),
        "plan": dict(plan),
        "checkpoint_metadata": {
            key: value
            for key, value in checkpoint_metadata.items()
            if isinstance(value, (str, int, float, bool))
        },
        "family_index": int(family_index),
        "family_id": str(family_id),
        "row": dict(row),
        "pit_rank": pit_rank,
        "reliability": reliability,
        "impact_metrics": impact_metrics,
    }
    path = evaluation_shard_path(Path(shard_dir), family_index)
    atomic_json_dump(payload, path)
    return path


def completed_evaluation_shard(
    path: Path,
    *,
    plan: dict[str, Any],
    family_index: int,
    family_id: str,
) -> bool:
    """Return true only for a complete shard matching this exact evaluation."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return False
    return bool(
        payload.get("schema_version") == EVAL_SHARD_SCHEMA_VERSION
        and payload.get("scientific_signature") == scientific_eval_signature(plan)
        and payload.get("family_index") == int(family_index)
        and payload.get("family_id") == str(family_id)
        and isinstance(payload.get("row"), dict)
        and isinstance(payload.get("pit_rank"), dict)
        and isinstance(payload.get("reliability"), dict)
    )


def _sum_histogram(shards: list[dict[str, Any]], key: str) -> list[int]:
    lengths = {len(shard["pit_rank"][key]) for shard in shards}
    if len(lengths) != 1:
        raise ValueError(f"{key} histogram length mismatch")
    length = lengths.pop()
    return [sum(int(shard["pit_rank"][key][i]) for shard in shards) for i in range(length)]


def merge_evaluation_shards(
    shard_dir: Path,
    *,
    output_path: Path,
    expected_families: Optional[int] = None,
) -> dict[str, Any]:
    """Validate and deterministically aggregate atomic per-family shards."""
    paths = sorted(Path(shard_dir).glob("[0-9][0-9][0-9][0-9][0-9].json"))
    if not paths:
        raise ValueError(f"no evaluation shards found in {shard_dir}")
    shards = [json.loads(path.read_text()) for path in paths]
    if any(s.get("schema_version") != EVAL_SHARD_SCHEMA_VERSION for s in shards):
        raise ValueError("evaluation shard schema mismatch")
    signatures = {s.get("scientific_signature") for s in shards}
    if len(signatures) != 1:
        raise ValueError("evaluation shards have mixed scientific signatures")
    checkpoint_metadata = {
        json.dumps(s.get("checkpoint_metadata", {}), sort_keys=True) for s in shards
    }
    if len(checkpoint_metadata) != 1:
        raise ValueError("evaluation shards have mixed checkpoint metadata")
    indices = [int(s["family_index"]) for s in shards]
    if len(indices) != len(set(indices)):
        raise ValueError("evaluation shards contain duplicate family indices")
    expected = len(shards) if expected_families is None else int(expected_families)
    if indices != list(range(expected)):
        raise ValueError(
            "evaluation shards must contain contiguous family indices "
            f"0..{expected - 1}; found {indices}"
        )
    family_ids = [str(s["family_id"]) for s in shards]
    if len(family_ids) != len(set(family_ids)):
        raise ValueError("evaluation shards contain duplicate family IDs")

    per_family = [dict(s["row"]) for s in shards]
    scalar_keys = [key for key, value in per_family[0].items() if isinstance(value, float)]
    for row in per_family[1:]:
        missing = [key for key in scalar_keys if key not in row]
        if missing:
            raise ValueError(f"per-family scalar key mismatch: missing {missing}")
    aggregate = {
        key: float(sum(float(row[key]) for row in per_family) / len(per_family))
        for key in scalar_keys
    }
    if "spread_error_corr" in aggregate:
        aggregate["spread_error_corr_mean"] = aggregate["spread_error_corr"]

    pit_total = {
        "pit_counts": _sum_histogram(shards, "pit_counts"),
        "rank_counts": _sum_histogram(shards, "rank_counts"),
    }
    if "pit_edges" in shards[0]["pit_rank"]:
        pit_edges = shards[0]["pit_rank"]["pit_edges"]
        if any(
            shard["pit_rank"].get("pit_edges") != pit_edges for shard in shards[1:]
        ):
            raise ValueError("pit_edges mismatch")
        pit_total["pit_edges"] = pit_edges
    reliability_sums: dict[str, list[dict[str, float]]] = {}
    for shard in shards:
        for key, bins in shard["reliability"].items():
            if key not in reliability_sums:
                reliability_sums[key] = [
                    {
                        "bin_lo": float(b["bin_lo"]),
                        "bin_hi": float(b["bin_hi"]),
                        "n": 0.0,
                        "sum_forecast_prob": 0.0,
                        "sum_observed_freq": 0.0,
                    }
                    for b in bins
                ]
            if len(reliability_sums[key]) != len(bins):
                raise ValueError(f"reliability bin mismatch for {key}")
            for acc, b in zip(reliability_sums[key], bins):
                edges = float(b["bin_lo"]), float(b["bin_hi"])
                if (acc["bin_lo"], acc["bin_hi"]) != edges:
                    raise ValueError(f"reliability edge mismatch for {key}")
                acc["n"] += float(b["n"])
                acc["sum_forecast_prob"] += float(b["sum_forecast_prob"])
                acc["sum_observed_freq"] += float(b["sum_observed_freq"])
    reliability_curves = {
        key: [
            {
                "bin_lo": b["bin_lo"],
                "bin_hi": b["bin_hi"],
                "n": int(b["n"]),
                "forecast_prob": b["sum_forecast_prob"] / b["n"] if b["n"] else None,
                "observed_freq": b["sum_observed_freq"] / b["n"] if b["n"] else None,
            }
            for b in bins
        ]
        for key, bins in reliability_sums.items()
    }
    impact_rows = [
        dict(s["impact_metrics"]) for s in shards if s.get("impact_metrics") is not None
    ]
    merged_plan = dict(shards[0]["plan"])
    # Preserve the recorded output/shard locations. Only clear controls that
    # describe an individual array task; replacing every execution-only value
    # with ``False`` previously corrupted ``output_dir`` provenance.
    merged_plan.update(
        {
            "family_index": None,
            "shard_only": False,
            "resume": False,
            "merge_only": False,
            "expected_families": expected,
        }
    )
    payload = {
        "plan": merged_plan,
        "checkpoint_metadata": shards[0]["checkpoint_metadata"],
        "aggregate": aggregate,
        "per_family": per_family,
        "pit_rank": pit_total,
        "reliability": reliability_curves,
        "impact_metrics": impact_rows,
    }
    atomic_json_dump(payload, Path(output_path))
    return payload
