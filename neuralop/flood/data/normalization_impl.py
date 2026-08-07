"""Normalization helpers and normalized dataset wrappers for WV flood data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from tqdm import tqdm

from neuralop.data.transforms.normalizers import UnitGaussianNormalizer
from neuralop.flood.data.structural_dry import dry_mask_to_wettable_mask

def collect_all_fields(dataset, expect_target=True):
    geometry_list = []
    static_list = []
    boundary_list = []
    dynamic_list = []
    target_list = []
    for i in range(len(dataset)):
        sample = dataset[i]
        geometry_list.append(sample["geometry"])
        static_list.append(sample["static"])
        boundary_list.append(sample["boundary"])
        dynamic_list.append(sample["dynamic"])
        if expect_target:
            target_list.append(sample.get("target", None))
    return geometry_list, static_list, boundary_list, dynamic_list, target_list


###############################################################################
# 4b) Streaming normalizer fit (avoids stacking full dataset in RAM)
###############################################################################
class _MaskedChannelAccumulator:
    """Streaming per-channel stats over the wettable domain."""

    def __init__(self):
        self.sum = None
        self.sq_sum = None
        self.count = None

    def update(self, batch: torch.Tensor, dry_mask: torch.Tensor) -> None:
        batch = batch.to(dtype=torch.float32)
        dry_mask = torch.as_tensor(dry_mask, dtype=torch.bool, device=batch.device)
        wettable = dry_mask_to_wettable_mask(dry_mask)
        if batch.ndim == 3:
            if wettable.ndim == 1:
                wettable = wettable.unsqueeze(0).expand(batch.shape[0], -1)
            expanded = wettable.unsqueeze(-1).to(dtype=batch.dtype)
        elif batch.ndim == 4:
            if wettable.ndim == 1:
                wettable = wettable.unsqueeze(0).expand(batch.shape[0], -1)
            expanded = wettable.unsqueeze(1).unsqueeze(-1).to(dtype=batch.dtype)
        else:
            raise ValueError(
                f"Masked water-state normalization only supports 3D/4D batches, got {tuple(batch.shape)}."
            )
        expanded = expanded.expand_as(batch)
        masked = batch * expanded
        reduce_dims = tuple(range(batch.ndim - 1))
        chunk_sum = masked.sum(dim=reduce_dims)
        chunk_sq_sum = masked.pow(2).sum(dim=reduce_dims)
        chunk_count = expanded.sum(dim=reduce_dims)
        if self.sum is None:
            self.sum = chunk_sum
            self.sq_sum = chunk_sq_sum
            self.count = chunk_count
        else:
            self.sum = self.sum + chunk_sum
            self.sq_sum = self.sq_sum + chunk_sq_sum
            self.count = self.count + chunk_count

    def to_normalizer(self) -> UnitGaussianNormalizer:
        if self.sum is None or self.sq_sum is None or self.count is None:
            raise RuntimeError("No masked water-state samples were accumulated.")
        count = self.count.clamp_min(1.0)
        mean = self.sum / count
        var = torch.clamp((self.sq_sum / count) - mean.pow(2), min=0.0)
        std = torch.sqrt(var)
        return UnitGaussianNormalizer(
            mean=mean.view(1, 1, -1),
            std=std.view(1, 1, -1),
            dim=[0, 1],
        )


class _ExactChannelAccumulator:
    """Exact per-channel moments with streaming-compatible std semantics."""

    def __init__(self, *, view_shape, dim):
        self.view_shape = tuple(view_shape)
        self.dim = list(dim)
        self.sum = None
        self.sq_sum = None
        self.count = 0.0

    def update(self, sum_vec, sq_sum_vec, count: float) -> None:
        sum_arr = np.asarray(sum_vec, dtype=np.float64)
        sq_sum_arr = np.asarray(sq_sum_vec, dtype=np.float64)
        if self.sum is None:
            self.sum = sum_arr
            self.sq_sum = sq_sum_arr
        else:
            self.sum = self.sum + sum_arr
            self.sq_sum = self.sq_sum + sq_sum_arr
        self.count += float(count)

    def to_normalizer(self) -> UnitGaussianNormalizer:
        if self.sum is None or self.sq_sum is None or self.count <= 0:
            raise RuntimeError("No samples were accumulated for exact normalizer fitting.")
        count = float(max(self.count, 1.0))
        mean = self.sum / count
        if count > 1.0:
            m2 = np.clip(self.sq_sum - count * np.square(mean), a_min=0.0, a_max=None)
            std = np.sqrt(m2 / (count - 1.0))
        else:
            std = np.zeros_like(mean)
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(self.view_shape)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(self.view_shape)
        return UnitGaussianNormalizer(mean=mean_t, std=std_t, dim=self.dim)


def _repo_git_sha() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    sha = result.stdout.strip()
    return sha or None


def _json_safe(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_base_dataset_and_indices(dataset):
    selected = None
    current = dataset
    while isinstance(current, Subset):
        current_indices = np.asarray(current.indices, dtype=np.int64)
        if selected is None:
            selected = current_indices
        else:
            selected = current_indices[selected]
        current = current.dataset
    if selected is None:
        selected = np.arange(len(current), dtype=np.int64)
    return current, selected


def _supports_fast_exact(dataset, structural_dry_policy: str) -> bool:
    if str(structural_dry_policy).strip().lower() != "legacy_full_domain":
        return False
    base_dataset, _ = _resolve_base_dataset_and_indices(dataset)
    return all(
        hasattr(base_dataset, attr)
        for attr in (
            "sample_index",
            "xy_coords",
            "static_data",
            "boundary_spec",
            "target_variables",
            "_load_run_aligned_arrays",
        )
    )


def resolve_normalizer_fit_method(
    dataset,
    *,
    method: str = "auto",
    structural_dry_policy: str = "legacy_full_domain",
) -> str:
    normalized = str(method or "auto").strip().lower()
    if normalized not in {"auto", "fast_exact", "streaming"}:
        raise ValueError(
            f"Unknown normalizer fit method {method!r}. Expected one of ['auto', 'fast_exact', 'streaming']."
        )
    if normalized == "streaming":
        return "streaming"
    supported = _supports_fast_exact(dataset, structural_dry_policy)
    if normalized == "auto":
        return "fast_exact" if supported else "streaming"
    if not supported:
        raise ValueError(
            "fast_exact normalizer fitting is only supported for maintained FloodDatasetHDF "
            "(or Subset[FloodDatasetHDF]) with structural_dry.policy='legacy_full_domain'."
        )
    return "fast_exact"


def _count_boundary_history_usage(target_indices: np.ndarray, *, n_history: int, n_time: int) -> np.ndarray:
    diff = np.zeros(int(n_time) + 1, dtype=np.int64)
    for target_t in np.asarray(target_indices, dtype=np.int64):
        start = int(target_t) - int(n_history)
        end = int(target_t)
        diff[start] += 1
        diff[end] -= 1
    return np.cumsum(diff[:-1], dtype=np.int64)


def fit_normalizers_fast_exact(
    dataset,
    *,
    expect_target: bool = True,
    structural_dry_policy: str = "legacy_full_domain",
):
    if str(structural_dry_policy).strip().lower() != "legacy_full_domain":
        raise ValueError(
            "fast_exact normalizer fitting currently supports only structural_dry.policy='legacy_full_domain'."
        )

    base_dataset, selected_indices = _resolve_base_dataset_and_indices(dataset)
    if len(selected_indices) == 0:
        return {}
    if not _supports_fast_exact(dataset, structural_dry_policy):
        raise ValueError(
            "fast_exact normalizer fitting requires maintained FloodDatasetHDF accessors."
        )

    sample_count = int(selected_indices.shape[0])
    geometry = torch.as_tensor(base_dataset.xy_coords, dtype=torch.float32).cpu().numpy()
    static = torch.as_tensor(base_dataset.static_data, dtype=torch.float32).cpu().numpy()
    n_cells = int(geometry.shape[0])

    normalizers = {
        "geometry": _ExactChannelAccumulator(view_shape=(1, 1, geometry.shape[1]), dim=[0, 1]),
        "static": _ExactChannelAccumulator(view_shape=(1, 1, static.shape[1]), dim=[0, 1]),
        "boundary": _ExactChannelAccumulator(view_shape=(1, 1, 1, len(base_dataset.boundary_spec)), dim=[0, 1, 2]),
    }
    if expect_target:
        normalizers["target"] = _ExactChannelAccumulator(
            view_shape=(1, 1, len(base_dataset.target_variables)),
            dim=[0, 1],
        )

    geom64 = geometry.astype(np.float64, copy=False)
    static64 = static.astype(np.float64, copy=False)
    repeated_count = float(sample_count * n_cells)
    normalizers["geometry"].update(
        sample_count * geom64.sum(axis=0),
        sample_count * np.square(geom64).sum(axis=0),
        repeated_count,
    )
    normalizers["static"].update(
        sample_count * static64.sum(axis=0),
        sample_count * np.square(static64).sum(axis=0),
        repeated_count,
    )

    grouped_by_run: dict[str, list[int]] = {}
    for sample_idx in np.asarray(selected_indices, dtype=np.int64):
        run_id, target_t = base_dataset.sample_index[int(sample_idx)]
        grouped_by_run.setdefault(str(run_id), []).append(int(target_t))

    for run_id, target_ts in grouped_by_run.items():
        run_payload = base_dataset._load_run_aligned_arrays(run_id)
        target_counts = np.bincount(
            np.asarray(target_ts, dtype=np.int64),
            minlength=int(run_payload["n_time"]),
        ).astype(np.float64, copy=False)

        if expect_target:
            target = np.asarray(run_payload["target"], dtype=np.float64)
            normalizers["target"].update(
                np.einsum("t,tnc->c", target_counts, target),
                np.einsum("t,tnc->c", target_counts, np.square(target)),
                float(target_counts.sum() * run_payload["n_cells"]),
            )

        boundary = np.asarray(run_payload["boundary"], dtype=np.float64)
        boundary_counts = _count_boundary_history_usage(
            np.asarray(target_ts, dtype=np.int64),
            n_history=int(base_dataset.n_history),
            n_time=int(run_payload["n_time"]),
        ).astype(np.float64, copy=False)
        normalizers["boundary"].update(
            np.einsum("t,tc->c", boundary_counts, boundary) * float(run_payload["n_cells"]),
            np.einsum("t,tc->c", boundary_counts, np.square(boundary)) * float(run_payload["n_cells"]),
            float(boundary_counts.sum() * run_payload["n_cells"]),
        )

    out = {key: acc.to_normalizer() for key, acc in normalizers.items()}
    if "target" in out:
        out["dynamic"] = out["target"]
    return out


def fit_normalizers(
    dataset,
    chunk_size=1000,
    expect_target=True,
    structural_dry_policy: str = "legacy_full_domain",
    method: str = "auto",
    return_method: bool = False,
):
    resolved_method = resolve_normalizer_fit_method(
        dataset,
        method=method,
        structural_dry_policy=structural_dry_policy,
    )
    if resolved_method == "fast_exact":
        normalizers = fit_normalizers_fast_exact(
            dataset,
            expect_target=expect_target,
            structural_dry_policy=structural_dry_policy,
        )
    else:
        normalizers = fit_normalizers_streaming(
            dataset,
            chunk_size=chunk_size,
            expect_target=expect_target,
            structural_dry_policy=structural_dry_policy,
        )
    if return_method:
        return normalizers, resolved_method
    return normalizers


def resolve_normalizer_metadata_path(normalizer_path: Path | str) -> Path:
    path = Path(normalizer_path)
    return path.with_name(f"{path.stem}.metadata.json")


def build_normalizer_metadata(
    dataset,
    *,
    structural_dry_policy: str,
    fit_method: str,
) -> dict[str, Any]:
    base_dataset, selected_indices = _resolve_base_dataset_and_indices(dataset)
    boundary_spec = _json_safe(getattr(base_dataset, "boundary_spec", None))
    boundary_spec_payload = json.dumps(boundary_spec, sort_keys=True, separators=(",", ":"))
    split_indices = np.asarray(selected_indices, dtype=np.int64)
    return {
        "fit_method": str(fit_method),
        "dataset_root": str(getattr(base_dataset, "data_root", "")),
        "dataset_class": type(base_dataset).__name__,
        "structural_dry_policy": str(structural_dry_policy),
        "target_variables": list(_json_safe(getattr(base_dataset, "target_variables", []))),
        "static_text_files": list(_json_safe(getattr(base_dataset, "static_text_files", []))),
        "boundary_spec": boundary_spec,
        "boundary_spec_fingerprint": hashlib.sha256(boundary_spec_payload.encode("utf-8")).hexdigest(),
        "split_sample_count": int(split_indices.shape[0]),
        "split_fingerprint": hashlib.sha256(split_indices.tobytes()).hexdigest(),
        "code_version": _repo_git_sha(),
    }


def normalizer_metadata_matches(expected: dict[str, Any], actual: dict[str, Any] | None) -> bool:
    """Low-level "do these two normalizer-metadata dicts agree" predicate.

    .. deprecated::
        Use :func:`neuralop.flood.data.normalizer_lifecycle.resolve_normalizer_artifact`
        for the full cached-or-refit-or-keep-on-resume decision. That helper
        owns the resume-safety contract end-to-end; calling
        ``normalizer_metadata_matches`` directly from training code lost the
        ``is_resuming`` context and was the root cause of the May 2026
        ens02/ens03 silent miscalibration incident.

        Callers that legitimately need a strict equality check (e.g., the
        diffusion ``_wait_for_normalizer_artifacts`` strict-mode opt-in, and
        the standalone normalizer-prep CLIs) may continue to call this
        function but should wrap with ``warnings.catch_warnings()`` to
        suppress the deprecation noise.
    """
    import warnings

    warnings.warn(
        "normalizer_metadata_matches() is a low-level primitive; prefer "
        "neuralop.flood.data.normalizer_lifecycle.resolve_normalizer_artifact() "
        "for trainer code paths. See its docstring for the resume-safe "
        "decision matrix.",
        DeprecationWarning,
        stacklevel=2,
    )
    if actual is None:
        return False
    stable_keys = (
        "fit_method",
        "dataset_root",
        "dataset_class",
        "structural_dry_policy",
        "target_variables",
        "static_text_files",
        "boundary_spec_fingerprint",
        "split_sample_count",
        "split_fingerprint",
        "code_version",
    )
    return all(actual.get(key) == expected.get(key) for key in stable_keys)


def load_normalizer_metadata(metadata_path: Path | str) -> dict[str, Any] | None:
    path = Path(metadata_path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_normalizer_metadata(metadata_path: Path | str, metadata: dict[str, Any]) -> Path:
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def fit_normalizers_streaming(
    dataset,
    chunk_size=1000,
    expect_target=True,
    structural_dry_policy: str = "legacy_full_domain",
):
    """
    Fit UnitGaussianNormalizers by iterating over the dataset in chunks.
    Returns a dict of normalizers (geometry, static, boundary, dynamic, target)
    without ever stacking the full dataset. Use with NormalizedDatasetOnTheFly.

    Normalization is per-channel for fields with multiple channels:
    - geometry (x, y): dim [0, 1] → one mean/std per coordinate
    - static: dim [0, 1] → one mean/std per static feature
    - boundary: dim [0, 1, 2] → one mean/std per boundary channel (last dim preserved)
    - target / dynamic (selected target variables): dim [0, 1] → one mean/std per output channel
    """
    n = len(dataset)
    if n == 0:
        return {}

    # Reduce over batch (0) and spatial dims only; preserve channel dim for per-channel stats.
    # Important for performance: do a single dataset pass per chunk and update all normalizers.
    keys_dims = [
        ("geometry", [0, 1]),      # (B, n_cells, 2) -> (1, 1, 2)
        ("static", [0, 1]),        # (B, n_cells, n_static) -> (1, 1, n_static)
        ("boundary", [0, 1, 2]),   # (B, n_hist, n_cells, bc_dim) -> (1, 1, 1, bc_dim)
        ("target", [0, 1]),        # (B, n_cells, C_out) -> (1, 1, C_out)
    ]
    active_keys_dims = [(k, d) for (k, d) in keys_dims if not (k == "target" and not expect_target)]
    normalizers = {k: UnitGaussianNormalizer(dim=d) for (k, d) in active_keys_dims}
    fitted = {k: False for (k, _) in active_keys_dims}
    masked_primary = str(structural_dry_policy).strip().lower() == "masked_primary"
    water_state_accum = _MaskedChannelAccumulator() if (masked_primary and expect_target) else None

    for start in tqdm(range(0, n, chunk_size), desc="Fitting normalizers (single pass)", leave=False):
        end = min(start + chunk_size, n)
        chunk_samples = [dataset[i] for i in range(start, end)]
        for key, _ in active_keys_dims:
            if masked_primary and key == "target":
                continue
            vals = [s.get(key, None) for s in chunk_samples]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            batch = torch.stack(vals, dim=0)
            if batch.numel() == 0:
                continue
            if not fitted[key]:
                normalizers[key].fit(batch)
                fitted[key] = True
            else:
                normalizers[key].partial_fit(batch, batch_size=batch.shape[0])
            del batch
        if water_state_accum is not None:
            target_vals = [s.get("target", None) for s in chunk_samples]
            target_vals = [v for v in target_vals if v is not None]
            dry_masks = [s.get("structural_dry_mask", None) for s in chunk_samples]
            dry_masks = [m for m in dry_masks if m is not None]
            if target_vals and len(dry_masks) == len(target_vals):
                target_batch = torch.stack(target_vals, dim=0)
                dry_batch = torch.stack([torch.as_tensor(m, dtype=torch.bool) for m in dry_masks], dim=0)
                water_state_accum.update(target_batch, dry_batch)
                dynamic_vals = [s.get("dynamic", None) for s in chunk_samples]
                dynamic_vals = [v for v in dynamic_vals if v is not None]
                if dynamic_vals and len(dynamic_vals) == len(target_vals):
                    dynamic_batch = torch.stack(dynamic_vals, dim=0)
                    water_state_accum.update(dynamic_batch, dry_batch)
        del chunk_samples

    if water_state_accum is not None:
        normalizers["target"] = water_state_accum.to_normalizer()
        fitted["target"] = True
    if "target" in normalizers:
        # Dynamic channels mirror selected target channels, so they share normalizer stats.
        normalizers["dynamic"] = normalizers["target"]
    return normalizers


###############################################################################
# 5) NormalizedDatasetOnTheFly (streaming normalization)
###############################################################################
class NormalizedDatasetOnTheFly(Dataset):
    """
    Wraps a raw dataset and applies normalizers on the fly in __getitem__.
    Avoids stacking the full dataset in RAM; use with fit_normalizers_streaming.
    """

    def __init__(self, raw_dataset, normalizers, query_res=None):
        self.raw_dataset = raw_dataset
        self.normalizers = normalizers
        self.query_res = query_res if query_res is not None else [64, 64]
        self._query_points = None
        self._norm_cache = {}
        for key in ["geometry", "static", "boundary", "dynamic", "target"]:
            norm = self.normalizers.get(key, None)
            if norm is None:
                continue
            self._norm_cache[key] = {
                "mean": norm.mean.detach().cpu(),
                "std": norm.std.detach().cpu(),
                "eps": float(norm.eps),
            }
        if len(raw_dataset) > 0:
            sample0 = raw_dataset[0]
            geom = sample0["geometry"]
            if isinstance(geom, torch.Tensor):
                geom = geom.cpu().numpy()
            else:
                geom = np.asarray(geom)
            x_vals, y_vals = geom[:, 0], geom[:, 1]
            min_x, max_x = x_vals.min(), x_vals.max()
            min_y, max_y = y_vals.min(), y_vals.max()
            tx = np.linspace(min_x, max_x, self.query_res[0], dtype=np.float32)
            ty = np.linspace(min_y, max_y, self.query_res[1], dtype=np.float32)
            grid_x, grid_y = np.meshgrid(tx, ty, indexing="ij")
            q_pts = np.stack([grid_x, grid_y], axis=-1)
            self._query_points = torch.tensor(q_pts, device="cpu", dtype=torch.float32)
        else:
            self._query_points = torch.zeros(
                (self.query_res[0], self.query_res[1], 2), dtype=torch.float32
            )
        # Pre-normalize query points once; they are constant for all samples.
        if "geometry" in self._norm_cache:
            self._query_points = self._normalize("geometry", self._query_points)

    @staticmethod
    def _match_stat_ndim(stat: torch.Tensor, val_ndim: int) -> torch.Tensor:
        out = stat
        while out.ndim > val_ndim and out.shape[0] == 1:
            out = out.squeeze(0)
        while out.ndim < val_ndim:
            out = out.unsqueeze(0)
        return out

    def _normalize(self, key: str, val: torch.Tensor) -> torch.Tensor:
        cache = self._norm_cache.get(key, None)
        if cache is None:
            return val
        mean = self._match_stat_ndim(cache["mean"], val.ndim)
        std = self._match_stat_ndim(cache["std"], val.ndim)
        if mean.device != val.device:
            mean = mean.to(val.device)
            std = std.to(val.device)
        return (val - mean) / (std + cache["eps"])

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        out = {}
        for key in ["geometry", "static", "boundary", "dynamic", "target"]:
            if key not in sample or sample[key] is None:
                continue
            val = sample[key]
            if key in self._norm_cache:
                val = self._normalize(key, val)
            out[key] = val
        # AR sequences: normalize each step with the same normalizer as target / boundary
        if "target_sequence" in sample and sample["target_sequence"] is not None and "target" in self._norm_cache:
            ts = sample["target_sequence"]
            out["target_sequence"] = self._normalize("target", ts)
        elif "target_sequence" in sample and sample["target_sequence"] is not None:
            out["target_sequence"] = sample["target_sequence"]
        if "boundary_sequence" in sample and sample["boundary_sequence"] is not None and "boundary" in self._norm_cache:
            bs = sample["boundary_sequence"]
            out["boundary_sequence"] = self._normalize("boundary", bs)
        elif "boundary_sequence" in sample and sample["boundary_sequence"] is not None:
            out["boundary_sequence"] = sample["boundary_sequence"]
        if "structural_dry_mask" in sample and sample["structural_dry_mask"] is not None:
            out["structural_dry_mask"] = sample["structural_dry_mask"]
        # Family-aware research trainers need stable identities after shuffling.
        # time_index carries the raw HDF frame number, which dispersion pinning
        # uses to look up the matching reference dispersion (and, for AR steps,
        # time_index + step).  These metadata values are not normalized or moved
        # to the accelerator.
        for key in ("run_id", "family_id", "time_index"):
            if key in sample and sample[key] is not None:
                out[key] = sample[key]
        # Latent queries must be in the same (normalized) coordinate system as geometry for GNO.
        out["query_points"] = self._query_points
        return out


###############################################################################
# 6) DataProcessor
###############################################################################

class NormalizedRolloutTestDataset(Dataset):
    def __init__(self, normalized_samples, query_res=None):
        self.normalized_samples = normalized_samples
        self.query_res = query_res if query_res is not None else [64, 64]

        if len(self.normalized_samples) > 0:
            geom_sample = self.normalized_samples[0]["geometry"].cpu().numpy()
            x_vals = geom_sample[:, 0]
            y_vals = geom_sample[:, 1]
            min_x, max_x = x_vals.min(), x_vals.max()
            min_y, max_y = y_vals.min(), y_vals.max()
            tx = np.linspace(min_x, max_x, self.query_res[0], dtype=np.float32)
            ty = np.linspace(min_y, max_y, self.query_res[1], dtype=np.float32)
            grid_x, grid_y = np.meshgrid(tx, ty, indexing="ij")
            q_pts = np.stack([grid_x, grid_y], axis=-1)
            self.query_points = torch.tensor(q_pts, device='cpu')
        else:
            self.query_points = torch.zeros((self.query_res[0], self.query_res[1], 2), dtype=torch.float32)

    def __len__(self):
        return len(self.normalized_samples)

    def __getitem__(self, idx):
        sample = self.normalized_samples[idx]
        out = dict(sample)
        out["query_points"] = self.query_points
        return out
