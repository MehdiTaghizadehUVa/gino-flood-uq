"""Normalization helpers and normalized dataset wrappers for WV flood data."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

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
def fit_normalizers_streaming(dataset, chunk_size=1000, expect_target=True):
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

    for start in tqdm(range(0, n, chunk_size), desc="Fitting normalizers (single pass)", leave=False):
        end = min(start + chunk_size, n)
        chunk_samples = [dataset[i] for i in range(start, end)]
        for key, _ in active_keys_dims:
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
        del chunk_samples

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
        sample["query_points"] = self.query_points
        return sample
