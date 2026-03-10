"""Dataset implementations for WV flood HDF data."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from neuralop.flood.data.hec_ras import (
    HDF_PATHS,
    build_cell_point_index,
    get_hec_ras_hdf_shape,
    h5py,
    read_hec_ras_hdf_slice,
    read_hec_ras_hdf_static,
)
from neuralop.flood.utils.runtime_core import (
    _load_clean_boundary_table,
    normalize_boundary_source,
    parse_family_id_from_run_id,
    parse_target_variables,
    write_train_txt_from_data_root,
)

class FloodDatasetHDF(Dataset):
    """
    Training/validation dataset: geometry = Cell Points; dynamic/boundary from HDF;
    static = elevation + area from HDF (aligned with HEC_RAS_Automation CE, CA) plus
    text files (e.g. M40_CS.txt, M40_CU.txt, M40_FA.txt).
    Dynamic/target channel order follows ``target_variables`` subset of [WD, VX, VY].
    """

    def __init__(
        self,
        data_root,
        n_history,
        query_res=None,
        run_ids=None,
        train_txt="train.txt",
        static_text_files=None,
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=0,
        noise_type="none",
        noise_std=None,
        hdf_paths=None,
        ar_rollout_steps=1,
        target_variables=None,
        write_train_txt=True,
        boundary_source="member_hdf",
        clean_boundary_root=None,
        clean_boundary_file=None,
    ):
        super().__init__()
        if h5py is None:
            raise ImportError("h5py is required for FloodDatasetHDF. Install with: pip install h5py")
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
        self.n_history = n_history
        self.ar_rollout_steps = max(1, int(ar_rollout_steps))
        self.query_res = query_res or [64, 64]
        self.static_text_files = static_text_files or []
        self.hdf_suffix = hdf_suffix
        self.raise_on_smaller = raise_on_smaller
        self.skip_before_timestep = skip_before_timestep
        self.hdf_paths = hdf_paths or HDF_PATHS
        self.target_variables = parse_target_variables(target_variables)
        self._channel_to_idx = {"wd": 0, "vx": 1, "vy": 2}
        self.target_indices = [self._channel_to_idx[v] for v in self.target_variables]
        self.write_train_txt = bool(write_train_txt)
        self.boundary_source = normalize_boundary_source(boundary_source)
        self.clean_boundary_root = clean_boundary_root
        self.clean_boundary_file = clean_boundary_file
        self._clean_boundary_bundle = None
        if self.boundary_source == "clean_family":
            self._clean_boundary_bundle = _load_clean_boundary_table(
                clean_boundary_root, clean_boundary_file
            )

        if noise_std is None or (isinstance(noise_std, (list, tuple)) and len(noise_std) == 0):
            self.noise_type = "none"
            self.noise_std = [0.0] * len(self.target_indices)
        else:
            self.noise_type = (noise_type or "none").lower()
            ns = list(noise_std)
            if len(ns) == 3:
                # Backward-compatible: provided as [wd, vx, vy], subset to selected targets.
                self.noise_std = [float(ns[i]) for i in self.target_indices]
            elif len(ns) == len(self.target_indices):
                # Provided exactly for selected target variables order.
                self.noise_std = [float(v) for v in ns]
            else:
                raise ValueError(
                    "noise_std must be length 3 ([wd, vx, vy]) or "
                    f"length {len(self.target_indices)} matching target_variables={self.target_variables}."
                )

        if run_ids is not None:
            self.run_ids = [str(r).strip() for r in run_ids if str(r).strip()]
        else:
            train_txt_path = self.data_root / train_txt
            if not train_txt_path.exists():
                if not self.write_train_txt:
                    raise FileNotFoundError(
                        f"train_txt {train_txt_path} does not exist and write_train_txt=false."
                    )
                # Auto-create train.txt from all existing *hdf_suffix in data_root
                self.run_ids = write_train_txt_from_data_root(
                    self.data_root, train_txt=train_txt, hdf_suffix=self.hdf_suffix
                )
            else:
                with open(train_txt_path, "r", encoding="utf-8-sig") as f:
                    lines = [ln.strip() for ln in f if ln.strip()]
                if len(lines) == 1 and "," in lines[0]:
                    self.run_ids = [r.strip() for r in lines[0].split(",") if r.strip()]
                else:
                    self.run_ids = [ln.strip() for ln in lines if ln.strip()]
        if not self.run_ids:
            raise ValueError("No valid run IDs found.")

        self.reference_cell_count = None
        self.xy_coords = None
        self.static_data = None
        self.cell_point_index = None  # index into full-cell arrays to get Cell Points subset
        self.sample_index = []
        self._load_static_and_build_indices()

    def _hdf_file(self, run_id: str) -> Path:
        return self.data_root / f"{run_id}{self.hdf_suffix}"

    def _resolve_boundary_series(
        self,
        run_id: str,
        inflow: np.ndarray,
        *,
        slice_start: int = 0,
        slice_end: int | None = None,
    ) -> np.ndarray:
        if self.boundary_source == "member_hdf":
            if inflow.ndim == 1:
                inflow = inflow[:, None]
            flow_col = inflow[:, -1] if inflow.shape[1] >= 2 else inflow[:, 0]
            return np.asarray(flow_col, dtype=np.float32)

        family_id = parse_family_id_from_run_id(run_id)
        boundary_by_family = self._clean_boundary_bundle["boundary_by_family"]
        if family_id not in boundary_by_family:
            raise KeyError(
                f"Family {family_id!r} not found in clean boundary file "
                f"{self._clean_boundary_bundle['path']}."
            )
        clean_series = np.asarray(boundary_by_family[family_id], dtype=np.float32)
        total_len = clean_series.shape[0]
        if slice_end is None:
            slice_end = total_len
        if slice_start < 0 or slice_end < slice_start or slice_end > total_len:
            raise ValueError(
                f"Clean boundary slice [{slice_start}:{slice_end}] is invalid for family {family_id!r}: "
                f"available length={total_len} in "
                f"{self._clean_boundary_bundle['path']}."
            )
        return clean_series[slice_start:slice_end]

    def _load_static_and_build_indices(self):
        # Text-file static (CS, CU, FA)
        static_list = []
        for fname in self.static_text_files:
            fpath = self.data_root / fname
            if not fpath.exists():
                warnings.warn(f"Static file not found: {fpath}, skipping.")
                continue
            arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            static_list.append(arr)

        for run_id in tqdm(self.run_ids, desc="Building HDF sample indices"):
            hpath = self._hdf_file(run_id)
            if not hpath.exists():
                warnings.warn(f"HDF not found: {hpath}, skipping run_id {run_id}.")
                continue
            try:
                n_cells, n_time = get_hec_ras_hdf_shape(hpath, self.hdf_paths)
            except Exception as e:
                warnings.warn(f"Could not read HDF shape from {hpath}: {e}. Skipping.")
                continue
            if self.reference_cell_count is None:
                self.cell_point_index = build_cell_point_index(hpath, self.hdf_paths)
                self.reference_cell_count = len(self.cell_point_index)
                geom, _, _, _, _ = read_hec_ras_hdf_slice(hpath, 0, 1, self.hdf_paths, cell_index=None)
                self.xy_coords = torch.tensor(geom, device="cpu")
                elev, area = read_hec_ras_hdf_static(hpath, self.hdf_paths, self.cell_point_index)
                # Static order: [elevation, area, CS, CU, FA] (aligned with HEC_RAS_Automation CE, CA + text)
                static_parts = [elev.reshape(-1, 1), area.reshape(-1, 1)]
                if static_list:
                    static_parts.extend([_pad_or_truncate(a, self.reference_cell_count) for a in static_list])
                combined = np.concatenate(static_parts, axis=1)
                self.static_data = torch.tensor(combined, device="cpu")
            elif n_cells != self.reference_cell_count:
                if self.raise_on_smaller:
                    raise ValueError(f"Run {run_id} has {n_cells} cells, expected {self.reference_cell_count}.")
                warnings.warn(f"Run {run_id} cell count {n_cells} != {self.reference_cell_count}, skipping.")
                continue
            start_t = max(self.n_history, self.skip_before_timestep)
            # For AR training we need target_t + ar_rollout_steps <= n_time (exclusive end: indices up to n_time-1)
            end_t = n_time - self.ar_rollout_steps + 1
            for t in range(start_t, end_t):
                self.sample_index.append((run_id, t))

        if self.reference_cell_count is None:
            raise ValueError("No valid HDF files found; could not set reference_cell_count.")
        if self.static_data is None:
            self.static_data = torch.zeros((self.reference_cell_count, 0), device="cpu")

    def _apply_noise(self, dynamic_hist: torch.Tensor):
        if self.noise_type == "none" or all(s <= 0.0 for s in self.noise_std):
            return dynamic_hist
        _, num_cells, d = dynamic_hist.shape
        device = dynamic_hist.device
        std_tensor = torch.tensor(self.noise_std, device=device).view(1, 1, d)
        if self.noise_type == "only_last":
            step_noise = torch.randn((num_cells, d), device=device) * std_tensor[0, 0]
            dynamic_hist = dynamic_hist.clone()
            dynamic_hist[-1] += step_noise
        elif self.noise_type == "uncorrelated":
            n_steps = dynamic_hist.shape[0]
            noise_ = torch.randn((n_steps, num_cells, d), device=device) * std_tensor
            dynamic_hist = dynamic_hist + noise_
        else:
            warnings.warn(f"Unknown noise_type {self.noise_type}, skipping.")
        return dynamic_hist

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        run_id, target_t = self.sample_index[idx]
        t0 = target_t - self.n_history
        t1 = target_t + self.ar_rollout_steps
        hpath = self._hdf_file(run_id)
        geom, wd, vx, vy, inflow = read_hec_ras_hdf_slice(
            hpath, t0, t1, self.hdf_paths, cell_index=self.cell_point_index
        )
        n_cells = geom.shape[0]
        boundary_series = self._resolve_boundary_series(run_id, inflow, slice_start=t0, slice_end=t1)
        # History: first n_history steps
        hist_all = [wd[: self.n_history], vx[: self.n_history], vy[: self.n_history]]
        dynamic_hist = np.stack([hist_all[i] for i in self.target_indices], axis=-1)
        dynamic_hist = torch.tensor(dynamic_hist, device="cpu", dtype=torch.float32)
        dynamic_hist = self._apply_noise(dynamic_hist)
        flow_col = boundary_series[: self.n_history][:, None]
        flow_col = flow_col[:, np.newaxis, :]
        inflow_bc = np.broadcast_to(flow_col, (self.n_history, n_cells, 1))
        bc_hist = torch.tensor(inflow_bc, device="cpu", dtype=torch.float32)
        # Single-step target (first step; for backward compat and normalizer fit)
        tgt_all = [wd[self.n_history], vx[self.n_history], vy[self.n_history]]
        target_all = torch.stack(
            [torch.tensor(tgt_all[i], dtype=torch.float32) for i in self.target_indices],
            dim=-1,
        )
        # AR target sequence and boundary sequence (for AR fine-tuning)
        target_sequence_list = []
        boundary_sequence_list = []
        for s in range(self.ar_rollout_steps):
            ts_all = [wd[self.n_history + s], vx[self.n_history + s], vy[self.n_history + s]]
            target_sequence_list.append(
                torch.stack(
                    [torch.tensor(ts_all[i], dtype=torch.float32) for i in self.target_indices],
                    dim=-1,
                )
            )
            flow_val = float(boundary_series[self.n_history + s])
            bc_s = np.full((n_cells, 1), flow_val, dtype=np.float32)
            boundary_sequence_list.append(torch.tensor(bc_s, device="cpu", dtype=torch.float32))
        target_sequence = torch.stack(target_sequence_list, dim=0)
        boundary_sequence = torch.stack(boundary_sequence_list, dim=0)
        in_geom = self.xy_coords if self.xy_coords is not None else torch.tensor(geom, device="cpu", dtype=torch.float32)
        static_feats = self.static_data
        return {
            "geometry": in_geom,
            "static": static_feats,
            "boundary": bc_hist,
            "dynamic": dynamic_hist,
            "target": target_all,
            "target_sequence": target_sequence,
            "boundary_sequence": boundary_sequence,
            "run_id": run_id,
            "time_index": target_t,
        }


def _pad_or_truncate(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape[0] >= size:
        return arr[:size, :]
    pad = np.zeros((size - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=0)


def _align_static_to_cells(static: torch.Tensor, n_cells: int) -> torch.Tensor:
    n, c = static.shape
    if n == n_cells:
        return static
    if n < n_cells:
        pad = torch.zeros((n_cells - n, c), dtype=static.dtype)
        return torch.cat([static, pad], dim=0)
    return static[:n_cells, :]


###############################################################################
# 3) ROLLOUT TEST DATASET FROM HDF (HEC-RAS results) + text static
###############################################################################
class FloodRolloutTestDatasetHDF(Dataset):
    """
    Rollout evaluation dataset: geometry = Cell Points; dynamic/boundary from HDF;
    static = elevation + area from HDF + text files (CS, CU, FA). Same as FloodDatasetHDF.
    """

    def __init__(
        self,
        rollout_data_root,
        n_history,
        rollout_length,
        run_ids=None,
        test_txt="test.txt",
        static_text_files=None,
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=0,
        hdf_paths=None,
        boundary_source="member_hdf",
        clean_boundary_root=None,
        clean_boundary_file=None,
    ):
        super().__init__()
        if h5py is None:
            raise ImportError("h5py is required for FloodRolloutTestDatasetHDF. Install with: pip install h5py")
        self.data_root = Path(rollout_data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Rollout data root not found: {self.data_root}")
        self.n_history = n_history
        self.rollout_length = rollout_length
        self.static_text_files = static_text_files or []
        self.hdf_suffix = hdf_suffix
        self.raise_on_smaller = raise_on_smaller
        self.skip_before_timestep = skip_before_timestep
        self.hdf_paths = hdf_paths or HDF_PATHS
        self.boundary_source = normalize_boundary_source(boundary_source)
        self.clean_boundary_root = clean_boundary_root
        self.clean_boundary_file = clean_boundary_file
        self._clean_boundary_bundle = None
        if self.boundary_source == "clean_family":
            self._clean_boundary_bundle = _load_clean_boundary_table(
                clean_boundary_root, clean_boundary_file
            )

        if run_ids is not None:
            self.run_ids = [str(r).strip() for r in run_ids if str(r).strip()]
        else:
            test_txt_path = self.data_root / test_txt
            if not test_txt_path.exists():
                raise FileNotFoundError(f"Expected {test_txt} at {test_txt_path}, not found.")
            with open(test_txt_path, "r", encoding="utf-8-sig") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if len(lines) == 1 and "," in lines[0]:
                self.run_ids = [r.strip() for r in lines[0].split(",") if r.strip()]
            else:
                self.run_ids = [ln.strip() for ln in lines if ln.strip()]
        if not self.run_ids:
            raise ValueError("No valid run IDs found.")

        self.valid_run_ids = []
        self.xy_coords = None
        self.static_data = None
        self.cell_point_index = None
        self._reference_cell_count = None
        self._load_static_and_validate_runs()

    def _hdf_file(self, run_id: str) -> Path:
        return self.data_root / f"{run_id}{self.hdf_suffix}"

    def _resolve_boundary_series(
        self,
        run_id: str,
        inflow: np.ndarray,
        *,
        slice_start: int = 0,
        slice_end: int | None = None,
    ) -> np.ndarray:
        if self.boundary_source == "member_hdf":
            if inflow.ndim == 1:
                inflow = inflow[:, None]
            flow_col = inflow[:, -1] if inflow.shape[1] >= 2 else inflow[:, 0]
            return np.asarray(flow_col, dtype=np.float32)

        family_id = parse_family_id_from_run_id(run_id)
        boundary_by_family = self._clean_boundary_bundle["boundary_by_family"]
        if family_id not in boundary_by_family:
            raise KeyError(
                f"Family {family_id!r} not found in clean boundary file "
                f"{self._clean_boundary_bundle['path']}."
            )
        clean_series = np.asarray(boundary_by_family[family_id], dtype=np.float32)
        total_len = clean_series.shape[0]
        if slice_end is None:
            slice_end = total_len
        if slice_start < 0 or slice_end < slice_start or slice_end > total_len:
            raise ValueError(
                f"Clean boundary slice [{slice_start}:{slice_end}] is invalid for family {family_id!r}: "
                f"available length={total_len} in "
                f"{self._clean_boundary_bundle['path']}."
            )
        return clean_series[slice_start:slice_end]

    def _load_static_and_validate_runs(self):
        static_list = []
        for fname in self.static_text_files:
            fpath = self.data_root / fname
            if not fpath.exists():
                warnings.warn(f"Static file not found: {fpath}, skipping.")
                continue
            arr = np.loadtxt(str(fpath), delimiter="\t", dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[:, None]
            static_list.append(arr)
        for run_id in tqdm(self.run_ids, desc="Validating HDF runs for rollout"):
            hpath = self._hdf_file(run_id)
            if not hpath.exists():
                warnings.warn(f"HDF not found: {hpath}, skipping {run_id}.")
                continue
            try:
                n_cells, n_time = get_hec_ras_hdf_shape(hpath, self.hdf_paths)
            except Exception as e:
                warnings.warn(f"Could not read HDF shape from {hpath}: {e}. Skipping.")
                continue
            required = self.skip_before_timestep + self.n_history + self.rollout_length
            if n_time < required:
                if self.raise_on_smaller:
                    raise ValueError(f"Run {run_id} has {n_time} steps, need {required}.")
                warnings.warn(f"Run {run_id} has {n_time} < {required} steps, skipping.")
                continue
            if self._reference_cell_count is None:
                self.cell_point_index = build_cell_point_index(hpath, self.hdf_paths)
                self._reference_cell_count = len(self.cell_point_index)
                geom, _, _, _, _ = read_hec_ras_hdf_slice(hpath, 0, 1, self.hdf_paths, cell_index=None)
                self.xy_coords = torch.tensor(geom, device="cpu")
                elev, area = read_hec_ras_hdf_static(hpath, self.hdf_paths, self.cell_point_index)
                static_parts = [elev.reshape(-1, 1), area.reshape(-1, 1)]
                if static_list:
                    static_parts.extend([_pad_or_truncate(a, self._reference_cell_count) for a in static_list])
                combined = np.concatenate(static_parts, axis=1)
                self.static_data = torch.tensor(combined, device="cpu")
            elif n_cells != self._reference_cell_count:
                warnings.warn(f"Run {run_id} cell count {n_cells} != {self._reference_cell_count}, skipping.")
                continue
            self.valid_run_ids.append(run_id)
        if not self.valid_run_ids:
            raise ValueError("No HDF runs have enough time steps for rollout evaluation.")
        if self.static_data is None:
            self.static_data = torch.zeros((self._reference_cell_count, 0), device="cpu")
        self.geometry = self.xy_coords
        self.static = self.static_data

    def __len__(self):
        return len(self.valid_run_ids)

    def __getitem__(self, idx):
        run_id = self.valid_run_ids[idx]
        hpath = self._hdf_file(run_id)
        n_cells, n_time = get_hec_ras_hdf_shape(hpath, self.hdf_paths)
        geom, wd, vx, vy, inflow = read_hec_ras_hdf_slice(
            hpath, 0, n_time, self.hdf_paths, cell_index=self.cell_point_index
        )
        dynamic = np.stack([wd, vx, vy], axis=-1)
        dynamic = torch.tensor(dynamic, device="cpu", dtype=torch.float32)
        boundary_series = self._resolve_boundary_series(run_id, inflow, slice_start=0, slice_end=n_time)
        flow_col = boundary_series[:, None]  # (n_time, 1)
        flow_col = flow_col[:, np.newaxis, :]  # (n_time, 1, 1) for broadcast over cells
        boundary = np.broadcast_to(flow_col, (n_time, n_cells, 1))
        boundary = torch.tensor(boundary, device="cpu", dtype=torch.float32)
        return {
            "run_id": run_id,
            "dynamic": dynamic,
            "boundary": boundary,
            "geometry": self.geometry,
            "static": self.static,
        }


###############################################################################
# 4) Normalization Helpers
###############################################################################
