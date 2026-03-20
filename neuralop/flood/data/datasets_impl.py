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
    read_hec_ras_hdf_run_series,
    h5py,
    read_hec_ras_hdf_slice,
    read_hec_ras_hdf_static,
)
from neuralop.flood.utils.runtime_core import (
    describe_boundary_spec,
    get_boundary_channel_count,
    normalize_boundary_spec,
    parse_target_variables,
    resolve_family_id_for_boundary,
    write_train_txt_from_data_root,
)


def _load_clean_boundary_channels_runtime(boundary_spec):
    # Import lazily to avoid the package-level runtime_core <-> flood.data cycle
    # that only shows up in some maintained entrypoint import orders.
    from neuralop.flood.utils.runtime_core import _load_clean_boundary_channels

    return _load_clean_boundary_channels(boundary_spec)

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
        boundary_spec=None,
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
        self.boundary_spec = normalize_boundary_spec(
            boundary=boundary_spec,
            boundary_source=boundary_source,
            clean_boundary_root=clean_boundary_root,
            clean_boundary_file=clean_boundary_file,
            section_root=self.data_root,
        )
        self.boundary_source = (
            "multi_channel"
            if get_boundary_channel_count(self.boundary_spec) > 1
            else self.boundary_spec[0]["mode"]
        )
        self.clean_boundary_root = clean_boundary_root
        self.clean_boundary_file = clean_boundary_file
        self._clean_boundary_bundles = _load_clean_boundary_channels_runtime(self.boundary_spec)
        self._member_hdf_boundary_spec = [
            channel for channel in self.boundary_spec if channel["mode"] == "member_hdf"
        ]

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
        self.full_cell_count = None
        self.xy_coords = None
        self.static_data = None
        self.cell_point_index = None  # index into full-cell arrays to get Cell Points subset
        self.sample_index = []
        self.run_time_steps = {}
        self.structural_dry_mask = None
        self._load_static_and_build_indices()

    def _hdf_file(self, run_id: str) -> Path:
        return self.data_root / f"{run_id}{self.hdf_suffix}"

    def _resolve_boundary_matrix(
        self,
        run_id: str,
        inflow: np.ndarray,
        *,
        slice_start: int = 0,
        slice_end: int | None = None,
    ) -> np.ndarray:
        if inflow.ndim == 1:
            inflow = inflow[:, None]
        if slice_end is None:
            slice_end = inflow.shape[0]

        member_col = 0
        columns = []
        for channel in self.boundary_spec:
            if channel["mode"] == "member_hdf":
                if member_col >= inflow.shape[1]:
                    raise ValueError(
                        f"member_hdf boundary matrix for run {run_id!r} has {inflow.shape[1]} columns "
                        f"but boundary spec requires at least {member_col + 1}: {describe_boundary_spec(self.boundary_spec)}."
                    )
                columns.append(np.asarray(inflow[:, member_col], dtype=np.float32))
                member_col += 1
                continue

            bundle = self._clean_boundary_bundles[str(channel["name"])]
            boundary_by_family = bundle["boundary_by_family"]
            family_id = resolve_family_id_for_boundary(run_id, boundary_by_family)
            clean_series = np.asarray(boundary_by_family[family_id], dtype=np.float32)
            total_len = clean_series.shape[0]
            if slice_start < 0 or slice_end < slice_start or slice_end > total_len:
                raise ValueError(
                    f"Clean boundary slice [{slice_start}:{slice_end}] is invalid for family {family_id!r} "
                    f"channel {channel['name']!r}: available length={total_len} in {bundle['path']}."
                )
            columns.append(clean_series[slice_start:slice_end])

        if not columns:
            return np.zeros((slice_end - slice_start, 0), dtype=np.float32)
        return np.stack(columns, axis=1).astype(np.float32, copy=False)

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
            static_list.append((str(fpath), arr))

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
                geom, _, _, _, _ = read_hec_ras_hdf_slice(
                    hpath,
                    0,
                    1,
                    self.hdf_paths,
                    cell_index=None,
                    boundary_channels=[],
                )
                self.xy_coords = torch.tensor(geom, device="cpu")
                elev_full, area_full = read_hec_ras_hdf_static(hpath, self.hdf_paths, cell_index=None)
                self.full_cell_count = int(elev_full.shape[0])
                elev = elev_full[self.cell_point_index]
                area = area_full[self.cell_point_index]
                # Static order: [elevation, area, CS, CU, FA] (aligned with HEC_RAS_Automation CE, CA + text)
                static_parts = [elev.reshape(-1, 1), area.reshape(-1, 1)]
                if static_list:
                    static_parts.extend(
                        [
                            _align_static_text_to_reference_cells(
                                arr,
                                reference_cell_count=self.reference_cell_count,
                                cell_point_index=self.cell_point_index,
                                full_cell_count=self.full_cell_count,
                                source=source,
                            )
                            for source, arr in static_list
                        ]
                    )
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
            self.run_time_steps[run_id] = int(n_time)
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

    def _load_run_aligned_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        """Load aligned full-run target and boundary arrays without window materialization."""
        hpath = self._hdf_file(run_id)
        if not hpath.exists():
            raise FileNotFoundError(f"HDF not found for run {run_id!r}: {hpath}")
        wd, vx, vy, inflow = read_hec_ras_hdf_run_series(
            hpath,
            self.hdf_paths,
            cell_index=self.cell_point_index,
            boundary_channels=self._member_hdf_boundary_spec,
        )
        n_time = int(self.run_time_steps.get(run_id, wd.shape[0]))
        boundary = self._resolve_boundary_matrix(run_id, inflow, slice_start=0, slice_end=n_time)
        target_all = [wd, vx, vy]
        target = np.stack([target_all[i] for i in self.target_indices], axis=-1).astype(np.float32, copy=False)
        return {
            "target": target,
            "boundary": boundary.astype(np.float32, copy=False),
            "n_time": n_time,
            "n_cells": int(target.shape[1]),
        }

    def __len__(self):
        return len(self.sample_index)

    def set_structural_dry_mask(self, dry_mask: torch.Tensor | np.ndarray | None):
        if dry_mask is None:
            self.structural_dry_mask = None
            return
        mask = torch.as_tensor(dry_mask, dtype=torch.bool, device="cpu").reshape(-1)
        if self.reference_cell_count is None:
            raise RuntimeError("reference_cell_count is not initialized yet.")
        if int(mask.numel()) != int(self.reference_cell_count):
            raise ValueError(
                f"structural_dry_mask length {mask.numel()} does not match reference_cell_count "
                f"{self.reference_cell_count}."
            )
        self.structural_dry_mask = mask

    def __getitem__(self, idx):
        run_id, target_t = self.sample_index[idx]
        t0 = target_t - self.n_history
        t1 = target_t + self.ar_rollout_steps
        hpath = self._hdf_file(run_id)
        geom, wd, vx, vy, inflow = read_hec_ras_hdf_slice(
            hpath,
            t0,
            t1,
            self.hdf_paths,
            cell_index=self.cell_point_index,
            boundary_channels=self._member_hdf_boundary_spec,
        )
        n_cells = geom.shape[0]
        boundary_matrix = self._resolve_boundary_matrix(run_id, inflow, slice_start=t0, slice_end=t1)
        n_boundary_channels = boundary_matrix.shape[1]
        # History: first n_history steps
        hist_all = [wd[: self.n_history], vx[: self.n_history], vy[: self.n_history]]
        dynamic_hist = np.stack([hist_all[i] for i in self.target_indices], axis=-1)
        dynamic_hist = torch.tensor(dynamic_hist, device="cpu", dtype=torch.float32)
        dynamic_hist = self._apply_noise(dynamic_hist)
        flow_hist = boundary_matrix[: self.n_history][:, np.newaxis, :]
        inflow_bc = np.broadcast_to(flow_hist, (self.n_history, n_cells, n_boundary_channels))
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
            flow_val = np.asarray(boundary_matrix[self.n_history + s], dtype=np.float32)
            bc_s = np.broadcast_to(flow_val[np.newaxis, :], (n_cells, n_boundary_channels))
            boundary_sequence_list.append(torch.tensor(bc_s, device="cpu", dtype=torch.float32))
        target_sequence = torch.stack(target_sequence_list, dim=0)
        boundary_sequence = torch.stack(boundary_sequence_list, dim=0)
        in_geom = self.xy_coords if self.xy_coords is not None else torch.tensor(geom, device="cpu", dtype=torch.float32)
        static_feats = self.static_data
        out = {
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
        if self.structural_dry_mask is not None:
            out["structural_dry_mask"] = self.structural_dry_mask
        return out


def _align_static_text_to_reference_cells(
    arr: np.ndarray,
    *,
    reference_cell_count: int,
    cell_point_index: np.ndarray,
    full_cell_count: int | None,
    source: str,
) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Static text array from {source} must be 2D, got shape {arr.shape}.")

    n_rows = int(arr.shape[0])
    if n_rows == int(reference_cell_count):
        return arr

    if full_cell_count is not None and n_rows == int(full_cell_count):
        return np.asarray(arr[cell_point_index, :], dtype=arr.dtype)

    raise ValueError(
        f"Static text file {source} has {n_rows} rows, which does not match either "
        f"reference_cell_count={reference_cell_count} or full_cell_count={full_cell_count}. "
        "Provide text statics on the Cell Points grid or the full Cells Center grid."
    )


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
        boundary_spec=None,
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
        self.boundary_spec = normalize_boundary_spec(
            boundary=boundary_spec,
            boundary_source=boundary_source,
            clean_boundary_root=clean_boundary_root,
            clean_boundary_file=clean_boundary_file,
            section_root=self.data_root,
        )
        self.boundary_source = (
            "multi_channel"
            if get_boundary_channel_count(self.boundary_spec) > 1
            else self.boundary_spec[0]["mode"]
        )
        self.clean_boundary_root = clean_boundary_root
        self.clean_boundary_file = clean_boundary_file
        self._clean_boundary_bundles = _load_clean_boundary_channels_runtime(self.boundary_spec)
        self._member_hdf_boundary_spec = [
            channel for channel in self.boundary_spec if channel["mode"] == "member_hdf"
        ]

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
        self._full_cell_count = None
        self.structural_dry_mask = None
        self._load_static_and_validate_runs()

    def _hdf_file(self, run_id: str) -> Path:
        return self.data_root / f"{run_id}{self.hdf_suffix}"

    def _resolve_boundary_matrix(
        self,
        run_id: str,
        inflow: np.ndarray,
        *,
        slice_start: int = 0,
        slice_end: int | None = None,
    ) -> np.ndarray:
        if inflow.ndim == 1:
            inflow = inflow[:, None]
        if slice_end is None:
            slice_end = inflow.shape[0]

        member_col = 0
        columns = []
        for channel in self.boundary_spec:
            if channel["mode"] == "member_hdf":
                if member_col >= inflow.shape[1]:
                    raise ValueError(
                        f"member_hdf boundary matrix for run {run_id!r} has {inflow.shape[1]} columns "
                        f"but boundary spec requires at least {member_col + 1}: {describe_boundary_spec(self.boundary_spec)}."
                    )
                columns.append(np.asarray(inflow[:, member_col], dtype=np.float32))
                member_col += 1
                continue

            bundle = self._clean_boundary_bundles[str(channel["name"])]
            boundary_by_family = bundle["boundary_by_family"]
            family_id = resolve_family_id_for_boundary(run_id, boundary_by_family)
            clean_series = np.asarray(boundary_by_family[family_id], dtype=np.float32)
            total_len = clean_series.shape[0]
            if slice_start < 0 or slice_end < slice_start or slice_end > total_len:
                raise ValueError(
                    f"Clean boundary slice [{slice_start}:{slice_end}] is invalid for family {family_id!r} "
                    f"channel {channel['name']!r}: available length={total_len} in {bundle['path']}."
                )
            columns.append(clean_series[slice_start:slice_end])

        if not columns:
            return np.zeros((slice_end - slice_start, 0), dtype=np.float32)
        return np.stack(columns, axis=1).astype(np.float32, copy=False)

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
            static_list.append((str(fpath), arr))
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
                geom, _, _, _, _ = read_hec_ras_hdf_slice(
                    hpath,
                    0,
                    1,
                    self.hdf_paths,
                    cell_index=None,
                    boundary_channels=[],
                )
                self.xy_coords = torch.tensor(geom, device="cpu")
                elev_full, area_full = read_hec_ras_hdf_static(hpath, self.hdf_paths, cell_index=None)
                self._full_cell_count = int(elev_full.shape[0])
                elev = elev_full[self.cell_point_index]
                area = area_full[self.cell_point_index]
                static_parts = [elev.reshape(-1, 1), area.reshape(-1, 1)]
                if static_list:
                    static_parts.extend(
                        [
                            _align_static_text_to_reference_cells(
                                arr,
                                reference_cell_count=self._reference_cell_count,
                                cell_point_index=self.cell_point_index,
                                full_cell_count=self._full_cell_count,
                                source=source,
                            )
                            for source, arr in static_list
                        ]
                    )
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

    def set_structural_dry_mask(self, dry_mask: torch.Tensor | np.ndarray | None):
        if dry_mask is None:
            self.structural_dry_mask = None
            return
        mask = torch.as_tensor(dry_mask, dtype=torch.bool, device="cpu").reshape(-1)
        if self._reference_cell_count is None:
            raise RuntimeError("_reference_cell_count is not initialized yet.")
        if int(mask.numel()) != int(self._reference_cell_count):
            raise ValueError(
                f"structural_dry_mask length {mask.numel()} does not match reference cell count "
                f"{self._reference_cell_count}."
            )
        self.structural_dry_mask = mask

    def __getitem__(self, idx):
        run_id = self.valid_run_ids[idx]
        hpath = self._hdf_file(run_id)
        n_cells, n_time = get_hec_ras_hdf_shape(hpath, self.hdf_paths)
        geom, wd, vx, vy, inflow = read_hec_ras_hdf_slice(
            hpath,
            0,
            n_time,
            self.hdf_paths,
            cell_index=self.cell_point_index,
            boundary_channels=self._member_hdf_boundary_spec,
        )
        dynamic = np.stack([wd, vx, vy], axis=-1)
        dynamic = torch.tensor(dynamic, device="cpu", dtype=torch.float32)
        boundary_matrix = self._resolve_boundary_matrix(run_id, inflow, slice_start=0, slice_end=n_time)
        n_boundary_channels = boundary_matrix.shape[1]
        flow_col = boundary_matrix[:, np.newaxis, :]
        boundary = np.broadcast_to(flow_col, (n_time, n_cells, n_boundary_channels))
        boundary = torch.tensor(boundary, device="cpu", dtype=torch.float32)
        out = {
            "run_id": run_id,
            "dynamic": dynamic,
            "boundary": boundary,
            "geometry": self.geometry,
            "static": self.static,
        }
        if self.structural_dry_mask is not None:
            out["structural_dry_mask"] = self.structural_dry_mask
        return out


###############################################################################
# 4) Normalization Helpers
###############################################################################
