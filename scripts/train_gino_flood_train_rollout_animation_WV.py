#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings
import os
import random
import logging
from functools import partial
from logging.handlers import RotatingFileHandler
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
import math
from scipy.spatial import cKDTree

# Config management
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

# Prefer project root for neuralop imports (script may be run from repo root or scripts/)
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Neural Operator imports
from neuralop.training import setup, AdamW
from neuralop.training.trainer import Trainer
from neuralop.losses.data_losses import LpLoss
from neuralop.losses.probabilistic_losses import (
    CRPSLoss,
    GaussianNLLLoss,
    fair_crps_univariate,
    split_gaussian_packed,
)
from neuralop.data.transforms.data_processors import DataProcessor
from neuralop.data.transforms.normalizers import (
    UnitGaussianNormalizer,
    save_normalizers,
    load_normalizers,
)
from neuralop import get_model
from neuralop.utils import get_wandb_api_key

import matplotlib as mpl

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore

mpl.rcParams.update({
    "font.family": "serif",  # use a classic academic serif
    "font.size": 14,  # good general-purpose size
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})


# ---------------------------------------------------------------------------
# Reproducibility: set all RNG seeds and PyTorch deterministic behavior
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set all relevant random seeds and PyTorch/CuDNN settings for full reproducibility.
    Call once at the start of main() after config is loaded.

    When deterministic=True, CuDNN benchmark is disabled so that convolutions etc.
    are deterministic; training may be slightly slower. For extra reproducibility
    (e.g. hash-based dict order), set env PYTHONHASHSEED=0 before starting Python.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def make_dataloader_generator(seed: int):
    """Return a fresh torch.Generator with the given seed for reproducible DataLoader shuffle."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def make_split_generator(seed: int):
    """Return a fresh torch.Generator with the given seed for reproducible random_split."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def parse_target_variables(target_variables):
    """
    Parse configured target variable names.

    Allowed names (case-insensitive): wd, vx, vy.
    Returns a normalized ordered list containing a non-empty subset of these.
    """
    allowed = ("wd", "vx", "vy")
    if target_variables is None:
        return list(allowed)
    out = []
    for v in target_variables:
        key = str(v).strip().lower()
        if key not in allowed:
            raise ValueError(
                f"Unknown target variable '{v}'. Allowed: {allowed}."
            )
        if key not in out:
            out.append(key)
    if not out:
        raise ValueError("target_variables must contain at least one of: wd, vx, vy.")
    return out


def _safe_float(val, default: float) -> float:
    """Convert config value to float with fallback for None/invalid inputs."""
    if val is None:
        return float(default)
    try:
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def dataloader_worker_init(worker_id: int, base_seed: int) -> None:
    """Top-level worker init for Windows multiprocessing pickling compatibility."""
    worker_seed = int(base_seed) + int(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------------------------------
# Logging setup: file + console, optional rotation, config-driven level/path
# ---------------------------------------------------------------------------
LOG_FORMAT_DETAILED = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
)
LOG_FORMAT_CONSOLE = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    log_file_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    log_file_backup_count: int = 3,
    logger_name: str = "flood_train",
) -> logging.Logger:
    """
    Configure and return a logger with optional file (rotating) and console handlers.
    Does not add duplicate handlers if the logger already has them.
    """
    logger = logging.getLogger(logger_name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    # Avoid duplicate handlers when re-calling (e.g. in tests)
    if logger.handlers:
        return logger

    formatter_file = logging.Formatter(LOG_FORMAT_DETAILED, datefmt=LOG_DATEFMT)
    formatter_console = logging.Formatter(LOG_FORMAT_CONSOLE, datefmt=LOG_DATEFMT)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter_console)
    logger.addHandler(ch)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            log_path,
            maxBytes=log_file_max_bytes,
            backupCount=log_file_backup_count,
            encoding="utf-8",
        )
        fh.setLevel(level)
        fh.setFormatter(formatter_file)
        logger.addHandler(fh)
        logger.info("Logging to file: %s", log_path.resolve())

    return logger


###############################################################################
# 1) CONFIG & SETUP
###############################################################################
def load_config_and_setup():
    """
    Reads gino_pluvial_flood_config_WV.yaml (or --config_path <path>) and sets up device.
    Use --config_path to avoid clash with ArgparseConfig's --config_name/--config_file.
    """
    import sys
    config_name = "flood"
    config_path = _REPO_ROOT / "config" / "gino_pluvial_flood_config_WV_depth_only.yaml"
    argv = list(sys.argv[1:])
    for i, a in enumerate(argv):
        if a == "--config_path" and i + 1 < len(argv):
            config_path = Path(argv[i + 1])
            if not config_path.is_absolute():
                config_path = _REPO_ROOT / config_path
            # Remove --config_path and its value so ArgparseConfig does not see them
            idx = sys.argv.index("--config_path")
            sys.argv.pop(idx + 1)
            sys.argv.pop(idx)
            break
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    pipe = ConfigPipeline([
        YamlConfig(str(config_path), config_name=config_name, config_folder=str(_REPO_ROOT / "config")),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
    ])
    config = pipe.read_conf()

    # Setup device (and distributed environment if needed)
    device, is_logger = setup(config)
    return config, device, is_logger


###############################################################################
# 1a) Write train.txt from all HDF run IDs in data_root
###############################################################################
def write_train_txt_from_data_root(
    data_root, train_txt: str = "train.txt", hdf_suffix: str = ".hdf"
):
    """
    Write train.txt with one run ID per line (filename stem of each *hdf_suffix in data_root).
    Returns the list of run_ids. Call when train.txt is missing or to refresh with all existing simulations.
    """
    data_root = Path(data_root)
    run_ids = sorted(p.stem for p in data_root.glob(f"*{hdf_suffix}") if p.is_file())
    out_path = data_root / train_txt
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(run_ids) + ("\n" if run_ids else ""))
    return run_ids


###############################################################################
# 1b) HEC-RAS HDF reader (aligned with HEC_RAS_Automation.py: Cell Points, elevation, area)
###############################################################################
# Geometry: Cell Points (subset of cells; 9953) — result arrays subset by cell_point_index
# Static from HDF: elevation (Cells Minimum Elevation), area (Cells Surface Area)
HDF_PATHS = {
    "geometry": "Geometry/2D Flow Areas/Cell Points",
    "geometry_cell_centers": "Geometry/2D Flow Areas/Flow Area/Cells Center Coordinate",
    "elevation": "Geometry/2D Flow Areas/Flow Area/Cells Minimum Elevation",
    "area": "Geometry/2D Flow Areas/Flow Area/Cells Surface Area",
    "wd": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Hydraulic Depth",
    "vx": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Velocity - Velocity X",
    "vy": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Cell Velocity - Velocity Y",
    "us_inflow": "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/Flow Area/Boundary Conditions/US Inflow",
}


def build_cell_point_index(hdf_path: Path, paths: dict = None) -> np.ndarray:
    """
    Build index such that Cells Center Coordinate[cell_point_index] matches Cell Points.
    Result arrays (wd, vx, vy) and HDF static (elevation, area) have one value per full cell;
    we subset to the Cell Points subset using this index.
    Returns: 1D int array of length n_cell_points (e.g. 9953).
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF data. Install with: pip install h5py")
    paths = paths or HDF_PATHS
    with h5py.File(hdf_path, "r") as f:
        cp = np.asarray(f[paths["geometry"]][:], dtype=np.float32)  # (n_cell_points, 2)
        ccc = np.asarray(f[paths["geometry_cell_centers"]][:], dtype=np.float32)  # (n_full_cells, 2)
    # For each Cell Point find the row index in Cells Center Coordinate (exact match)
    tree = cKDTree(ccc)
    _, idx = tree.query(cp, k=1, distance_upper_bound=0.001)
    if np.any(idx >= ccc.shape[0]):
        raise ValueError("Some Cell Points have no matching Cell Center; check HDF geometry.")
    return np.asarray(idx, dtype=np.intp)


def read_hec_ras_hdf_slice(
    hdf_path: Path,
    t0: int,
    t1: int,
    paths: dict = None,
    cell_index: np.ndarray = None,
) -> tuple:
    """
    Read geometry and time-series slice from a HEC-RAS 2D result HDF file.
    Uses Cell Points for geometry; result arrays (wd, vx, vy) are subset by cell_index
    when provided (so geometry and results have same n_cells = n_cell_points).

    Returns
    -------
    geometry : np.ndarray (n_cell_points, 2) from Cell Points
    wd, vx, vy : np.ndarray (T, n_cell_points)
    inflow : np.ndarray (T, 1 or 2)
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF data. Install with: pip install h5py")
    paths = paths or HDF_PATHS
    with h5py.File(hdf_path, "r") as f:
        geom = np.asarray(f[paths["geometry"]][:], dtype=np.float32)
        wd = np.asarray(f[paths["wd"]][t0:t1, :], dtype=np.float32)
        vx = np.asarray(f[paths["vx"]][t0:t1, :], dtype=np.float32)
        vy = np.asarray(f[paths["vy"]][t0:t1, :], dtype=np.float32)
        if cell_index is not None:
            wd = wd[:, cell_index]
            vx = vx[:, cell_index]
            vy = vy[:, cell_index]
        inflow_ds = f[paths["us_inflow"]]
        inflow = np.asarray(inflow_ds[t0:t1], dtype=np.float32)
        if inflow.ndim == 1:
            inflow = inflow[:, np.newaxis]
    return geom, wd, vx, vy, inflow


def get_hec_ras_hdf_shape(hdf_path: Path, paths: dict = None) -> tuple:
    """Return (n_cell_points, n_time). n_cell_points = Cell Points count (geometry)."""
    if h5py is None:
        raise ImportError("h5py is required for HDF data. Install with: pip install h5py")
    paths = paths or HDF_PATHS
    with h5py.File(hdf_path, "r") as f:
        geom = f[paths["geometry"]]
        n_cells = geom.shape[0]
        wd = f[paths["wd"]]
        n_time = wd.shape[0]
    return n_cells, n_time


def read_hec_ras_hdf_static(hdf_path: Path, paths: dict = None, cell_index: np.ndarray = None) -> tuple:
    """
    Read elevation and area from HDF (aligned with HEC_RAS_Automation: CE, CA).
    Returns (elevation, area) each (n_cell_points,) float32, subset by cell_index if provided.
    """
    if h5py is None:
        raise ImportError("h5py is required for HDF data. Install with: pip install h5py")
    paths = paths or HDF_PATHS
    with h5py.File(hdf_path, "r") as f:
        elev = np.asarray(f[paths["elevation"]][:], dtype=np.float32)
        area = np.asarray(f[paths["area"]][:], dtype=np.float32)
        if cell_index is not None:
            elev = elev[cell_index]
            area = area[cell_index]
    return elev, area


###############################################################################
# 2) RAW DATASET FROM HDF (HEC-RAS results) + HDF static (elevation, area) + text static (CS, CU, FA)
###############################################################################
###############################################################################
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
        # History: first n_history steps
        hist_all = [wd[: self.n_history], vx[: self.n_history], vy[: self.n_history]]
        dynamic_hist = np.stack([hist_all[i] for i in self.target_indices], axis=-1)
        dynamic_hist = torch.tensor(dynamic_hist, device="cpu", dtype=torch.float32)
        dynamic_hist = self._apply_noise(dynamic_hist)
        inflow_hist = inflow[: self.n_history]
        if inflow_hist.ndim == 1:
            inflow_hist = inflow_hist[:, None]
        flow_col = inflow_hist[:, -1:] if inflow_hist.shape[1] >= 2 else inflow_hist
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
            inflow_s = np.asarray(inflow[self.n_history + s], dtype=np.float32).flatten()
            flow_val = float(inflow_s[-1]) if inflow_s.size else 0.0
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
        if inflow.ndim == 1:
            inflow = inflow[:, None]
        flow_col = inflow[:, -1:] if inflow.shape[1] >= 2 else inflow  # (n_time, 1)
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
class FloodGINODataProcessor(DataProcessor):
    """
    Preprocesses samples for GINO and optionally inverse-transforms outputs for eval.
    Training: pred and y are both in normalized space (same as dataset target).
    Eval (when inverse_test=True): pred and y are inverse-transformed so metrics are in physical space.
    """
    def __init__(
        self,
        device="cuda",
        target_norm=None,
        inverse_test=True,
        output_distribution: str = "deterministic",
    ):
        super().__init__()
        self.device = device
        self.model = None
        self.target_norm = target_norm
        self.inverse_test = inverse_test
        self.output_distribution = str(output_distribution).strip().lower()
        if self.output_distribution not in {"deterministic", "gaussian"}:
            raise ValueError(
                "output_distribution must be one of {'deterministic', 'gaussian'}, "
                f"got {output_distribution!r}."
            )

    def preprocess(self, sample: dict) -> dict:
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                sample[k] = v.to(self.device)
        # Preserve optional keys (e.g. ada_in for FGN) — we only add/overwrite below

        # dynamic => (B, num_cells, n_history * n_target_channels)
        dyn_ = sample["dynamic"]
        if dyn_.dim() == 3:
            dyn_ = dyn_.unsqueeze(0)
        dyn_ = dyn_.permute(0, 2, 1, 3)
        B, N, H, D = dyn_.shape
        dyn_ = dyn_.reshape(B, N, H * D)

        # boundary => (B, num_cells, n_history * bc_dim)
        bc_ = sample["boundary"]
        if bc_.dim() == 3:
            bc_ = bc_.unsqueeze(0)
        bc_ = bc_.permute(0, 2, 1, 3)
        B2, N2, H2, C2 = bc_.shape
        bc_ = bc_.reshape(B2, N2, H2 * C2)

        # static => (B, num_cells, static_dim)
        st_ = sample["static"]
        if st_.dim() == 2:
            st_ = st_.unsqueeze(0)

        x_ = torch.cat([st_, bc_, dyn_], dim=2)

        geom_ = sample["geometry"]
        if geom_.dim() == 2:
            geom_ = geom_.unsqueeze(0)
        # GINO expects geometry with leading dim 1 (shared across batch). Use first sample when B > 1.
        if geom_.shape[0] > 1:
            geom_ = geom_[0:1]

        y_ = sample.get("target", None)
        if y_ is not None and y_.dim() == 2:
            y_ = y_.unsqueeze(0)

        q_ = sample["query_points"]
        if q_.dim() == 3:
            q_ = q_.unsqueeze(0)
        # GINO expects latent_queries / output_queries with leading dim 1 (shared). Use first when B > 1.
        if q_.shape[0] > 1:
            q_ = q_[0:1]

        sample["input_geom"] = geom_
        sample["latent_queries"] = q_
        sample["output_queries"] = geom_.clone()
        sample["x"] = x_
        sample["y"] = y_
        return sample

    @staticmethod
    def _match_stat_ndim(stat: torch.Tensor, val_ndim: int) -> torch.Tensor:
        out = stat
        while out.ndim > val_ndim and out.shape[0] == 1:
            out = out.squeeze(0)
        while out.ndim < val_ndim:
            out = out.unsqueeze(0)
        return out

    def postprocess(self, out: torch.Tensor, sample: dict):
        if (not self.training) and self.inverse_test and (self.target_norm is not None):
            if self.output_distribution == "gaussian":
                y_ref = sample.get("y")
                n_channels = y_ref.shape[-1] if y_ref is not None else (out.shape[-1] // 2)
                mu, logvar = split_gaussian_packed(out, n_channels=n_channels)
                mu = self.target_norm.inverse_transform(mu)

                std_stat = self._match_stat_ndim(self.target_norm.std, logvar.ndim)
                if std_stat.device != logvar.device:
                    std_stat = std_stat.to(logvar.device)
                eps = float(getattr(self.target_norm, "eps", 1e-7))
                logvar = logvar + 2.0 * torch.log(std_stat + eps)
                out = torch.cat([mu, logvar], dim=-1)
            else:
                out = self.target_norm.inverse_transform(out)
            if sample["y"] is not None:
                sample["y"] = self.target_norm.inverse_transform(sample["y"])
        return out, sample

    def to(self, device: str):
        self.device = device
        if self.target_norm is not None:
            self.target_norm.to(device)
        return self

    def wrap(self, model: torch.nn.Module):
        self.model = model

    def forward(self, sample: dict):
        sample = self.preprocess(sample)
        if self.model is None:
            raise RuntimeError("No model attached. Call wrap(model).")
        out = self.model(sample)
        out, sample = self.postprocess(out, sample)
        return out, sample


def get_flood_crps_weights(
    static: torch.Tensor,
    y: torch.Tensor,
    wet_threshold: float = 0.01,
    wet_smooth_scale: float = 0.02,
    dry_weight_alpha: float = 0.1,
    static_normalizer=None,
) -> torch.Tensor:
    """
    Compute per-(batch, cell, channel) weights for flood CRPS: area weighting plus
    soft wet/dry masking so the loss approximates a physical integral and avoids
    ill-defined velocities in dry cells.

    - Area: from static column index 1 (Cells Surface Area). If static_normalizer
      is provided, area is denormalized then used as weight_i = area_i / total_area
      (so weights sum to 1 per batch item). If no normalizer, raw static[:,:,1] is
      used and still normalized by total_area when possible.
    - Soft wetness m = sigmoid((depth - wet_threshold) / wet_smooth_scale) from
      ground-truth depth y[..., 0].
    - Depth weight: (area_ratio) * (alpha + (1 - alpha) * m).
    - Velocity weights (u, v): (area_ratio) * m.

    Parameters
    ----------
    static : torch.Tensor
        (B, n_cells, n_static) or (n_cells, n_static). Area must be at index 1
        (HDF "Cells Surface Area", same order as dataset: elevation, area, ...).
    y : torch.Tensor
        (B, n_cells, 3) targets [depth (h), vx (u), vy (v)].
    wet_threshold : float
        Depth threshold (m) below which cell is considered dry.
    wet_smooth_scale : float
        Smoothing scale (m) for sigmoid transition at wet/dry front.
    dry_weight_alpha : float
        Relative weight for dry-cell depth errors (e.g. 0.1 => dry counts 10x less than wet).
    static_normalizer : optional
        UnitGaussianNormalizer fit on static (dim [0,1]). If provided, area column
        (index 1) is inverse-transformed to physical units before area/total_area.

    Returns
    -------
    spatial_weights : torch.Tensor
        (B, n_cells, 3) with [w_depth, w_vel, w_vel], same device/dtype as y.
    """
    if static.dim() == 2:
        static = static.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    B, n_cells, _ = y.shape
    if static.shape[0] != B:
        static = static.expand(B, -1, -1)
    if static.shape[1] < n_cells:
        raise ValueError("get_flood_crps_weights: static n_cells < y n_cells")
    static = static[:, :n_cells, :]
    if static.shape[2] < 2:
        # No area column: fallback to uniform weight (1/n_cells per cell)
        area_ratio = torch.ones(B, n_cells, device=y.device, dtype=y.dtype) / n_cells
    else:
        area_raw = static[:, :, 1].clone().to(device=y.device, dtype=y.dtype)
        if static_normalizer is not None and hasattr(static_normalizer, "mean") and hasattr(static_normalizer, "std"):
            # Denormalize area column (index 1). Normalizer mean/std shape (1, 1, n_static)
            m = static_normalizer.mean.to(y.device)
            s = static_normalizer.std.to(y.device)
            if m.dim() >= 3 and m.shape[2] > 1:
                area_mean = m[0, 0, 1]
                area_std = s[0, 0, 1] + getattr(static_normalizer, "eps", 1e-7)
                area_phys = area_raw * area_std + area_mean
            else:
                area_phys = area_raw
            area_phys = torch.clamp(area_phys, min=0.0)
        else:
            area_phys = torch.clamp(area_raw, min=0.0)
        total_area = area_phys.sum(dim=1, keepdim=True).clamp(min=1e-12)
        area_ratio = area_phys / total_area
    depth = y[:, :, 0]
    m = torch.sigmoid((depth - wet_threshold) / max(wet_smooth_scale, 1e-8))
    w_depth = area_ratio * (dry_weight_alpha + (1.0 - dry_weight_alpha) * m)
    w_vel = area_ratio * m
    spatial_weights = torch.stack([w_depth, w_vel, w_vel], dim=-1)
    return spatial_weights


def _get_area_ratio_from_static(
    static: torch.Tensor,
    n_cells: int,
    device: torch.device,
    dtype: torch.dtype,
    static_normalizer=None,
) -> torch.Tensor:
    """
    Return area_ratio (B, n_cells) with sum(area_ratio, dim=1) = 1.
    Used by pooled functionals (e.g. hazard proxy). Static column index 1 = area.
    """
    if static.dim() == 2:
        static = static.unsqueeze(0)
    B = static.shape[0]
    if static.shape[1] < n_cells:
        raise ValueError("_get_area_ratio_from_static: static n_cells < n_cells")
    static = static[:, :n_cells, :]
    if static.shape[2] < 2:
        return torch.ones(B, n_cells, device=device, dtype=dtype) / n_cells
    area_raw = static[:, :, 1].clone().to(device=device, dtype=dtype)
    if static_normalizer is not None and hasattr(static_normalizer, "mean") and hasattr(static_normalizer, "std"):
        m = static_normalizer.mean.to(device)
        s = static_normalizer.std.to(device)
        if m.dim() >= 3 and m.shape[2] > 1:
            area_mean = m[0, 0, 1]
            area_std = s[0, 0, 1] + getattr(static_normalizer, "eps", 1e-7)
            area_phys = area_raw * area_std + area_mean
        else:
            area_phys = area_raw
        area_phys = torch.clamp(area_phys, min=0.0)
    else:
        area_phys = torch.clamp(area_raw, min=0.0)
    total_area = area_phys.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return area_phys / total_area


def compute_hazard_proxy_pooled(
    static: torch.Tensor,
    field: torch.Tensor,
    wet_threshold: float = 0.01,
    wet_smooth_scale: float = 0.02,
    static_normalizer=None,
) -> torch.Tensor:
    """
    Velocity–depth hazard proxy pooled over the mesh (scalar per batch, or per ensemble and batch).

    H = sum_i (area_ratio_i) * m_i * h_i * (u_i^2 + v_i^2),
    with m_i = sigmoid((h_i - tau) / eps) (soft wet mask). Couples (h, u, v) and penalizes
    nonphysical fast water in near-dry cells. Differentiable.

    Parameters
    ----------
    static : torch.Tensor
        (B, n_cells, n_static) or (n_cells, n_static). Area at column index 1.
    field : torch.Tensor
        (B, n_cells, 3) or (N, B, n_cells, 3) with channels [depth (h), vx (u), vy (v)].
    wet_threshold : float
        Depth threshold (m) for soft wet mask.
    wet_smooth_scale : float
        Sigmoid smoothing scale (m).
    static_normalizer : optional
        UnitGaussianNormalizer for static; if provided, area is denormalized before area_ratio.

    Returns
    -------
    pooled : torch.Tensor
        (B,) or (N, B) scalar hazard proxy per batch item (and per ensemble member if field is 4D).
    """
    has_ensemble = field.dim() == 4
    if has_ensemble:
        N, B, n_cells, _ = field.shape
        area_ratio = _get_area_ratio_from_static(
            static, n_cells, field.device, field.dtype, static_normalizer
        )
        h = field[:, :, :, 0]
        u = field[:, :, :, 1]
        v = field[:, :, :, 2]
        m = torch.sigmoid((h - wet_threshold) / max(wet_smooth_scale, 1e-8))
        kinetic = h * (u * u + v * v)
        pooled = (area_ratio.unsqueeze(0) * m * kinetic).sum(dim=2)
        return pooled
    else:
        if field.dim() == 2:
            field = field.unsqueeze(0)
        B, n_cells, _ = field.shape
        area_ratio = _get_area_ratio_from_static(
            static, n_cells, field.device, field.dtype, static_normalizer
        )
        h = field[:, :, 0]
        u = field[:, :, 1]
        v = field[:, :, 2]
        m = torch.sigmoid((h - wet_threshold) / max(wet_smooth_scale, 1e-8))
        kinetic = h * (u * u + v * v)
        pooled = (area_ratio * m * kinetic).sum(dim=1)
        return pooled


def _build_x_from_dynamic_boundary(static: torch.Tensor, boundary: torch.Tensor, dynamic: torch.Tensor):
    """
    Build GINO input x from (static, boundary, dynamic) in the same format as FloodGINODataProcessor.
    static (B, n_cells, s), boundary (B, n_history, n_cells, bc), dynamic (B, n_history, n_cells, 3).
    Returns x (B, n_cells, s + n_history*bc + n_history*3).
    """
    if dynamic.dim() == 3:
        dynamic = dynamic.unsqueeze(0)
    if boundary.dim() == 3:
        boundary = boundary.unsqueeze(0)
    if static.dim() == 2:
        static = static.unsqueeze(0)
    dyn_ = dynamic.permute(0, 2, 1, 3).reshape(dynamic.shape[0], dynamic.shape[2], -1)
    bc_ = boundary.permute(0, 2, 1, 3).reshape(boundary.shape[0], boundary.shape[2], -1)
    x_ = torch.cat([static, bc_, dyn_], dim=2)
    return x_


def _gaussian_mean_from_packed(out: torch.Tensor, n_channels: int) -> torch.Tensor:
    """Extract Gaussian predictive mean from packed [mu, logvar] output."""
    mu, _ = split_gaussian_packed(out, n_channels=n_channels)
    return mu


def _sample_from_packed_gaussian(
    out: torch.Tensor,
    n_channels: int,
    min_logvar: float = -9.0,
    max_logvar: float = 4.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reparameterized sample from packed [mu, logvar] output.

    Returns (sample, mu, logvar_clamped).
    """
    mu, logvar = split_gaussian_packed(out, n_channels=n_channels)
    logvar = torch.clamp(logvar, min=float(min_logvar), max=float(max_logvar))
    std = torch.exp(0.5 * logvar)
    sample = mu + std * torch.randn_like(mu)
    return sample, mu, logvar


###############################################################################
# 6b) FGN Trainer (two forwards + CRPS per batch)
###############################################################################
class FGNTrainer(Trainer):
    """
    Trainer for FGN (Functional Generative Networks): two forward passes per batch
    with different noise z, then CRPS loss on (out1, out2, y).

    Same data sample (x, y) is used for both passes; only the noise z differs.
    Returns (loss, {'rel_l2': rel_l2}). rel_l2 should be sum-over-batch so train_err = sum(rel_l2)/n_samples
    (mean per sample, same scale as test_l2). Use LpLoss with reduction='sum' for rel_l2_loss_fn.

    Supports autoregressive (AR) fine-tuning: when epoch >= ar_finetune_start_epoch and
    ar_rollout_steps > 1, runs an AR rollout, computes loss at each step, averages over
    steps, and backpropagates through the rollout (FGN-style). Optional curriculum: set
    ar_curriculum_epochs_per_step > 0 to ramp rollout length (1 step for E epochs, then
    2 steps for E epochs, ... up to ar_rollout_steps).
    """

    def __init__(
        self,
        fgn_noise_dim=32,
        crps_n_samples=2,
        rel_l2_loss_fn=None,
        crps_l2_weight=0.0,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=1,
        ar_curriculum_epochs_per_step=0,
        use_flood_crps_spatial_weights=False,
        flood_crps_wet_threshold=0.01,
        flood_crps_wet_smooth_scale=0.02,
        flood_crps_dry_weight_alpha=0.1,
        static_normalizer=None,
        use_hazard_proxy_crps=False,
        hazard_proxy_crps_weight=0.15,
        ar_pooled_crps_gamma=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fgn_noise_dim = fgn_noise_dim
        self.crps_n_samples = max(2, int(crps_n_samples))
        self.rel_l2_loss_fn = rel_l2_loss_fn
        self.crps_l2_weight = float(crps_l2_weight)
        self.ar_finetune_start_epoch = max(0, int(ar_finetune_start_epoch))
        self.ar_rollout_steps = max(1, int(ar_rollout_steps))
        self.ar_curriculum_epochs_per_step = max(0, int(ar_curriculum_epochs_per_step))
        self.use_flood_crps_spatial_weights = bool(use_flood_crps_spatial_weights)
        self.flood_crps_wet_threshold = float(flood_crps_wet_threshold)
        self.flood_crps_wet_smooth_scale = float(flood_crps_wet_smooth_scale)
        self.flood_crps_dry_weight_alpha = float(flood_crps_dry_weight_alpha)
        self.static_normalizer = static_normalizer
        self.use_hazard_proxy_crps = bool(use_hazard_proxy_crps)
        self.hazard_proxy_crps_weight = float(hazard_proxy_crps_weight)
        self.ar_pooled_crps_gamma = float(ar_pooled_crps_gamma)

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        """Single-step FGN: n_crps forward passes with different z, loss on (pred_samples, y)."""
        n_crps = self.crps_n_samples
        outs = []
        batch_size = sample["x"].shape[0]
        for _ in range(n_crps):
            z = torch.randn(
                batch_size, self.fgn_noise_dim, device=self.device, dtype=sample["x"].dtype
            )
            samp = {**sample, "ada_in": z}
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    out = self.model(**samp)
            else:
                out = self.model(**samp)
            if self.data_processor is not None:
                out, sample = self.data_processor.postprocess(out, sample)
            outs.append(out)
        pred_samples = torch.stack(outs, dim=0)
        pred_mean = pred_samples.mean(dim=0)
        y_target = sample["y"]
        if self.use_flood_crps_spatial_weights and "static" in sample and y_target.shape[-1] >= 3:
            spatial_weights = get_flood_crps_weights(
                sample["static"],
                y_target,
                wet_threshold=self.flood_crps_wet_threshold,
                wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                dry_weight_alpha=self.flood_crps_dry_weight_alpha,
                static_normalizer=self.static_normalizer,
            )
            loss = training_loss(pred_samples, y_target, spatial_weights=spatial_weights)
        else:
            loss = training_loss(pred_samples, y_target)
        if self.crps_l2_weight > 0 and self.rel_l2_loss_fn is not None:
            loss = loss + self.crps_l2_weight * self.rel_l2_loss_fn(pred_mean, sample["y"])
        if self.use_hazard_proxy_crps and "static" in sample and y_target.shape[-1] >= 3:
            pred_pooled = compute_hazard_proxy_pooled(
                sample["static"],
                pred_samples,
                wet_threshold=self.flood_crps_wet_threshold,
                wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                static_normalizer=self.static_normalizer,
            )
            y_pooled = compute_hazard_proxy_pooled(
                sample["static"],
                y_target,
                wet_threshold=self.flood_crps_wet_threshold,
                wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                static_normalizer=self.static_normalizer,
            )
            crps_pooled = fair_crps_univariate(pred_pooled, y_pooled).mean()
            loss = loss + self.hazard_proxy_crps_weight * crps_pooled
        metrics = {}
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                metrics["rel_l2"] = self.rel_l2_loss_fn(pred_mean, sample["y"])
        return loss, metrics

    def train_one_batch(self, idx, sample, training_loss):
        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        use_ar = (
            self.ar_rollout_steps > 1
            and self.epoch >= self.ar_finetune_start_epoch
            and "target_sequence" in sample
            and sample["target_sequence"] is not None
        )
        if use_ar:
            target_sequence = sample["target_sequence"]
            boundary_sequence = sample["boundary_sequence"]
            # Normalize to (B, T, ...) only if collate produced 5D (B, 1, T, n_cells, C)
            if target_sequence.dim() == 5:
                target_sequence = target_sequence.squeeze(1)
            if boundary_sequence.dim() == 5:
                boundary_sequence = boundary_sequence.squeeze(1)
            max_available_steps = target_sequence.shape[1]
            if self.ar_curriculum_epochs_per_step > 0:
                ar_epoch_index = self.epoch - self.ar_finetune_start_epoch
                curriculum_step_index = ar_epoch_index // self.ar_curriculum_epochs_per_step
                effective_ar_steps = min(curriculum_step_index + 1, self.ar_rollout_steps)
            else:
                effective_ar_steps = self.ar_rollout_steps
            n_ar_steps = min(effective_ar_steps, max_available_steps)
            self.n_samples += sample["y"].shape[0] * n_ar_steps

            n_history = sample["dynamic"].shape[1]
            dynamic_sliding = sample["dynamic"].clone()
            boundary_sliding = sample["boundary"].clone()
            static = sample["static"]
            geom = sample["input_geom"]
            q = sample["latent_queries"]
            out_q = sample["output_queries"]

            total_loss = 0.0
            last_rel_l2 = None
            n_crps = self.crps_n_samples
            for s in range(n_ar_steps):
                x = _build_x_from_dynamic_boundary(static, boundary_sliding, dynamic_sliding)
                y_s = target_sequence[:, s]
                if y_s.dim() == 2:
                    y_s = y_s.unsqueeze(0)
                kwargs_base = {"input_geom": geom, "latent_queries": q, "output_queries": out_q, "x": x}
                outs_s = []
                for _ in range(n_crps):
                    z = torch.randn(
                        x.shape[0], self.fgn_noise_dim, device=self.device, dtype=x.dtype
                    )
                    if self.mixed_precision:
                        with torch.autocast(device_type=self.autocast_device_type):
                            out = self.model(**kwargs_base, ada_in=z)
                    else:
                        out = self.model(**kwargs_base, ada_in=z)
                    if self.data_processor is not None:
                        out, _ = self.data_processor.postprocess(out, {**sample, "y": y_s})
                    outs_s.append(out)
                pred_samples = torch.stack(outs_s, dim=0)
                pred_mean = pred_samples.mean(dim=0)
                if self.use_flood_crps_spatial_weights and "static" in sample and y_s.shape[-1] >= 3:
                    spatial_weights_s = get_flood_crps_weights(
                        static,
                        y_s,
                        wet_threshold=self.flood_crps_wet_threshold,
                        wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                        dry_weight_alpha=self.flood_crps_dry_weight_alpha,
                        static_normalizer=self.static_normalizer,
                    )
                    loss_s = training_loss(pred_samples, y_s, spatial_weights=spatial_weights_s)
                else:
                    loss_s = training_loss(pred_samples, y_s)
                if self.crps_l2_weight > 0 and self.rel_l2_loss_fn is not None:
                    loss_s = loss_s + self.crps_l2_weight * self.rel_l2_loss_fn(pred_mean, y_s)
                if self.use_hazard_proxy_crps and y_s.shape[-1] >= 3:
                    pred_pooled_s = compute_hazard_proxy_pooled(
                        static,
                        pred_samples,
                        wet_threshold=self.flood_crps_wet_threshold,
                        wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                        static_normalizer=self.static_normalizer,
                    )
                    y_pooled_s = compute_hazard_proxy_pooled(
                        static,
                        y_s,
                        wet_threshold=self.flood_crps_wet_threshold,
                        wet_smooth_scale=self.flood_crps_wet_smooth_scale,
                        static_normalizer=self.static_normalizer,
                    )
                    gamma_s = self.ar_pooled_crps_gamma ** s
                    loss_s = loss_s + gamma_s * self.hazard_proxy_crps_weight * fair_crps_univariate(pred_pooled_s, y_pooled_s).mean()
                total_loss = total_loss + loss_s
                if self.rel_l2_loss_fn is not None:
                    with torch.no_grad():
                        last_rel_l2 = self.rel_l2_loss_fn(pred_mean, y_s)
                dynamic_sliding = torch.cat([dynamic_sliding[:, 1:], pred_mean.unsqueeze(1)], dim=1)
                dynamic_sliding = dynamic_sliding[:, -n_history:]
                if boundary_sequence.dim() == 5:
                    time_dim = next(i for i, sz in enumerate(boundary_sequence.shape) if sz == max_available_steps)
                    sl = [slice(None)] * boundary_sequence.dim()
                    sl[time_dim] = slice(s, s + 1)
                    bc_step = boundary_sequence[tuple(sl)].squeeze(time_dim)
                else:
                    bc_step = boundary_sequence[:, s : s + 1]
                if bc_step.dim() == 3:
                    bc_step = bc_step.unsqueeze(1)
                boundary_sliding = torch.cat([boundary_sliding[:, 1:], bc_step], dim=1)[:, -n_history:]
            loss = total_loss / n_ar_steps
            metrics = {"rel_l2": last_rel_l2 if last_rel_l2 is not None else torch.tensor(0.0, device=self.device)}
            if idx == 0 and self.logger is not None:
                self.logger.info(
                    "AR fine-tuning: epoch=%s, rollout_steps=%s (max=%s)%s, loss averaged over steps, backprop through rollout.",
                    self.epoch,
                    n_ar_steps,
                    self.ar_rollout_steps,
                    " [curriculum]" if self.ar_curriculum_epochs_per_step > 0 else "",
                )
        else:
            self.n_samples += sample["y"].shape[0]
            loss, metrics = self._train_one_batch_single_step(idx, sample, training_loss)

        if self.epoch == 0 and idx == 0 and self.verbose:
            B = sample["y"].shape[0]
            print(f"FGN {'AR' if use_ar else 'single-step'}: loss = {loss.item():.8f} (B={B})")

        if self.regularizer:
            loss = loss + self.regularizer.loss
        return loss, metrics

    def eval_one_batch(self, sample, eval_losses, return_output=False):
        """FGN eval: crps_n_samples forward passes with different z; L2 on ensemble mean, CRPS on pred_samples."""
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].size(0)
        n_crps = self.crps_n_samples
        outs = []
        y_eval = sample["y"]
        batch_size = sample["x"].shape[0]
        for _ in range(n_crps):
            z = torch.randn(
                batch_size, self.fgn_noise_dim, device=self.device, dtype=sample["x"].dtype
            )
            samp = {**sample, "ada_in": z}
            out = self.model(**samp)
            if self.data_processor is not None:
                # Important: avoid repeatedly inverse-transforming the same y across ensemble
                # members when inverse_test=True. Use an isolated sample dict per pass.
                sample_for_post = {"y": sample["y"]}
                out, sample_post = self.data_processor.postprocess(out, sample_for_post)
                y_eval = sample_post["y"]
            outs.append(out)
        pred_samples = torch.stack(outs, dim=0)
        pred_mean = pred_samples.mean(dim=0)

        eval_step_losses = {}
        for loss_name, loss_fn in eval_losses.items():
            if loss_name == "crps":
                val = loss_fn(pred_samples, y_eval)
            else:
                val = loss_fn(pred_mean, y_eval)
            eval_step_losses[loss_name] = val

        out = pred_mean if return_output else None
        return eval_step_losses, out


class GaussianNLLTrainer(Trainer):
    """
    Trainer for heteroscedastic Gaussian outputs packed as [mu, logvar].

    Supports single-step and autoregressive (AR) training.
    AR updates are sampled via reparameterization to preserve trajectory diversity.
    """

    def __init__(
        self,
        rel_l2_loss_fn=None,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=1,
        ar_curriculum_epochs_per_step=0,
        gaussian_min_logvar=-9.0,
        gaussian_max_logvar=4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rel_l2_loss_fn = rel_l2_loss_fn
        self.ar_finetune_start_epoch = max(0, int(ar_finetune_start_epoch))
        self.ar_rollout_steps = max(1, int(ar_rollout_steps))
        self.ar_curriculum_epochs_per_step = max(0, int(ar_curriculum_epochs_per_step))
        self.gaussian_min_logvar = float(gaussian_min_logvar)
        self.gaussian_max_logvar = float(gaussian_max_logvar)

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = training_loss(out, sample["y"])
        metrics = {}
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                pred_mean = _gaussian_mean_from_packed(out, sample["y"].shape[-1])
                metrics["rel_l2"] = self.rel_l2_loss_fn(pred_mean, sample["y"])
        if self.epoch == 0 and idx == 0 and self.verbose:
            B = sample["y"].shape[0]
            print(f"Gaussian NLL single-step: loss = {loss.item():.8f} (B={B})")
        return loss, metrics

    def train_one_batch(self, idx, sample, training_loss):
        self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        use_ar = (
            self.ar_rollout_steps > 1
            and self.epoch >= self.ar_finetune_start_epoch
            and "target_sequence" in sample
            and sample["target_sequence"] is not None
        )
        if use_ar:
            target_sequence = sample["target_sequence"]
            boundary_sequence = sample["boundary_sequence"]
            if target_sequence.dim() == 5:
                target_sequence = target_sequence.squeeze(1)
            if boundary_sequence.dim() == 5:
                boundary_sequence = boundary_sequence.squeeze(1)
            max_available_steps = target_sequence.shape[1]
            if self.ar_curriculum_epochs_per_step > 0:
                ar_epoch_index = self.epoch - self.ar_finetune_start_epoch
                curriculum_step_index = ar_epoch_index // self.ar_curriculum_epochs_per_step
                effective_ar_steps = min(curriculum_step_index + 1, self.ar_rollout_steps)
            else:
                effective_ar_steps = self.ar_rollout_steps
            n_ar_steps = min(effective_ar_steps, max_available_steps)
            self.n_samples += sample["y"].shape[0] * n_ar_steps

            n_history = sample["dynamic"].shape[1]
            dynamic_sliding = sample["dynamic"].clone()
            boundary_sliding = sample["boundary"].clone()
            static = sample["static"]
            geom = sample["input_geom"]
            q = sample["latent_queries"]
            out_q = sample["output_queries"]

            total_loss = 0.0
            last_rel_l2 = None
            for s in range(n_ar_steps):
                x = _build_x_from_dynamic_boundary(static, boundary_sliding, dynamic_sliding)
                y_s = target_sequence[:, s]
                if y_s.dim() == 2:
                    y_s = y_s.unsqueeze(0)
                kwargs_base = {
                    "input_geom": geom,
                    "latent_queries": q,
                    "output_queries": out_q,
                    "x": x,
                }
                if self.mixed_precision:
                    with torch.autocast(device_type=self.autocast_device_type):
                        out = self.model(**kwargs_base)
                else:
                    out = self.model(**kwargs_base)
                if self.data_processor is not None:
                    out, _ = self.data_processor.postprocess(out, {**sample, "y": y_s})

                loss_s = training_loss(out, y_s)
                total_loss = total_loss + loss_s

                sampled_next, pred_mean, _ = _sample_from_packed_gaussian(
                    out,
                    n_channels=y_s.shape[-1],
                    min_logvar=self.gaussian_min_logvar,
                    max_logvar=self.gaussian_max_logvar,
                )
                if self.rel_l2_loss_fn is not None:
                    with torch.no_grad():
                        last_rel_l2 = self.rel_l2_loss_fn(pred_mean, y_s)

                dynamic_sliding = torch.cat(
                    [dynamic_sliding[:, 1:], sampled_next.unsqueeze(1)], dim=1
                )
                dynamic_sliding = dynamic_sliding[:, -n_history:]
                if boundary_sequence.dim() == 5:
                    time_dim = next(i for i, sz in enumerate(boundary_sequence.shape) if sz == max_available_steps)
                    sl = [slice(None)] * boundary_sequence.dim()
                    sl[time_dim] = slice(s, s + 1)
                    bc_step = boundary_sequence[tuple(sl)].squeeze(time_dim)
                else:
                    bc_step = boundary_sequence[:, s : s + 1]
                if bc_step.dim() == 3:
                    bc_step = bc_step.unsqueeze(1)
                boundary_sliding = torch.cat([boundary_sliding[:, 1:], bc_step], dim=1)[:, -n_history:]

            loss = total_loss / n_ar_steps
            metrics = {"rel_l2": last_rel_l2 if last_rel_l2 is not None else torch.tensor(0.0, device=self.device)}
            if idx == 0 and self.logger is not None:
                self.logger.info(
                    "Gaussian AR fine-tuning: epoch=%s, rollout_steps=%s (max=%s)%s.",
                    self.epoch,
                    n_ar_steps,
                    self.ar_rollout_steps,
                    " [curriculum]" if self.ar_curriculum_epochs_per_step > 0 else "",
                )
        else:
            self.n_samples += sample["y"].shape[0]
            loss, metrics = self._train_one_batch_single_step(idx, sample, training_loss)

        if self.regularizer:
            loss = loss + self.regularizer.loss
        return loss, metrics

    def eval_one_batch(self, sample, eval_losses, return_output=False):
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            sample = {k: v.to(self.device) for k, v in sample.items() if torch.is_tensor(v)}

        self.n_samples += sample["y"].size(0)
        out = self.model(**sample)
        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        pred_mean = _gaussian_mean_from_packed(out, sample["y"].shape[-1])

        eval_step_losses = {}
        for loss_name, loss_fn in eval_losses.items():
            if loss_name == "l2":
                val = loss_fn(pred_mean, sample["y"])
            else:
                val = loss_fn(out, sample["y"])
            eval_step_losses[loss_name] = val

        return eval_step_losses, (pred_mean if return_output else None)


###############################################################################
# 7) NormalizedRolloutTestDataset
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


##############################################################################
# 8) ANIMATION & PUBLICATION PLOTTING HELPERS
###############################################################################
def create_rollout_animation(
        geometry,
        wd_gt, wd_pred,
        vx_gt, vy_gt,
        vx_pred, vy_pred,
        run_id=None,
        out_dir=".",
        filename_prefix="rollout",
        dt_seconds: float = 1200.0
):
    """
    Creates an animation comparing Ground Truth and Predictions in a 3x2 grid.
    - Row 1: Ground Truth Depth vs. Predicted Depth
    - Row 2: Ground Truth VX vs. Predicted VX
    - Row 3: Ground Truth VY vs. Predicted VY
    """
    # Convert inputs to numpy arrays
    if isinstance(geometry, torch.Tensor):
        geometry = geometry.cpu().numpy()
    x_coords, y_coords = geometry[:, 0], geometry[:, 1]

    wd_gt, wd_pred = np.asarray(wd_gt), np.asarray(wd_pred)
    vx_gt, vy_gt = np.asarray(vx_gt), np.asarray(vy_gt)
    vx_pred, vy_pred = np.asarray(vx_pred), np.asarray(vy_pred)
    rollout_length = wd_gt.shape[0]

    # Prepare figure with a 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), constrained_layout=True)
    fig.suptitle(f"Rollout Comparison (Run: {run_id or 'unknown'})", fontsize=20)
    (ax_gt_wd, ax_pred_wd), (ax_gt_vx, ax_pred_vx), (ax_gt_vy, ax_pred_vy) = axes

    # --- Set Color Limits ---
    depth_max = max(np.nanmax(wd_gt), np.nanmax(wd_pred))
    # For velocities, find the max absolute value for a symmetric color scale
    vx_abs_max = np.max([np.abs(vx_gt), np.abs(vx_pred)])
    vy_abs_max = np.max([np.abs(vy_gt), np.abs(vy_pred)])

    # --- Row 1: Water Depth ---
    sc_gt_wd = ax_gt_wd.scatter(x_coords, y_coords, c=wd_gt[0], vmin=0, vmax=depth_max, s=15, cmap='viridis')
    ax_gt_wd.set_title("Ground Truth Depth", pad=10)
    ax_gt_wd.axis('off')
    fig.colorbar(sc_gt_wd, ax=ax_gt_wd, fraction=0.046, pad=0.04).set_label("Depth (m)")

    sc_pred_wd = ax_pred_wd.scatter(x_coords, y_coords, c=wd_pred[0], vmin=0, vmax=depth_max, s=15, cmap='viridis')
    ax_pred_wd.set_title("Predicted Depth", pad=10)
    ax_pred_wd.axis('off')
    fig.colorbar(sc_pred_wd, ax=ax_pred_wd, fraction=0.046, pad=0.04).set_label("Depth (m)")

    # --- Row 2: X-Velocity (VX) ---
    sc_gt_vx = ax_gt_vx.scatter(x_coords, y_coords, c=vx_gt[0], vmin=-vx_abs_max, vmax=vx_abs_max, s=15,
                                cmap='coolwarm')
    ax_gt_vx.set_title("Ground Truth VX", pad=10)
    ax_gt_vx.axis('off')
    fig.colorbar(sc_gt_vx, ax=ax_gt_vx, fraction=0.046, pad=0.04).set_label("VX (m/s)")

    sc_pred_vx = ax_pred_vx.scatter(x_coords, y_coords, c=vx_pred[0], vmin=-vx_abs_max, vmax=vx_abs_max, s=15,
                                    cmap='coolwarm')
    ax_pred_vx.set_title("Predicted VX", pad=10)
    ax_pred_vx.axis('off')
    fig.colorbar(sc_pred_vx, ax=ax_pred_vx, fraction=0.046, pad=0.04).set_label("VX (m/s)")

    # --- Row 3: Y-Velocity (VY) ---
    sc_gt_vy = ax_gt_vy.scatter(x_coords, y_coords, c=vy_gt[0], vmin=-vy_abs_max, vmax=vy_abs_max, s=15,
                                cmap='coolwarm')
    ax_gt_vy.set_title("Ground Truth VY", pad=10)
    ax_gt_vy.axis('off')
    fig.colorbar(sc_gt_vy, ax=ax_gt_vy, fraction=0.046, pad=0.04).set_label("VY (m/s)")

    sc_pred_vy = ax_pred_vy.scatter(x_coords, y_coords, c=vy_pred[0], vmin=-vy_abs_max, vmax=vy_abs_max, s=15,
                                    cmap='coolwarm')
    ax_pred_vy.set_title("Predicted VY", pad=10)
    ax_pred_vy.axis('off')
    fig.colorbar(sc_pred_vy, ax=ax_pred_vy, fraction=0.046, pad=0.04).set_label("VY (m/s)")

    # Animation update function
    def animate(frame_idx):
        time_hours = (frame_idx + 1) * dt_seconds / 3600.0
        fig.suptitle(f"Rollout Comparison (Run: {run_id or 'unknown'}) - Time: {time_hours:.2f} hrs", fontsize=20)
        sc_gt_wd.set_array(wd_gt[frame_idx])
        sc_pred_wd.set_array(wd_pred[frame_idx])
        sc_gt_vx.set_array(vx_gt[frame_idx])
        sc_pred_vx.set_array(vx_pred[frame_idx])
        sc_gt_vy.set_array(vy_gt[frame_idx])
        sc_pred_vy.set_array(vy_pred[frame_idx])
        return sc_gt_wd, sc_pred_wd, sc_gt_vx, sc_pred_vx, sc_gt_vy, sc_pred_vy

    ani = animation.FuncAnimation(fig, animate, frames=rollout_length, interval=200, blit=False)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{filename_prefix}_{run_id or 'unknown'}.gif")
    ani.save(out_path, writer='pillow', fps=5)
    plt.close(fig)
    print(f"Saved rollout animation to: {out_path}")


def generate_publication_maps(
        geometry,
        wd_gt_array: np.ndarray, wd_pred_array: np.ndarray,
        vx_gt_array: np.ndarray, vy_gt_array: np.ndarray,
        vx_pred_array: np.ndarray, vy_pred_array: np.ndarray,
        steps,
        out_dir: str = ".",
        run_id: str = None,
        filename_prefix: str = "step"
):
    """
    Generates high-quality 3x3 comparison maps for specific timesteps.
    - Row 1: Ground Truth Depth, Predicted Depth, Absolute Depth Error.
    - Row 2: Ground Truth VX, Predicted VX, Absolute VX Error.
    - Row 3: Ground Truth VY, Predicted VY, Absolute VY Error.
    """
    if isinstance(steps, int):
        steps = [steps]
    geo_np = geometry.cpu().numpy() if hasattr(geometry, "cpu") else np.asarray(geometry)
    x, y = geo_np[:, 0], geo_np[:, 1]
    rid = run_id or "unknown"
    os.makedirs(out_dir, exist_ok=True)
    plt.rc("font", family="serif", size=12)

    for t in steps:
        if t < 0 or t >= wd_gt_array.shape[0]:
            print(f"  Skipping invalid step {t}")
            continue

        # Extract data for the specific timestep
        wd_gt, wd_pred = wd_gt_array[t], wd_pred_array[t]
        vx_gt, vy_gt = vx_gt_array[t], vy_gt_array[t]
        vx_pred, vy_pred = vx_pred_array[t], vy_pred_array[t]

        # Calculate errors
        err_wd = np.abs(wd_pred - wd_gt)
        err_vx = np.abs(vx_pred - vx_gt)
        err_vy = np.abs(vy_pred - vy_gt)

        # Determine robust color limits
        dmax = max(np.nanmax(wd_gt), np.nanmax(wd_pred))
        emax_wd = np.nanmax(err_wd)
        vx_abs_max = np.max([np.abs(vx_gt), np.abs(vx_pred)])
        vy_abs_max = np.max([np.abs(vy_gt), np.abs(vy_pred)])
        emax_vx = np.nanmax(err_vx)
        emax_vy = np.nanmax(err_vy)

        fig, axs = plt.subplots(3, 3, figsize=(18, 17), dpi=300, constrained_layout=True)
        panels = [
            ("(a) Ground Truth Depth", wd_gt, "viridis", 0.0, dmax, "Depth (m)"),
            ("(b) Predicted Depth", wd_pred, "viridis", 0.0, dmax, "Depth (m)"),
            ("(c) Depth Abs. Error", err_wd, "magma", 0.0, emax_wd, "Error (m)"),
            ("(d) Ground Truth VX", vx_gt, "coolwarm", -vx_abs_max, vx_abs_max, "VX (m/s)"),
            ("(e) Predicted VX", vx_pred, "coolwarm", -vx_abs_max, vx_abs_max, "VX (m/s)"),
            ("(f) VX Abs. Error", err_vx, "magma", 0.0, emax_vx, "Error (m/s)"),
            ("(g) Ground Truth VY", vy_gt, "coolwarm", -vy_abs_max, vy_abs_max, "VY (m/s)"),
            ("(h) Predicted VY", vy_pred, "coolwarm", -vy_abs_max, vy_abs_max, "VY (m/s)"),
            ("(i) VY Abs. Error", err_vy, "magma", 0.0, emax_vy, "Error (m/s)"),
        ]

        for ax, (title, data, cmap, vmin, vmax, cblabel) in zip(axs.flatten(), panels):
            sc = ax.scatter(x, y, c=data, cmap=cmap, vmin=vmin, vmax=vmax, s=6, marker="s", linewidths=0,
                            rasterized=True)
            ax.set_title(title, pad=8, fontsize=14)
            ax.set_aspect("equal")
            ax.axis("off")
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(cblabel, labelpad=10, fontsize=12)
            cbar.ax.tick_params(labelsize=10)

        fname = f"{filename_prefix}_{rid}_t{t}.png"
        out_path = os.path.join(out_dir, fname)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"  Saved publication map for t={t} -> {out_path}")


###############################################################################
# 8b) Training verification (gradient flow + overfit sanity check)
###############################################################################
def verify_training_gradient_flow(trainer, train_loader, training_loss):
    """
    Verify that loss is differentiable and gradients flow to model parameters.
    Runs one forward + backward and checks loss.grad_fn and param.grad norms.
    """
    trainer.model.train()
    if trainer.data_processor is not None:
        trainer.data_processor.train()
    batch = next(iter(train_loader))
    result = trainer.train_one_batch(0, batch, training_loss)
    loss, _ = (result[0], result[1]) if isinstance(result, tuple) and len(result) == 2 else (result, {})

    if not isinstance(loss, torch.Tensor):
        raise AssertionError(f"train_one_batch must return a Tensor (or (loss, metrics)), got {type(loss)}")
    if not loss.requires_grad:
        raise AssertionError("Loss does not require grad; check that model output is used in loss.")
    if loss.grad_fn is None:
        raise AssertionError("Loss has no grad_fn; graph may be detached.")

    loss.backward()

    total_norm = 0.0
    num_params_with_grad = 0
    for p in trainer.model.parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params_with_grad += 1
    total_norm = total_norm ** 0.5

    if num_params_with_grad == 0:
        raise AssertionError("No parameter received gradients; optimizer will not update the model.")
    if total_norm == 0.0:
        raise AssertionError("Total gradient norm is zero; loss may not depend on model parameters.")

    print(f"[Verify] Gradient flow OK: loss={loss.item():.6f}, grad_norm={total_norm:.6e}, params_with_grad={num_params_with_grad}")
    return True


def overfit_sanity_check(trainer, train_loader, training_loss, optimizer, n_steps=15):
    """
    Overfit a single batch for n_steps. If optimization is correct, loss should decrease.
    """
    trainer.model.train()
    if trainer.data_processor is not None:
        trainer.data_processor.train()
    batch = next(iter(train_loader))
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        result = trainer.train_one_batch(0, batch, training_loss)
        loss = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if losses[-1] >= losses[0]:
        warnings.warn(
            f"[Overfit check] Loss did not decrease over {n_steps} steps "
            f"(start={losses[0]:.6f}, end={losses[-1]:.6f}). "
            "Check learning rate, loss scale, or data/model.",
            UserWarning,
            stacklevel=2,
        )
    else:
        print(f"[Overfit check] OK: loss decreased from {losses[0]:.6f} to {losses[-1]:.6f} over {n_steps} steps.")
    return losses


###############################################################################
# 9) ROLLOUT PREDICTION (with additional metrics computation and saving)
###############################################################################
def rollout_prediction(
        trainer,
        rollout_dataset,
        rollout_length,
        history_steps,
        dynamic_norm,
        target_norm,
        device,
        skip_before_timestep,
        dt,
        out_dir="./rollout_gifs",
        fgn_noise_dim=None,
        n_ensemble_samples=1,
        gaussian_min_logvar: float = -9.0,
        gaussian_max_logvar: float = 4.0,
):
    """
    Performs autoregressive rollout, computing and plotting metrics for water depth, VX, and VY.
    FGN mode: when fgn_noise_dim is set and n_ensemble_samples > 1, runs an ensemble of
    forwards per step with per-sample noise z shaped [B, D] and uses the mean prediction.
    Gaussian mode: samples members from packed [mu, logvar] outputs and propagates sampled
    member states autoregressively.
    """
    model = trainer.model
    model.eval()
    dynamic_norm.to(device)
    target_norm.to(device)
    output_distribution = str(getattr(model, "output_distribution", "deterministic")).strip().lower()
    use_gaussian = output_distribution == "gaussian"
    n_ens = max(1, int(n_ensemble_samples))

    def compute_csi(threshold, pred, gt):
        event_pred = pred >= threshold
        event_gt = gt >= threshold
        TP = np.sum(event_pred & event_gt)
        FP = np.sum(event_pred & (~event_gt))
        FN = np.sum((~event_pred) & event_gt)
        return TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0

    # Containers to aggregate metrics from all rollout samples
    aggregated_rmse, aggregated_csi_005, aggregated_csi_03 = [], [], []
    aggregated_rmse_vx, aggregated_rmse_vy = [], []
    aggregated_spread_wd, aggregated_spread_vx, aggregated_spread_vy = [], [], []

    for idx, sample in enumerate(tqdm(rollout_dataset, desc="Performing rollout evaluation")):
        run_id = sample.get("run_id", f"sample_{idx}")
        full_dynamic = sample["dynamic"].to(device)
        full_boundary = sample["boundary"].to(device)
        geometry = sample["geometry"]

        start_pred_t = skip_before_timestep + history_steps
        end_pred_t = start_pred_t + rollout_length
        gt_rollout = full_dynamic[start_pred_t:end_pred_t]
        gt_boundary_rollout = full_boundary[start_pred_t:end_pred_t]

        # Containers for per-run data arrays
        wd_pred_list, wd_gt_list, vx_pred_list, vy_pred_list, vx_gt_list, vy_gt_list = [], [], [], [], [], []

        # Containers for per-run metrics
        run_rmse, run_csi_005, run_csi_03 = [], [], []
        run_rmse_vx, run_rmse_vy = [], []
        run_spread_wd, run_spread_vx, run_spread_vy = [], [], []

        if use_gaussian and n_ens > 1:
            current_dynamics = [
                full_dynamic[skip_before_timestep:start_pred_t].clone() for _ in range(n_ens)
            ]
        else:
            current_dynamic = full_dynamic[skip_before_timestep:start_pred_t].clone()
        current_boundary = full_boundary[skip_before_timestep:start_pred_t].clone()
        n_target_channels = int(full_dynamic.shape[-1])

        for t in range(rollout_length):
            with torch.no_grad():
                if use_gaussian:
                    if n_ens > 1:
                        sampled_members = []
                        mean_members = []
                        for ens_idx in range(n_ens):
                            dyn_hist = current_dynamics[ens_idx]
                            dyn_flat = dyn_hist.permute(1, 0, 2).reshape(1, dyn_hist.shape[1], -1)
                            bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                            x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                            out = model(
                                input_geom=geometry.to(device).unsqueeze(0),
                                latent_queries=sample["query_points"].to(device).unsqueeze(0),
                                output_queries=geometry.to(device).unsqueeze(0),
                                x=x,
                            )
                            sampled, mu, _ = _sample_from_packed_gaussian(
                                out,
                                n_channels=n_target_channels,
                                min_logvar=gaussian_min_logvar,
                                max_logvar=gaussian_max_logvar,
                            )
                            sampled_members.append(sampled)
                            mean_members.append(mu)
                        pred_stack = torch.stack(sampled_members, dim=0)  # [E, 1, n_cells, C]
                        pred = torch.stack(mean_members, dim=0).mean(dim=0)
                    else:
                        dyn_flat = current_dynamic.permute(1, 0, 2).reshape(1, current_dynamic.shape[1], -1)
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                        x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                        out = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x,
                        )
                        sampled_single, pred, _ = _sample_from_packed_gaussian(
                            out,
                            n_channels=n_target_channels,
                            min_logvar=gaussian_min_logvar,
                            max_logvar=gaussian_max_logvar,
                        )
                        pred_stack = sampled_single.unsqueeze(0)
                else:
                    dyn_flat = current_dynamic.permute(1, 0, 2).reshape(1, current_dynamic.shape[1], -1)
                    bc_flat = current_boundary.permute(1, 0, 2).reshape(1, current_boundary.shape[1], -1)
                    x = torch.cat([sample["static"].to(device).unsqueeze(0), bc_flat, dyn_flat], dim=2)
                    if fgn_noise_dim is not None and n_ens > 1:
                        preds = []
                        for _ in range(n_ens):
                            z = torch.randn(x.shape[0], fgn_noise_dim, device=device, dtype=x.dtype)
                            p = model(
                                input_geom=geometry.to(device).unsqueeze(0),
                                latent_queries=sample["query_points"].to(device).unsqueeze(0),
                                output_queries=geometry.to(device).unsqueeze(0),
                                x=x,
                                ada_in=z,
                            )
                            preds.append(p)
                        pred_stack = torch.stack(preds, dim=0)
                        pred = pred_stack.mean(dim=0)
                    else:
                        pred = model(
                            input_geom=geometry.to(device).unsqueeze(0),
                            latent_queries=sample["query_points"].to(device).unsqueeze(0),
                            output_queries=geometry.to(device).unsqueeze(0),
                            x=x
                        )
                        pred_stack = pred.unsqueeze(0)

            inv_pred = target_norm.inverse_transform(pred)
            inv_gt = dynamic_norm.inverse_transform(gt_rollout[t].unsqueeze(0))
            inv_pred_ens = target_norm.inverse_transform(pred_stack.squeeze(1))

            # Extract all channels and convert to numpy
            wd_pred, vx_pred, vy_pred = [ch.cpu().numpy() for ch in inv_pred[0].T]
            wd_gt, vx_gt, vy_gt = [ch.cpu().numpy() for ch in inv_gt[0].T]
            wd_ens, vx_ens, vy_ens = [ch.cpu().numpy() for ch in inv_pred_ens.permute(2, 0, 1)]

            # Append data to lists
            wd_pred_list.append(wd_pred)
            wd_gt_list.append(wd_gt)
            vx_pred_list.append(vx_pred)
            vx_gt_list.append(vx_gt)
            vy_pred_list.append(vy_pred)
            vy_gt_list.append(vy_gt)

            # --- Compute metrics for the current step ---
            # Water Depth metrics
            run_rmse.append(np.sqrt(np.mean((wd_pred - wd_gt) ** 2)))
            run_csi_005.append(compute_csi(0.05, wd_pred, wd_gt))
            run_csi_03.append(compute_csi(0.3, wd_pred, wd_gt))

            # Velocity metrics
            run_rmse_vx.append(np.sqrt(np.mean((vx_pred - vx_gt) ** 2)))
            run_rmse_vy.append(np.sqrt(np.mean((vy_pred - vy_gt) ** 2)))
            run_spread_wd.append(float(np.mean(np.std(wd_ens, axis=0))))
            run_spread_vx.append(float(np.mean(np.std(vx_ens, axis=0))))
            run_spread_vy.append(float(np.mean(np.std(vy_ens, axis=0))))

            # Update state for next step
            if use_gaussian:
                if n_ens > 1:
                    for ens_idx in range(n_ens):
                        current_dynamics[ens_idx] = torch.cat(
                            [current_dynamics[ens_idx][1:], pred_stack[ens_idx, 0].unsqueeze(0)], dim=0
                        )
                else:
                    current_dynamic = torch.cat(
                        [current_dynamic[1:], sampled_single.squeeze(0).unsqueeze(0)], dim=0
                    )
            else:
                current_dynamic = torch.cat([current_dynamic[1:], pred.squeeze(0).unsqueeze(0)], dim=0)
            current_boundary = torch.cat([current_boundary[1:], gt_boundary_rollout[t].unsqueeze(0)], dim=0)

        # Convert lists to numpy arrays
        wd_pred_arr, wd_gt_arr = np.stack(wd_pred_list), np.stack(wd_gt_list)
        vx_pred_arr, vy_pred_arr = np.stack(vx_pred_list), np.stack(vy_pred_list)
        vx_gt_arr, vy_gt_arr = np.stack(vx_gt_list), np.stack(vy_gt_list)

        # Append per-run metrics to aggregated lists
        aggregated_rmse.append(np.array(run_rmse))
        aggregated_csi_005.append(np.array(run_csi_005))
        aggregated_csi_03.append(np.array(run_csi_03))
        aggregated_rmse_vx.append(np.array(run_rmse_vx))
        aggregated_rmse_vy.append(np.array(run_rmse_vy))
        aggregated_spread_wd.append(np.array(run_spread_wd))
        aggregated_spread_vx.append(np.array(run_spread_vx))
        aggregated_spread_vy.append(np.array(run_spread_vy))

        generate_publication_maps(
            geometry=geometry,
            wd_gt_array=wd_gt_arr, wd_pred_array=wd_pred_arr,
            vx_gt_array=vx_gt_arr, vy_gt_array=vy_gt_arr,
            vx_pred_array=vx_pred_arr, vy_pred_array=vy_pred_arr,
            steps=[12, 24, 36, 48, 60, 72],
            out_dir=os.path.join(out_dir, "figures"),
            run_id=run_id,
            filename_prefix="flood"
        )

        create_rollout_animation(
            geometry=geometry,
            wd_gt=wd_gt_arr, wd_pred=wd_pred_arr,
            vx_gt=vx_gt_arr, vy_gt=vy_gt_arr,
            vx_pred=vx_pred_arr, vy_pred=vy_pred_arr,
            run_id=run_id, out_dir=out_dir, dt_seconds=dt
        )

    # After all runs, aggregate and plot final metrics
    if aggregated_rmse:
        # Stack and calculate mean/std for all metrics
        metrics = {
            'rmse_wd': np.stack(aggregated_rmse),
            'csi_005': np.stack(aggregated_csi_005),
            'csi_03': np.stack(aggregated_csi_03),
            'rmse_vx': np.stack(aggregated_rmse_vx),
            'rmse_vy': np.stack(aggregated_rmse_vy),
            'spread_wd': np.stack(aggregated_spread_wd),
            'spread_vx': np.stack(aggregated_spread_vx),
            'spread_vy': np.stack(aggregated_spread_vy),
        }
        stats = {key: {'mean': arr.mean(axis=0), 'std': arr.std(axis=0)} for key, arr in metrics.items()}

        time_hours = (np.arange(1, rollout_length + 1) * dt) / 3600.0

        fig, axs = plt.subplots(4, 2, figsize=(16, 24), tight_layout=True)
        axs = axs.flatten()

        plot_info = {
            0: ('rmse_wd', 'RMSE (Depth)', 'RMSE (m)'),
            1: ('rmse_vx', 'RMSE (VX)', 'RMSE (m/s)'),
            2: ('rmse_vy', 'RMSE (VY)', 'RMSE (m/s)'),
            3: ('csi_005', 'CSI (0.05m)', 'CSI'),
            4: ('csi_03', 'CSI (0.3m)', 'CSI'),
            5: ('spread_wd', 'Spread (Depth)', 'Std (m)'),
            6: ('spread_vx', 'Spread (VX)', 'Std (m/s)'),
            7: ('spread_vy', 'Spread (VY)', 'Std (m/s)'),
        }

        for i in range(len(axs)):
            ax = axs[i]
            if i in plot_info:
                key, title, ylabel = plot_info[i]
                mean, std = stats[key]['mean'], stats[key]['std']
                ax.plot(time_hours, mean, label=f'{title} Mean', marker='o')
                ax.fill_between(time_hours, mean - std, mean + std, alpha=0.3, label='±1 Std')
                ax.set_title(title + " over Time")
                ax.set_xlabel("Time (hour)")
                ax.set_ylabel(ylabel)
                ax.legend()
                ax.grid(True)
            else:
                ax.set_visible(False)  # Hide unused subplots

        summary_path = os.path.join(out_dir, "rollout_metrics_summary.png")
        plt.savefig(summary_path)
        plt.close(fig)
        logging.getLogger("flood_train").info("Saved aggregated rollout metrics plot to %s", summary_path)

        # Save data for external plotting
        npz_data = {'time_hours': time_hours}
        for key, stat_dict in stats.items():
            npz_data[f'{key}_mean'] = stat_dict['mean']
            npz_data[f'{key}_std'] = stat_dict['std']
            npz_data[f'{key}_all'] = metrics[key]

        data_save_path = os.path.join(out_dir, "rollout_metrics_data.npz")
        np.savez(data_save_path, **npz_data)
        logging.getLogger("flood_train").info("Saved aggregated rollout metrics data to %s", data_save_path)


###############################################################################
# 10) MAIN
###############################################################################
def main():
    config, device, is_logger = load_config_and_setup()

    # Logging: file (rotating) + console, config-driven level and path
    log_level = getattr(config, "log_level", "INFO")
    log_file = getattr(config, "log_file", None)
    if log_file is not None:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            save_dir = getattr(config.checkpoint, "save_dir", ".")
            log_path = Path(save_dir) / log_path
    else:
        save_dir = getattr(config.checkpoint, "save_dir", ".")
        log_path = Path(save_dir) / "training.log"
    logger = setup_logging(
        log_level=log_level,
        log_file=str(log_path),
        logger_name="flood_train",
    )
    logger.info("Config loaded; device=%s", device)

    # Reproducibility: set all RNG seeds and deterministic CuDNN (override setup() for full reproducibility)
    seed = getattr(config.distributed, "seed", 123)
    deterministic = getattr(config, "deterministic", True)
    set_seed(seed, deterministic=deterministic)
    logger.info("Random seed set to %s (deterministic=%s)", seed, deterministic)

    # Possibly adjust FNO modes
    if hasattr(config.data, "resolution") and (config.data.resolution < config.gino.fno_n_modes[0]):
        config.gino.fno_n_modes = [config.data.resolution] * 2

    # Initialize wandb if needed
    wandb_init_args = {}
    if config.wandb.log and is_logger:
        wandb.login(key=get_wandb_api_key())
        wandb_name = config.wandb.name if config.wandb.name else f"flood-run_{getattr(config.data, 'resolution', 64)}"
        wandb_init_args = dict(
            config=config,
            name=wandb_name,
            group=config.wandb.group,
            project=config.wandb.project,
            entity=config.wandb.entity
        )
        if config.wandb.sweep:
            for key in wandb.config.keys():
                config.params[key] = wandb.config[key]
        wandb.init(**wandb_init_args)

    # ---------------------- Setup training dataset (HDF only) -----------------------------
    skip_before_timestep = getattr(config.data, "skip_before_timestep", 0)
    noise_type = getattr(config.data, "noise_type", "none")
    noise_std = getattr(config.data, "noise_std", None)
    static_text_files = getattr(config.data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"])
    n_history = config.data.n_history
    target_variables = parse_target_variables(getattr(config.data, "target_variables", ["wd", "vx", "vy"]))
    n_target_channels = len(target_variables)
    # Optionally (over)write train.txt with all existing *\.hdf run IDs in data.root
    if getattr(config.data, "write_train_txt", False):
        run_ids = write_train_txt_from_data_root(
            config.data.root,
            train_txt=getattr(config.data, "train_txt", "train.txt"),
            hdf_suffix=".hdf",
        )
        logger.info("Wrote train.txt with %s run IDs from %s", len(run_ids), config.data.root)
    # Static: 2 from HDF (elevation, area) + text files (CS, CU, FA) — aligned with HEC_RAS_Automation
    n_static = 2 + len(static_text_files)
    data_channels = n_static + n_history * 1 + n_history * n_target_channels
    if hasattr(config, "gino"):
        setattr(config.gino, "data_channels", data_channels)
        setattr(config.gino, "out_channels", n_target_channels)
    ar_rollout_steps = max(1, int(getattr(config.opt, "ar_rollout_steps", 1)))
    full_dataset = FloodDatasetHDF(
        data_root=config.data.root,
        n_history=config.data.n_history,
        query_res=getattr(config.data, "query_res", [64, 64]),
        run_ids=None,
        train_txt=getattr(config.data, "train_txt", "train.txt"),
        static_text_files=static_text_files,
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=skip_before_timestep,
        noise_type=noise_type,
        noise_std=noise_std,
        ar_rollout_steps=ar_rollout_steps,
        target_variables=target_variables,
    )
    n_samples_max = getattr(config.data, "n_samples_max", None)
    if n_samples_max is not None:
        total_avail = len(full_dataset)
        n_samples_max = int(n_samples_max)  # CLI may pass str
        n_use = min(n_samples_max, total_avail)
        full_dataset = Subset(full_dataset, range(n_use))
        logger.info("Limited to %s samples (n_samples_max=%s)", n_use, n_samples_max)

    total_len = len(full_dataset)
    train_sz = max(1, int(0.9 * total_len))
    test_sz = total_len - train_sz
    train_data_raw, test_data_raw_temp = random_split(
        full_dataset, [train_sz, test_sz], generator=make_split_generator(seed)
    )

    logger.info("Dataset: total=%s, train=%s, test (one-step)=%s", total_len, train_sz, test_sz)

    # No leakage: normalizers are fit only on train_data_raw. Test data is transformed with
    # train-fit stats in NormalizedDatasetOnTheFly; evaluation uses model.eval() and torch.no_grad().
    # Normalizers: load from disk if path exists and is set; otherwise fit and optionally save
    normalizer_path = getattr(config.data, "normalizer_path", None)
    if normalizer_path is not None:
        normalizer_path = Path(normalizer_path)
        if not normalizer_path.is_absolute():
            normalizer_path = Path(config.data.root) / normalizer_path
    if normalizer_path is not None and normalizer_path.exists():
        normalizers = load_normalizers(normalizer_path, device=None)
        logger.info("Loaded normalizers from %s", normalizer_path)
    else:
        norm_chunk_size = getattr(config.data, "normalizer_chunk_size", 10000)
        normalizers = fit_normalizers_streaming(
            train_data_raw, chunk_size=norm_chunk_size, expect_target=True
        )
        if normalizer_path is not None:
            save_normalizers(normalizers, normalizer_path)
            logger.info("Saved normalizers to %s", normalizer_path)

    train_normalized_dataset = NormalizedDatasetOnTheFly(
        train_data_raw, normalizers, query_res=config.data.query_res
    )
    num_workers = getattr(config.data, "num_workers", 0)

    def _cfg_get(obj, key, default):
        try:
            return getattr(obj, key)
        except (AttributeError, KeyError):
            return default

    pin_memory = bool(_cfg_get(config.data, "pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(_cfg_get(config.data, "persistent_workers", True))
    prefetch_default = 1 if os.name == "nt" else 2
    prefetch_factor = int(_cfg_get(config.data, "prefetch_factor", prefetch_default))

    # Windows + multiprocessing + long-running HDF workloads can be less stable with
    # persistent workers; prefer safer defaults unless explicitly overridden in code.
    if os.name == "nt" and num_workers > 0 and persistent_workers:
        logger.warning(
            "Windows detected with num_workers=%s and persistent_workers=True; "
            "overriding persistent_workers=False for DataLoader stability.",
            num_workers,
        )
        persistent_workers = False

    worker_init_fn = partial(dataloader_worker_init, base_seed=seed) if num_workers > 0 else None
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_normalized_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        generator=make_dataloader_generator(seed),
        **loader_kwargs,
    )

    # One-step test: same on-the-fly normalization (no stacking test set)
    test_normalized_dataset = NormalizedDatasetOnTheFly(
        test_data_raw_temp, normalizers, query_res=config.data.query_res
    )
    test_loader = DataLoader(
        test_normalized_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    logger.info(
        "Data: device=%s, train_samples=%s, test_samples=%s, batch_size=%s, noise=%s std=%s",
        device, train_sz, len(test_normalized_dataset), config.data.batch_size, noise_type, noise_std,
    )

    # Model
    model = get_model(config)

    # Optimizer/scheduler
    optimizer = AdamW(model.parameters(),
                      lr=config.opt.learning_rate,
                      weight_decay=config.opt.weight_decay)

    if config.opt.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=getattr(config.opt, "gamma", 0.5),
            patience=getattr(config.opt, "scheduler_patience", 5),
            mode=getattr(config.opt, "scheduler_mode", "min"),
            threshold=getattr(config.opt, "scheduler_threshold", 1e-4),
            threshold_mode=getattr(config.opt, "scheduler_threshold_mode", "rel"),
            cooldown=getattr(config.opt, "scheduler_cooldown", 0),
            min_lr=getattr(config.opt, "scheduler_min_lr", 0.0),
        )
    elif config.opt.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=getattr(config.opt, "scheduler_T_max", 200),
            eta_min=getattr(config.opt, "scheduler_eta_min", 0.0),
        )
    elif config.opt.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=getattr(config.opt, "step_size", 50),
            gamma=getattr(config.opt, "gamma", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler {config.opt.scheduler}")

    # Loss (LpLoss(d=2,p=2): relative L2 by default; use 'l2_abs' if relative plateaus)
    # reduction='sum' so train_err = sum(loss)/n_samples = mean per sample (same scale as test_l2)
    l2loss = LpLoss(d=2, p=2)
    use_fgn = bool(getattr(config.gino, "use_fgn_noise", False))
    output_distribution = str(getattr(config.gino, "output_distribution", "deterministic")).strip().lower()
    if output_distribution not in {"deterministic", "gaussian"}:
        raise ValueError(
            f"Unknown gino.output_distribution={output_distribution!r}. "
            "Use 'deterministic' or 'gaussian'."
        )
    setattr(config.gino, "output_distribution", output_distribution)
    training_loss_name = str(getattr(config.opt, "training_loss", "l2")).strip().lower()
    setattr(config.opt, "training_loss", training_loss_name)

    if training_loss_name == "gaussian_nll" and output_distribution != "gaussian":
        raise ValueError(
            "training_loss='gaussian_nll' requires gino.output_distribution='gaussian'."
        )
    fno_norm_mode = getattr(config.gino, "fno_norm", None)
    fno_norm_mode = None if fno_norm_mode is None else str(fno_norm_mode).strip().lower()
    if output_distribution == "gaussian" and use_fgn:
        raise ValueError(
            "gino.output_distribution='gaussian' requires gino.use_fgn_noise=false."
        )
    if output_distribution == "gaussian" and fno_norm_mode != "instance_norm":
        if fno_norm_mode == "ada_in":
            warnings.warn(
                "gino.output_distribution='gaussian' is incompatible with gino.fno_norm='ada_in'. "
                "Overriding gino.fno_norm -> 'instance_norm'.",
                UserWarning,
                stacklevel=2,
            )
        else:
            warnings.warn(
                "gino.output_distribution='gaussian' uses gino.fno_norm='instance_norm' in this "
                f"pipeline (got {fno_norm_mode!r}). Overriding gino.fno_norm -> 'instance_norm'.",
                UserWarning,
                stacklevel=2,
            )
        setattr(config.gino, "fno_norm", "instance_norm")
    if training_loss_name == "crps" and not use_fgn:
        raise ValueError(
            "training_loss='crps' requires gino.use_fgn_noise=true in this pipeline."
        )

    if training_loss_name == "l2":
        train_loss_fn = l2loss
        if use_fgn:
            warnings.warn(
                "gino.use_fgn_noise is True but training_loss is 'l2'. "
                "FGN noise will not be trained; use training_loss: 'crps' for probabilistic FGN.",
                UserWarning,
                stacklevel=2,
            )
    elif training_loss_name == "crps" and use_fgn:
        crps_n_samples = max(2, int(getattr(config.opt, "crps_n_samples", 2)))
        crps_channel_weights = getattr(config.opt, "crps_channel_weights", None)
        train_loss_fn = CRPSLoss(
            n_samples=crps_n_samples,
            channel_weights=crps_channel_weights,
            reduction="mean",
        )
    elif training_loss_name == "gaussian_nll":
        train_loss_fn = GaussianNLLLoss(
            channel_weights=getattr(config.opt, "crps_channel_weights", None),
            reduction="mean",
            min_logvar=_safe_float(getattr(config.opt, "gaussian_min_logvar", -9.0), -9.0),
            max_logvar=_safe_float(getattr(config.opt, "gaussian_max_logvar", 4.0), 4.0),
            logvar_reg_weight=_safe_float(
                getattr(config.opt, "gaussian_logvar_reg_weight", 1e-6), 1e-6
            ),
        )
    elif training_loss_name == "l2_abs":
        # Absolute L2 (||pred-y||) instead of relative (||pred-y||/||y||); can help if relative plateaus
        train_loss_fn = lambda y_pred, y, **kw: l2loss.abs(y_pred, y)
        if use_fgn:
            warnings.warn(
                "gino.use_fgn_noise is True but training_loss is 'l2_abs'. FGN will not be trained.",
                UserWarning,
                stacklevel=2,
            )
    else:
        raise ValueError(
            f"Unknown training loss: {config.opt.training_loss}. "
            "Use 'l2', 'l2_abs', 'gaussian_nll', or for FGN: 'crps' with gino.use_fgn_noise: true."
        )

    if config.opt.testing_loss == "l2":
        test_loss_fn = l2loss
    else:
        test_loss_fn = l2loss

    # Eval metrics:
    # - FGN/CRPS: L2 on ensemble mean + CRPS.
    # - Gaussian NLL: L2 on predictive mean + Gaussian NLL on packed output.
    if output_distribution == "gaussian" and training_loss_name == "gaussian_nll":
        eval_losses = {
            "l2": test_loss_fn,
            "gaussian_nll": GaussianNLLLoss(
                channel_weights=getattr(config.opt, "crps_channel_weights", None),
                reduction="mean",
                min_logvar=_safe_float(getattr(config.opt, "gaussian_min_logvar", -9.0), -9.0),
                max_logvar=_safe_float(getattr(config.opt, "gaussian_max_logvar", 4.0), 4.0),
                logvar_reg_weight=0.0,
            ),
        }
    elif use_fgn and training_loss_name == "crps":
        crps_n_samples = max(2, int(getattr(config.opt, "crps_n_samples", 2)))
        crps_channel_weights = getattr(config.opt, "crps_channel_weights", None)
        eval_losses = {
            "l2": test_loss_fn,
            "crps": CRPSLoss(n_samples=crps_n_samples, channel_weights=crps_channel_weights, reduction="mean"),
        }
    else:
        eval_losses = {config.opt.testing_loss: test_loss_fn}

    # DataProcessor: training loss is always in normalized space (pred and y from dataset are normalized).
    # Eval: when inverse_test=True, pred and y are inverse-transformed so test_l2/test_crps are in physical space.
    inverse_test = getattr(config, "inverse_test", True)
    data_processor = FloodGINODataProcessor(
        device=device,
        target_norm=normalizers.get("target", None),
        inverse_test=inverse_test,
        output_distribution=output_distribution,
    )
    data_processor.wrap(model)

    # Trainer (FGN: two forwards + CRPS per batch)
    fgn_noise_dim = getattr(config.gino, "fgn_noise_dim", 32)  # used for rollout ensemble when use_fgn
    use_progress_bar = getattr(config, "use_progress_bar", True)
    scheduler_monitor = getattr(config.opt, "scheduler_monitor", "train_err")
    eval_interval = getattr(config.wandb, "eval_interval", 1)
    if use_fgn and training_loss_name == "crps":
        crps_l2_weight = getattr(config.opt, "crps_l2_weight", 0.5)
        ar_finetune_start_epoch = max(0, int(getattr(config.opt, "ar_finetune_start_epoch", 0)))
        ar_curriculum_epochs_per_step = max(0, int(getattr(config.opt, "ar_curriculum_epochs_per_step", 0)))
        use_flood_crps_spatial_weights = getattr(config.opt, "flood_crps_spatial_weights", False)
        flood_crps_wet_threshold = getattr(config.opt, "wet_threshold", 0.01)
        flood_crps_wet_smooth_scale = getattr(config.opt, "wet_smooth_scale", 0.02)
        flood_crps_dry_weight_alpha = getattr(config.opt, "dry_weight_alpha", 0.1)
        crps_n_samples = max(2, int(getattr(config.opt, "crps_n_samples", 2)))
        trainer = FGNTrainer(
            model=model,
            n_epochs=config.opt.n_epochs,
            data_processor=data_processor,
            device=device,
            wandb_log=config.wandb.log,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
            fgn_noise_dim=fgn_noise_dim,
            crps_n_samples=crps_n_samples,
            rel_l2_loss_fn=l2loss,
            crps_l2_weight=crps_l2_weight,
            ar_finetune_start_epoch=ar_finetune_start_epoch,
            ar_rollout_steps=ar_rollout_steps,
            ar_curriculum_epochs_per_step=ar_curriculum_epochs_per_step,
            use_flood_crps_spatial_weights=use_flood_crps_spatial_weights,
            flood_crps_wet_threshold=flood_crps_wet_threshold,
            flood_crps_wet_smooth_scale=flood_crps_wet_smooth_scale,
            flood_crps_dry_weight_alpha=flood_crps_dry_weight_alpha,
            static_normalizer=normalizers.get("static") if use_flood_crps_spatial_weights else None,
            use_hazard_proxy_crps=getattr(config.opt, "hazard_proxy_crps", False),
            hazard_proxy_crps_weight=getattr(config.opt, "hazard_proxy_crps_weight", 0.15),
            ar_pooled_crps_gamma=getattr(config.opt, "ar_pooled_crps_gamma", 1.0),
        )
    elif output_distribution == "gaussian" and training_loss_name == "gaussian_nll":
        trainer = GaussianNLLTrainer(
            model=model,
            n_epochs=config.opt.n_epochs,
            data_processor=data_processor,
            device=device,
            wandb_log=config.wandb.log,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
            rel_l2_loss_fn=l2loss,
            ar_finetune_start_epoch=max(0, int(getattr(config.opt, "ar_finetune_start_epoch", 0))),
            ar_rollout_steps=ar_rollout_steps,
            ar_curriculum_epochs_per_step=max(0, int(getattr(config.opt, "ar_curriculum_epochs_per_step", 0))),
            gaussian_min_logvar=_safe_float(getattr(config.opt, "gaussian_min_logvar", -9.0), -9.0),
            gaussian_max_logvar=_safe_float(getattr(config.opt, "gaussian_max_logvar", 4.0), 4.0),
        )
    else:
        trainer = Trainer(
            model=model,
            n_epochs=config.opt.n_epochs,
            data_processor=data_processor,
            device=device,
            wandb_log=config.wandb.log,
            verbose=is_logger,
            logger=logger,
            use_progress_bar=use_progress_bar,
            scheduler_monitor=scheduler_monitor,
            eval_interval=eval_interval,
        )

    # Optional: verify gradient flow and overfit one batch (set verify_training: true in config)
    try:
        do_verify = config.verify_training
    except (KeyError, AttributeError):
        do_verify = False
    if do_verify:
        logger.info("--- Training verification ---")
        trainer.optimizer = optimizer  # trainer.train() sets these; set early for verification
        trainer.regularizer = None
        trainer.n_samples = 0
        trainer.epoch = 0
        trainer.model = trainer.model.to(trainer.device)
        if trainer.data_processor is not None and trainer.data_processor.device != trainer.device:
            trainer.data_processor = trainer.data_processor.to(trainer.device)
        verify_training_gradient_flow(trainer, train_loader, train_loss_fn)
        overfit_sanity_check(trainer, train_loader, train_loss_fn, optimizer, n_steps=15)
        logger.info("--- End verification ---")

    # Train
    trainer.train(
        train_loader=train_loader,
        test_loaders={'test': test_loader},
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=train_loss_fn,
        eval_losses=eval_losses,
        regularizer=None,
        save_every=1,
        save_dir=config.checkpoint.save_dir,
        resume_from_dir=config.checkpoint.resume_from_dir
    )

    # ----------------- Optional: rollout evaluation on new data -----------------------
    run_rollout = getattr(config.rollout, "run_after_training", False)
    if not run_rollout:
        if is_logger:
            logger.info("Skipping rollout (run_after_training: false).")
        if config.wandb.log:
            wandb.finish()
        return
    if n_target_channels != 3:
        if is_logger:
            logger.warning(
                "Skipping rollout plotting/eval because target_variables=%s (C_out=%s) "
                "is not supported by rollout visualization code (expects [wd, vx, vy]).",
                target_variables, n_target_channels,
            )
        if config.wandb.log:
            wandb.finish()
        return

    rollout_length = config.data.rollout_length
    history_steps = config.data.n_history
    rollout_skip_before_timestep = getattr(config.data, "skip_before_timestep", 0)

    rollout_data_root = config.rollout_data.root
    rollout_test_dataset = FloodRolloutTestDatasetHDF(
        rollout_data_root=rollout_data_root,
        n_history=history_steps,
        rollout_length=rollout_length,
        run_ids=None,
        test_txt=getattr(config.rollout_data, "test_txt", "test.txt"),
        static_text_files=getattr(config.rollout_data, "static_text_files", ["M40_CS.txt", "M40_CU.txt", "M40_FA.txt"]),
        hdf_suffix=".hdf",
        raise_on_smaller=True,
        skip_before_timestep=rollout_skip_before_timestep,
    )

    # Normalizing rollout data
    rollout_geom_list, rollout_static_list, rollout_boundary_list, rollout_dyn_list, _ = collect_all_fields(
        rollout_test_dataset, expect_target=False
    )

    def transform_with_existing_normalizers(geom_list, static_list, boundary_list, dyn_list, normalizers):
        ref_device = None
        for key in ("dynamic", "target", "geometry"):
            if key in normalizers and normalizers[key] is not None and hasattr(normalizers[key], "mean"):
                ref_device = normalizers[key].mean.device
                break
        if ref_device is None:
            ref_device = torch.device("cpu")

        for key in ["geometry", "static", "boundary", "dynamic", "target"]:
            if key in normalizers and normalizers[key] is not None:
                normalizers[key].to(ref_device)

        geometry_big = torch.stack(geom_list, dim=0) if geom_list else None
        static_big = torch.stack(static_list, dim=0) if static_list else None
        boundary_big = torch.stack(boundary_list, dim=0) if boundary_list else None
        dynamic_big = torch.stack(dyn_list, dim=0) if dyn_list else None

        if ref_device is not None:
            if geometry_big is not None:
                geometry_big = geometry_big.to(ref_device)
            if static_big is not None:
                static_big = static_big.to(ref_device)
            if boundary_big is not None:
                boundary_big = boundary_big.to(ref_device)
            if dynamic_big is not None:
                dynamic_big = dynamic_big.to(ref_device)

        if geometry_big is not None and "geometry" in normalizers:
            geometry_big = normalizers["geometry"].transform(geometry_big)
        if static_big is not None and "static" in normalizers:
            static_big = normalizers["static"].transform(static_big)
        if boundary_big is not None and "boundary" in normalizers:
            boundary_big = normalizers["boundary"].transform(boundary_big)
        if dynamic_big is not None and "dynamic" in normalizers:
            dynamic_big = normalizers["dynamic"].transform(dynamic_big)

        return {
            "geometry": geometry_big,
            "static": static_big,
            "boundary": boundary_big,
            "dynamic": dynamic_big,
        }

    transformed_rollout = transform_with_existing_normalizers(
        rollout_geom_list,
        rollout_static_list,
        rollout_boundary_list,
        rollout_dyn_list,
        normalizers
    )

    normalized_rollout_samples = []
    for i in range(len(rollout_test_dataset)):
        normalized_rollout_samples.append({
            "run_id": rollout_test_dataset.valid_run_ids[i],
            "geometry": transformed_rollout["geometry"][i],
            "static": transformed_rollout["static"][i],
            "boundary": transformed_rollout["boundary"][i],
            "dynamic": transformed_rollout["dynamic"][i],
        })

    rollout_normalized_dataset = NormalizedRolloutTestDataset(
        normalized_samples=normalized_rollout_samples,
        query_res=config.data.query_res
    )

    rollout_prediction(
        trainer=trainer,
        rollout_dataset=rollout_normalized_dataset,
        rollout_length=rollout_length,
        history_steps=history_steps,
        dynamic_norm=normalizers["dynamic"],
        target_norm=normalizers["target"],
        device=device,
        skip_before_timestep=rollout_skip_before_timestep,
        dt=config.data.dt,
        out_dir=config.rollout.out_dir,
        fgn_noise_dim=fgn_noise_dim if use_fgn else None,
        n_ensemble_samples=getattr(config.rollout, "n_ensemble_samples", 1),
        gaussian_min_logvar=_safe_float(getattr(config.opt, "gaussian_min_logvar", -9.0), -9.0),
        gaussian_max_logvar=_safe_float(getattr(config.opt, "gaussian_max_logvar", 4.0), 4.0),
    )

    if config.wandb.log:
        wandb.finish()


if __name__ == "__main__":
    main()
