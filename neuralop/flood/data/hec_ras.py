"""HEC-RAS HDF readers for WV flood datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore

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
