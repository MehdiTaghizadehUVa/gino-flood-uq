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


def _resolve_member_hdf_boundary_channels(boundary_channels, paths: dict) -> list[dict]:
    if boundary_channels is None:
        return [
            {
                "name": "inflow",
                "hdf_path": paths["us_inflow"],
                "column_index": -1,
            }
        ]
    resolved = []
    for channel in boundary_channels:
        entry = dict(channel)
        entry.setdefault("name", f"boundary_{len(resolved)}")
        hdf_path = entry.get("hdf_path", None)
        if hdf_path is None:
            if len(boundary_channels) == 1 and "us_inflow" in paths:
                hdf_path = paths["us_inflow"]
                entry.setdefault("column_index", -1)
            else:
                raise ValueError(
                    f"member_hdf boundary channel {entry['name']!r} requires hdf_path."
                )
        entry["hdf_path"] = str(hdf_path)
        if (
            entry.get("column_index", None) is None
            and len(boundary_channels) == 1
            and str(hdf_path) == str(paths.get("us_inflow", ""))
        ):
            entry["column_index"] = -1
        resolved.append(entry)
    return resolved


def _read_boundary_channel_slice(handle, channel: dict, *, t0: int, t1: int) -> np.ndarray:
    raw = np.asarray(handle[channel["hdf_path"]][t0:t1], dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, np.newaxis]
    elif raw.ndim != 2:
        raise ValueError(
            f"Boundary dataset {channel['hdf_path']} must be 1D or 2D, got {tuple(raw.shape)}."
        )

    column_index = channel.get("column_index", None)
    if column_index is None:
        if raw.shape[1] != 1:
            raise ValueError(
                f"Boundary dataset {channel['hdf_path']} produced shape {tuple(raw.shape)}. "
                "Provide column_index or use a single-column dataset for each channel."
            )
        return raw

    column_index = int(column_index)
    if column_index < 0:
        column_index += raw.shape[1]
    if column_index < 0 or column_index >= raw.shape[1]:
        raise IndexError(
            f"Boundary dataset {channel['hdf_path']} has {raw.shape[1]} columns; "
            f"column_index={channel['column_index']} is out of bounds."
        )
    return raw[:, column_index : column_index + 1]


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
    boundary_channels: list[dict] | None = None,
) -> tuple:
    """
    Read geometry and time-series slice from a HEC-RAS 2D result HDF file.
    Uses Cell Points for geometry; result arrays (wd, vx, vy) are subset by cell_index
    when provided (so geometry and results have same n_cells = n_cell_points).

    Returns
    -------
    geometry : np.ndarray (n_cell_points, 2) from Cell Points
    wd, vx, vy : np.ndarray (T, n_cell_points)
    inflow : np.ndarray (T, bc_dim)
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
        resolved_boundary_channels = _resolve_member_hdf_boundary_channels(
            boundary_channels, paths
        )
        inflow_list = [
            _read_boundary_channel_slice(f, channel, t0=t0, t1=t1)
            for channel in resolved_boundary_channels
        ]
        inflow = (
            np.concatenate(inflow_list, axis=1)
            if inflow_list
            else np.zeros((max(int(t1) - int(t0), 0), 0), dtype=np.float32)
        )
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
