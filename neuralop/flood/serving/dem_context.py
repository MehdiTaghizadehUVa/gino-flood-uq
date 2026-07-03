"""Build a rectangular DEM context image for serving map products.

The deployed coastal FGN geometry contains only the active computational cells,
but publication figures use a rectangular terrain context behind those cells.
This module builds that background from the same bundle geometry and elevation
values so serving maps can match the publication style without changing model
outputs or cell-level scientific data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _nearest_fill(grid: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(grid)
    if not np.any(mask):
        return np.asarray(grid, dtype=np.float32)
    try:
        from scipy import ndimage

        _, indices = ndimage.distance_transform_edt(mask, return_indices=True)
        return np.asarray(grid[tuple(indices)], dtype=np.float32)
    except Exception:
        filled = np.asarray(grid, dtype=np.float32).copy()
        finite = filled[np.isfinite(filled)]
        fallback = float(np.nanmedian(finite)) if finite.size else 0.0
        filled[mask] = fallback
        return filled


def _upsample(grid: np.ndarray, factor: int) -> np.ndarray:
    factor = max(int(factor), 1)
    if factor == 1:
        return np.asarray(grid, dtype=np.float32)
    try:
        from scipy import ndimage

        return np.asarray(ndimage.zoom(grid, (factor, factor), order=1), dtype=np.float32)
    except Exception:
        return np.asarray(np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1), dtype=np.float32)


def build_dem_context(
    *,
    geometry_xy: np.ndarray,
    elevation_m: np.ndarray,
    output_path: str | Path,
    crs: str = "EPSG:32618",
    upsample_factor: int = 8,
    pad_fraction: float = 0.025,
    metadata: dict[str, Any] | None = None,
) -> Path:
    xy = np.asarray(geometry_xy, dtype=np.float64)
    elev = np.asarray(elevation_m, dtype=np.float64).reshape(-1)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("geometry_xy must have shape [n_cells, 2].")
    if elev.shape[0] != xy.shape[0]:
        raise ValueError("elevation_m must have one value per geometry cell.")

    x = xy[:, 0]
    y = xy[:, 1]
    ux = np.unique(x)
    uy = np.unique(y)
    nx = int(ux.size)
    ny = int(uy.size)
    if nx < 2 or ny < 2:
        raise ValueError("DEM context requires at least a 2x2 coordinate lattice.")

    ix = np.searchsorted(ux, x)
    iy = np.searchsorted(uy, y)
    grid = np.full((ny, nx), np.nan, dtype=np.float32)
    grid[iy, ix] = elev.astype(np.float32)
    filled = _nearest_fill(grid)
    smooth = _upsample(filled, upsample_factor)

    finite = elev[np.isfinite(elev)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = np.nanquantile(finite, [0.01, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if np.isclose(lo, hi):
            hi = lo + 1.0
    clipped = np.clip(smooth, float(lo), float(hi))
    normed = (clipped - float(lo)) / max(float(hi - lo), 1.0e-12)

    import matplotlib

    matplotlib.use("Agg", force=True)
    from neuralop.flood.eval import render as eval_render

    cmap = eval_render._dem_elevation_cmap("hecras_dem")
    # imshow(origin="upper") maps row 0 to ymax, so flip the y-ascending grid.
    image = cmap(np.flipud(normed))[..., :3].astype(np.float32)
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xpad = max((xmax - xmin) * float(pad_fraction), 1.0)
    ypad = max((ymax - ymin) * float(pad_fraction), 1.0)
    extent = np.asarray([xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad], dtype=np.float64)

    payload = {
        "source": "bundle geometry/static elevation",
        "crs": crs,
        "n_cells": int(xy.shape[0]),
        "lattice_shape": [ny, nx],
        "structured_coverage": float(xy.shape[0] / float(nx * ny)),
        "upsample_factor": int(max(upsample_factor, 1)),
        "pad_fraction": float(pad_fraction),
        "elevation_quantiles": [float(lo), float(hi)],
    }
    if metadata:
        payload.update(metadata)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        image=image,
        extent=extent,
        metadata_json=np.asarray(json.dumps(payload, sort_keys=True)),
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a rectangular DEM context NPZ for FGN serving maps.")
    parser.add_argument("--geometry", required=True, type=Path, help="Path to bundle domain/geometry.npy.")
    parser.add_argument("--static", required=True, type=Path, help="Path to bundle domain/static.npy.")
    parser.add_argument("--output", required=True, type=Path, help="Output dem_context.npz path.")
    parser.add_argument("--elevation-column", default=0, type=int, help="Static tensor column containing elevation in meters.")
    parser.add_argument("--upsample-factor", default=8, type=int, help="Image upsample factor over the coordinate lattice.")
    parser.add_argument("--crs", default="EPSG:32618")
    args = parser.parse_args(argv)

    geometry = np.load(args.geometry)
    static = np.load(args.static)
    if static.ndim != 2 or args.elevation_column < 0 or args.elevation_column >= static.shape[1]:
        raise ValueError(f"static array has shape {static.shape}; invalid elevation column {args.elevation_column}.")
    out = build_dem_context(
        geometry_xy=geometry,
        elevation_m=static[:, args.elevation_column],
        output_path=args.output,
        crs=args.crs,
        upsample_factor=args.upsample_factor,
    )
    print(out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
