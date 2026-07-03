"""Serving-safe spatial map rendering helpers.

This adapter reuses the evaluation renderer's map machinery so web artifacts
match the coastal calibration figures' cartographic style without pulling
ground-truth/error diagnostics into user-serving products.
"""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SERVING_DEM_BASEMAP_ALPHA = 0.55
SERVING_DEM_VMIN_M = -15.1
SERVING_DEM_VMAX_M = 19.9
SERVING_MAP_FACE_COLOR = "#FFFFFF"


DEFAULT_SERVING_VISUALIZATION_CONFIG = {
    "map": {
        "enabled": True,
        "mode": "dem_elevation",
        "crs": "EPSG:32618",
        "provider": "local_elevation",
        "fallback": "elevation_hillshade",
        "cache_scope": "run_extent",
        "alpha": SERVING_DEM_BASEMAP_ALPHA,
        "dem_cmap": "hecras_dem",
        "dem_vmin": SERVING_DEM_VMIN_M,
        "dem_vmax": SERVING_DEM_VMAX_M,
        "dem_quantiles": [0.01, 0.99],
    },
    "wd": {
        "colormap": "cyan_depth",
        "overlay_alpha": 0.88,
        "show_wet_edge": False,
        "wet_edge_threshold_m": 0.05,
    },
    "diagnostics": {
        "basemap_alpha": SERVING_DEM_BASEMAP_ALPHA,
        "zero_fraction": 0.03,
        "zero_threshold": 1.0e-10,
        "spread_colormap": "spread_violet_alpha_ramp",
    },
}


def _deep_merge_mapping(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _deep_merge_mapping(result[key], value)
        else:
            result[key] = value
    return result


def serving_visualization_config(visualization_config: Any | None = None) -> dict[str, Any]:
    """Return website map styling with publication DEM defaults filled in.

    Run metadata may contain older partial visualization configs. Merging here
    keeps website artifacts visually consistent while still allowing explicit
    per-run overrides for map alpha, DEM range, colormaps, or thresholds.
    """
    if visualization_config is None:
        result = deepcopy(DEFAULT_SERVING_VISUALIZATION_CONFIG)
    elif not isinstance(visualization_config, Mapping):
        result = deepcopy(DEFAULT_SERVING_VISUALIZATION_CONFIG)
    else:
        result = _deep_merge_mapping(DEFAULT_SERVING_VISUALIZATION_CONFIG, visualization_config)
    map_cfg = result.setdefault("map", {})
    terrain_tif = _resolve_terrain_tif_path(result)
    if terrain_tif is not None:
        map_cfg["terrain_tif"] = str(terrain_tif)
    return result


def _map_config_value(visualization_config: Mapping[str, Any], key: str) -> Any | None:
    map_cfg = visualization_config.get("map")
    if isinstance(map_cfg, Mapping):
        return map_cfg.get(key)
    return None


@lru_cache(maxsize=1)
def _discover_model_bundle_terrain_tifs() -> tuple[str, ...]:
    """Return likely external terrain rasters available inside the container.

    The production deployment mounts the model bundle read-only at
    ``/model_bundle``. Keeping discovery constrained to that small tree avoids
    expensive filesystem walks while allowing the lab deployment to use the
    same HEC-RAS DEM raster as the publication figures without requiring every
    run artifact to repeat the path in metadata.
    """
    root = Path(os.environ.get("FGN_MODEL_BUNDLE_ROOT", "/model_bundle")).expanduser()
    if not root.exists() or not root.is_dir():
        return ()
    patterns = (
        "**/Terrain_V2.DEM1m_V2.tif",
        "**/Terrain.DEM1m.tif",
        "**/*DEM1m*.tif",
        "**/*Terrain*.tif",
    )
    discovered: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        try:
            matches = sorted(root.glob(pattern))
        except Exception:
            continue
        for candidate in matches[:25]:
            if not candidate.is_file():
                continue
            text = str(candidate)
            if text in seen:
                continue
            discovered.append(text)
            seen.add(text)
    return tuple(discovered)


def _candidate_terrain_tif_paths(visualization_config: Mapping[str, Any]) -> list[Path]:
    raw_values = [
        _map_config_value(visualization_config, "terrain_tif"),
        os.environ.get("FGN_SERVING_TERRAIN_TIF"),
        os.environ.get("FGN_TERRAIN_TIF"),
        "/model_bundle/domain/Terrain_V2.DEM1m_V2.tif",
        "/model_bundle/domain/Terrain.DEM1m.tif",
        "/model_bundle/Terrain/Terrain_V2.DEM1m_V2.tif",
        "/model_bundle/Terrain/Terrain.DEM1m.tif",
        *_discover_model_bundle_terrain_tifs(),
    ]
    paths: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.append(Path("/model_bundle") / path)
        for candidate in candidates:
            text = str(candidate)
            if text in seen:
                continue
            paths.append(candidate)
            seen.add(text)
    return paths


def _resolve_terrain_tif_path(visualization_config: Mapping[str, Any]) -> Path | None:
    for path in _candidate_terrain_tif_paths(visualization_config):
        try:
            if path.exists() and path.is_file():
                return path
        except OSError:
            continue
    return None


def _candidate_dem_context_paths(visualization_config: Mapping[str, Any]) -> list[Path]:
    raw_values = [
        _map_config_value(visualization_config, "dem_context_path"),
        _map_config_value(visualization_config, "context_npz_path"),
        os.environ.get("FGN_DEM_CONTEXT_PATH"),
        "/model_bundle/domain/dem_context.npz",
    ]
    paths: list[Path] = []
    for raw in raw_values:
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        paths.append(path)
        if not path.is_absolute():
            paths.append(Path("/model_bundle") / path)
    return paths


def _load_dem_context(
    *,
    x: np.ndarray,
    y: np.ndarray,
    visualization_config: Mapping[str, Any],
    eval_render: Any,
) -> dict[str, Any] | None:
    """Load an optional pre-rendered DEM context image for serving maps.

    The deployed coastal geometry is an irregular subset of a rectangular
    UTM lattice. The publication figures showed a rectangular DEM context,
    while forecast values still live only on the 5,904 computational cells.
    This optional context image restores that publication-style background
    without changing the scientific arrays being plotted.
    """
    for path in _candidate_dem_context_paths(visualization_config):
        if not path.exists():
            continue
        try:
            data = np.load(path, allow_pickle=False)
            image = np.asarray(data["image"], dtype=np.float32)
            extent = tuple(float(v) for v in np.asarray(data["extent"], dtype=np.float64).reshape(4))
            req = eval_render._spatial_extent(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
            covers = extent[0] <= req[0] and extent[1] >= req[1] and extent[2] <= req[2] and extent[3] >= req[3]
            if not covers:
                continue
            metadata: dict[str, Any] = {
                "mode": "external_basemap",
                "map_mode": "dem_context",
                "source": str(path),
                "extent": list(extent),
            }
            if "metadata_json" in data.files:
                try:
                    metadata.update(json.loads(str(np.asarray(data["metadata_json"]).item())))
                except Exception:
                    pass
            options = eval_render._visualization_options(visualization_config)
            return {
                "mode": "external_basemap",
                "image": image,
                "extent": extent,
                "metadata": metadata,
                "options": options,
                "source_path": str(path),
            }
        except Exception:
            continue
    return None


def _cartographic_context(
    *,
    x: np.ndarray,
    y: np.ndarray,
    elevation_raw: np.ndarray | None,
    out_dir: str,
    visualization_config: Any | None,
    eval_render: Any,
) -> dict[str, Any]:
    config = serving_visualization_config(visualization_config)
    if _map_config_value(config, "terrain_tif"):
        terrain_context = eval_render._cartographic_context(
            x=x,
            y=y,
            elevation_raw=elevation_raw,
            out_dir=out_dir,
            visualization_config=config,
        )
        metadata = terrain_context.get("metadata", {}) if isinstance(terrain_context, Mapping) else {}
        if terrain_context.get("mode") == "external_basemap" and metadata.get("terrain_tif"):
            return terrain_context

    dem_context = _load_dem_context(x=x, y=y, visualization_config=config, eval_render=eval_render)
    if dem_context is not None:
        cache_dir = Path(out_dir) / "cartographic_context"
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            source_path = dem_context.get("source_path")
            if source_path:
                shutil.copyfile(source_path, cache_dir / "basemap_context.npz")
        except Exception:
            pass
        try:
            (cache_dir / "basemap_metadata.json").write_text(
                json.dumps(dem_context.get("metadata", {}), indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        return dem_context
    return eval_render._cartographic_context(
        x=x,
        y=y,
        elevation_raw=elevation_raw,
        out_dir=out_dir,
        visualization_config=config,
    )


def serving_cartographic_context(
    *,
    x: np.ndarray,
    y: np.ndarray,
    elevation_raw: np.ndarray | None,
    out_dir: str,
    visualization_config: Any | None,
    eval_render: Any,
) -> dict[str, Any]:
    """Return the serving map background context used by PNG and GIF products."""
    return _cartographic_context(
        x=x,
        y=y,
        elevation_raw=elevation_raw,
        out_dir=out_dir,
        visualization_config=visualization_config,
        eval_render=eval_render,
    )


def compute_rivanna_style_data_viewport(
    *,
    geometry_xy: np.ndarray,
    elevation_raw: np.ndarray | None = None,
    visualization_config: Any | None = None,
    dpi: int = 180,
) -> dict[str, float]:
    """Return the data-axes viewport inside a saved Rivanna-style PNG.

    The frontend click inspector receives browser image coordinates. The PNG
    contains a title, colorbar, padding, and a tightly cropped Matplotlib
    canvas, so mapping the full image rectangle to UTM coordinates selects the
    wrong cell. This helper mirrors ``write_rivanna_style_map_png`` and
    computes the data axes rectangle after Matplotlib layout and tight-crop
    padding, expressed as fractions of the saved image.
    """
    xy = np.asarray(geometry_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("geometry_xy must have shape [n_cells,2].")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from neuralop.flood.eval import render as eval_render

    x = xy[:, 0]
    y = xy[:, 1]
    fig_size = (7.2, 6.1)
    dpi = int(dpi)
    renderer_cfg = eval_render._build_spatial_renderer(x, y, figsize=fig_size, dpi=dpi, n_rows=1, n_cols=1)
    import tempfile

    tmp_context = tempfile.TemporaryDirectory(prefix="fgn-map-viewport-")
    try:
        context = _cartographic_context(
            x=x,
            y=y,
            elevation_raw=elevation_raw,
            out_dir=tmp_context.name,
            visualization_config=visualization_config,
            eval_render=eval_render,
        )
        fig, ax = plt.subplots(
            1,
            1,
            figsize=fig_size,
            dpi=dpi,
            constrained_layout=True,
            facecolor=SERVING_MAP_FACE_COLOR,
        )
        try:
            ax.set_facecolor(SERVING_MAP_FACE_COLOR)
            artist, _ = eval_render._plot_spatial_panel(
                ax=ax,
                x=x,
                y=y,
                arr=np.zeros(xy.shape[0], dtype=np.float64),
                renderer=renderer_cfg,
                context=context,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                is_wd_depth=False,
                zero_transparent=False,
                annotate=False,
            )
            ax.set_title("Map", loc="left", color="#102027", pad=8)
            ax.set_aspect("equal")
            ax.axis("off")
            cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label("Value", color="#102027", fontsize=8.5)
            cbar.ax.tick_params(labelsize=8, colors="#3f535b")
            cbar.outline.set_edgecolor("#aebfc7")

            fig.canvas.draw()
            canvas_renderer = fig.canvas.get_renderer()
            tight = fig.get_tightbbox(canvas_renderer).padded(0.08)
            ax_bbox = ax.get_window_extent(canvas_renderer)
            crop_x0 = tight.x0 * dpi
            crop_y0 = tight.y0 * dpi
            crop_w = max(tight.width * dpi, 1.0)
            crop_h = max(tight.height * dpi, 1.0)
            left = (ax_bbox.x0 - crop_x0) / crop_w
            right = (ax_bbox.x1 - crop_x0) / crop_w
            bottom_from_bottom = (ax_bbox.y0 - crop_y0) / crop_h
            top_from_bottom = (ax_bbox.y1 - crop_y0) / crop_h
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            return {
                "left": round(float(np.clip(left, 0.0, 1.0)), 6),
                "right": round(float(np.clip(right, 0.0, 1.0)), 6),
                "top": round(float(np.clip(1.0 - top_from_bottom, 0.0, 1.0)), 6),
                "bottom": round(float(np.clip(1.0 - bottom_from_bottom, 0.0, 1.0)), 6),
                "data_bounds": {
                    "x_min": float(min(x0, x1)),
                    "x_max": float(max(x0, x1)),
                    "y_min": float(min(y0, y1)),
                    "y_max": float(max(y0, y1)),
                },
            }
        finally:
            plt.close(fig)
    finally:
        tmp_context.cleanup()


def write_rivanna_style_map_png(
    *,
    values: np.ndarray,
    geometry_xy: np.ndarray,
    output_path: str | Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float = 0.0,
    vmax: float | None = None,
    elevation_raw: np.ndarray | None = None,
    is_wd_depth: bool = False,
    zero_transparent: bool = False,
    visualization_config: Any | None = None,
    annotate: bool = False,
    dpi: int = 320,
) -> Path:
    """Write one forecast-only map with the same spatial style as eval figures.

    ``dpi`` defaults to 320 (publication-quality static maps). Pass a smaller
    value (e.g. 140) when rendering many frames for the interactive scrubber,
    where moderate resolution keeps the per-run disk footprint manageable.
    """
    xy = np.asarray(geometry_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("geometry_xy must have shape [n_cells,2].")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != xy.shape[0]:
        raise ValueError(f"values length {arr.shape[0]} does not match geometry cells {xy.shape[0]}.")

    import matplotlib
    import matplotlib as mpl

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from neuralop.flood.eval import render as eval_render

    x = xy[:, 0]
    y = xy[:, 1]
    fig_size = (7.2, 6.1)
    dpi = int(dpi)
    # rc_context isolates rcParams mutations to this render so any corruption
    # (see render._plot_spatial_field for the matplotlib RcParams gotcha)
    # cannot leak across runs in the long-running Celery worker.
    with mpl.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        renderer = eval_render._build_spatial_renderer(x, y, figsize=fig_size, dpi=dpi, n_rows=1, n_cols=1)
        context = _cartographic_context(
            x=x,
            y=y,
            elevation_raw=elevation_raw,
            out_dir=str(Path(output_path).parent),
            visualization_config=visualization_config,
            eval_render=eval_render,
        )
        if vmax is None:
            vmax = eval_render._wd_spatial_vmax(arr) if is_wd_depth else eval_render._robust_nonnegative_vmax(arr)
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=fig_size,
            dpi=dpi,
            constrained_layout=True,
            facecolor=SERVING_MAP_FACE_COLOR,
        )
        try:
            ax.set_facecolor(SERVING_MAP_FACE_COLOR)
            artist, _ = eval_render._plot_spatial_panel(
                ax=ax,
                x=x,
                y=y,
                arr=arr,
                renderer=renderer,
                context=context,
                cmap=cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                is_wd_depth=is_wd_depth,
                zero_transparent=zero_transparent,
                annotate=annotate,
            )
            ax.set_title(title, loc="left", color="#102027", pad=8)
            ax.set_aspect("equal")
            ax.axis("off")
            cbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(colorbar_label)
            cbar.ax.tick_params(labelsize=8, colors="#3f535b")
            cbar.outline.set_edgecolor("#aebfc7")
            cbar.set_label(colorbar_label, color="#102027", fontsize=8.5)
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08, facecolor=fig.get_facecolor())
        finally:
            plt.close(fig)
        return out_path
