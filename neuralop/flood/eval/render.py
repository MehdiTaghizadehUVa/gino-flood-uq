"""Spatial and non-spatial rendering helpers for flood evaluation outputs."""

from __future__ import annotations

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.animation as animation
from matplotlib import colors as mcolors
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from neuralop.flood.eval.metrics import (
    _build_member_model_indices,
    _compute_csi,
    _median_positive_step,
    _nanmax_floor,
    _pit_rank_counts_from_reference,
    _variance_decomposition_by_model,
)
from neuralop.flood.eval.runtime import (
    ANIMATION_FPS,
    ANIMATION_INTERVAL_MS,
    CBAR_FRAC,
    CBAR_PAD,
    CSI_THRESHOLDS,
    MIN_EPS,
    PUBLICATION_TIMESTEPS,
    UQ_BOXPLOT_PNG,
    UQ_EXCEEDANCE_THRESHOLD,
    UQ_INTERVAL_COVERAGE_PNG,
    UQ_OVERALL_JSON,
    UQ_PIT_RANK_PNG,
    UQ_RELIABILITY_PNG,
    UQ_SPREAD_SKILL_PNG,
    UQ_VAR_DECOMP_PNG,
)

def _geometry_xy(geometry):
    """Extract x, y coordinates from geometry tensor or array."""
    arr = geometry.detach().cpu().numpy() if hasattr(geometry, "detach") else np.asarray(geometry)
    return arr[:, 0], arr[:, 1]


ROBUST_SPATIAL_COLOR_QUANTILE = 0.995
WD_SPATIAL_VMAX_CAP_METERS = 3.0
DEFAULT_VISUALIZATION_CRS = "EPSG:32618"
DEFAULT_VISUALIZATION_MAP_MODE = "3dep_hillshade"
DEFAULT_BASEMAP_PROVIDER_BY_MODE = {
    "topo": "Esri.WorldTopoMap",
    "imagery": "Esri.WorldImagery",
    "3dep_hillshade": "USGS.3DEP",
    "elevation_hillshade": "local_elevation",
    "dem_elevation": "local_elevation",
}
USGS_3DEP_HILLSHADE_SERVICE_URL = "https://basemap.nationalmap.gov/arcgis/rest/services/USGSShadedReliefOnly/MapServer"
BASEMAP_CACHE_DIRNAME = "cartographic_context"
BASEMAP_NPZ = "basemap_context.npz"
BASEMAP_METADATA_JSON = "basemap_metadata.json"
DEFAULT_FORECAST_HISTORY_STEPS = 3


def _cfg_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        return getattr(obj, key)
    except (AttributeError, KeyError, TypeError):
        return default


def _cfg_path(obj: Any, path: Tuple[str, ...], default: Any = None) -> Any:
    cur = obj
    for key in path:
        cur = _cfg_value(cur, key, None)
        if cur is None:
            return default
    return cur


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_map_mode(value: Any) -> str:
    mode = str(value or DEFAULT_VISUALIZATION_MAP_MODE).strip().lower()
    aliases = {
        "3dep": "3dep_hillshade",
        "usgs_3dep": "3dep_hillshade",
        "usgs_3dep_hillshade": "3dep_hillshade",
        "hillshade": "elevation_hillshade",
        "elevation": "elevation_hillshade",
        "satellite": "imagery",
        "aerial": "imagery",
        "topographic": "topo",
        "dem": "dem_elevation",
        "colored_dem": "dem_elevation",
        "hecras_dem": "dem_elevation",
    }
    mode = aliases.get(mode, mode)
    supported = set(DEFAULT_BASEMAP_PROVIDER_BY_MODE)
    if mode not in supported:
        raise ValueError(f"Unsupported visualization.map.mode={value!r}; expected one of {sorted(supported)}")
    return mode


def _as_quantile_pair(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    try:
        if value is None:
            return default
        lo, hi = value
        lo_f = float(lo)
        hi_f = float(hi)
        if not 0.0 <= lo_f < hi_f <= 1.0:
            return default
        return lo_f, hi_f
    except Exception:
        return default


def _visualization_options(visualization_config: Any = None) -> Dict[str, Any]:
    map_mode = _normalize_map_mode(_cfg_path(visualization_config, ("map", "mode"), DEFAULT_VISUALIZATION_MAP_MODE))
    default_provider = DEFAULT_BASEMAP_PROVIDER_BY_MODE[map_mode]
    default_alpha = 1.0 if map_mode == "dem_elevation" else (0.65 if map_mode == "3dep_hillshade" else 0.42)
    default_wd_alpha = 0.88 if map_mode == "dem_elevation" else 0.78
    default_wd_cmap = "cyan_depth" if map_mode == "dem_elevation" else "viridis"
    return {
        "map_enabled": _as_bool(_cfg_path(visualization_config, ("map", "enabled"), True), True),
        "mode": map_mode,
        "crs": str(_cfg_path(visualization_config, ("map", "crs"), DEFAULT_VISUALIZATION_CRS)),
        "provider": str(_cfg_path(visualization_config, ("map", "provider"), default_provider)),
        "fallback": str(_cfg_path(visualization_config, ("map", "fallback"), "elevation_hillshade")),
        "basemap_alpha": float(_cfg_path(visualization_config, ("map", "alpha"), default_alpha)),
        "overlay_alpha": float(_cfg_path(visualization_config, ("map", "overlay_alpha"), 0.84)),
        "wd_overlay_alpha": float(_cfg_path(visualization_config, ("wd", "overlay_alpha"), default_wd_alpha)),
        "wd_colormap": str(_cfg_path(visualization_config, ("wd", "colormap"), default_wd_cmap)),
        "show_wet_edge": _as_bool(_cfg_path(visualization_config, ("wd", "show_wet_edge"), True), True),
        "wet_edge_threshold_m": float(_cfg_path(visualization_config, ("wd", "wet_edge_threshold_m"), 0.05)),
        "diagnostic_zero_threshold": float(_cfg_path(visualization_config, ("diagnostics", "zero_threshold"), 1e-10)),
        "diagnostic_zero_fraction": float(_cfg_path(visualization_config, ("diagnostics", "zero_fraction"), 0.03)),
        "diagnostic_overlay_alpha": float(_cfg_path(visualization_config, ("diagnostics", "overlay_alpha"), 0.70)),
        "diagnostic_basemap_alpha": float(_cfg_path(visualization_config, ("diagnostics", "basemap_alpha"), 0.28)),
        "cache_scope": str(_cfg_path(visualization_config, ("map", "cache_scope"), "run_extent")),
        "export_size_px": int(_cfg_path(visualization_config, ("map", "export_size_px"), 1024)),
        "hillshade_cmap": str(_cfg_path(visualization_config, ("map", "hillshade_cmap"), _cfg_path(visualization_config, ("map", "colormap"), "copper"))),
        "hillshade_tint_strength": float(_cfg_path(visualization_config, ("map", "hillshade_tint_strength"), 0.45)),
        "dem_cmap": str(_cfg_path(visualization_config, ("map", "dem_cmap"), "hecras_dem")),
        "dem_quantiles": _as_quantile_pair(_cfg_path(visualization_config, ("map", "dem_quantiles"), None), (0.01, 0.99)),
        "write_gif": _as_bool(_cfg_path(visualization_config, ("output", "write_gif"), True), True),
        "write_mp4": _as_bool(_cfg_path(visualization_config, ("output", "write_mp4"), True), True),
        "initial_history_steps": int(_cfg_path(visualization_config, ("time", "initial_history_steps"), DEFAULT_FORECAST_HISTORY_STEPS)),
    }


def _to_numpy_1d_optional(values: Optional[Any]) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return arr if arr.size > 0 else None


def _spatial_extent(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.025) -> Tuple[float, float, float, float]:
    xmin, xmax = float(np.nanmin(x)), float(np.nanmax(x))
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    xpad = max((xmax - xmin) * pad_frac, 1.0)
    ypad = max((ymax - ymin) * pad_frac, 1.0)
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


def _resolve_contextily_provider(ctx: Any, provider_name: str) -> Any:
    provider: Any = ctx.providers
    for part in str(provider_name).split("."):
        if isinstance(provider, dict):
            provider = provider[part]
        else:
            provider = getattr(provider, part)
    return provider


def _is_usgs_3dep_provider(options: Dict[str, Any]) -> bool:
    provider_name = str(options.get("provider") or "").strip().lower()
    return options.get("mode") == "3dep_hillshade" or provider_name in {"usgs.3dep", "usgs.3dep_hillshade", "usgsshadedreliefonly", "3dep_hillshade"}


def _resolve_basemap_source(ctx: Any, options: Dict[str, Any]) -> Any:
    mode = options.get("mode", DEFAULT_VISUALIZATION_MAP_MODE)
    provider_name = str(options.get("provider") or DEFAULT_BASEMAP_PROVIDER_BY_MODE.get(mode, ""))
    if mode in {"elevation_hillshade", "dem_elevation"}:
        raise RuntimeError(f"visualization.map.mode={mode} uses local elevation, not external tiles")
    if _is_usgs_3dep_provider(options):
        return USGS_3DEP_HILLSHADE_SERVICE_URL
    if "://" in provider_name and all(token in provider_name for token in ("{x}", "{y}", "{z}")):
        return provider_name
    if ctx is None:
        raise RuntimeError(f"contextily is required for visualization.map.provider={provider_name!r}")
    return _resolve_contextily_provider(ctx, provider_name)


def _arcgis_export_url(service_url: str, bbox_mercator: Tuple[float, float, float, float], size_px: int) -> str:
    from urllib.parse import urlencode

    left, bottom, right, top = bbox_mercator
    size_px = max(256, min(int(size_px), 4096))
    params = {
        "bbox": f"{left},{bottom},{right},{top}",
        "bboxSR": "102100",
        "imageSR": "102100",
        "size": f"{size_px},{size_px}",
        "format": "png32",
        "transparent": "false",
        "f": "image",
    }
    return f"{service_url.rstrip('/')}/export?{urlencode(params)}"


def _fetch_arcgis_export_image(service_url: str, bbox_mercator: Tuple[float, float, float, float], size_px: int, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, str]:
    from io import BytesIO
    from urllib.request import Request, urlopen

    from matplotlib import image as mpimg

    url = _arcgis_export_url(service_url, bbox_mercator, size_px)
    request = Request(url, headers={"User-Agent": "neuraloperator-flood-eval/1.0"})
    with urlopen(request, timeout=60) as response:
        image = mpimg.imread(BytesIO(response.read()), format="png")
    opts = options or {}
    return _enhance_hillshade_rgb(
        image,
        cmap_name=str(opts.get("hillshade_cmap", "copper")),
        tint_strength=float(opts.get("hillshade_tint_strength", 0.45)),
    ), url


def _as_rgb01(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    if img.max(initial=0.0) > 1.5:
        img = img / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return np.clip(img[..., :3], 0.0, 1.0).astype(np.float32)


def _enhance_hillshade_rgb(image: np.ndarray, cmap_name: str = "copper", tint_strength: float = 0.45) -> np.ndarray:
    img = _as_rgb01(image)
    gray = np.dot(img[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    finite = gray[np.isfinite(gray)]
    if finite.size == 0:
        return img
    lo, hi = np.nanpercentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if np.isclose(lo, hi):
        return img
    stretched = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    neutral = np.repeat((0.50 + 0.48 * stretched)[:, :, None], 3, axis=2)
    try:
        tinted = plt.get_cmap(cmap_name)(stretched)[..., :3].astype(np.float32)
    except ValueError:
        warnings.warn(f"Unknown hillshade colormap {cmap_name!r}; using neutral grayscale hillshade")
        return neutral.astype(np.float32)
    tint_strength = float(np.clip(tint_strength, 0.0, 1.0))
    # Whiten the tint before blending so it reads as paper terrain, not a competing data layer.
    tinted_paper = 0.30 + 0.70 * tinted
    enhanced = (1.0 - tint_strength) * neutral + tint_strength * tinted_paper
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def _mute_basemap_rgb(image: np.ndarray) -> np.ndarray:
    img = _as_rgb01(image)
    gray = np.dot(img[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    muted = 0.42 * img[..., :3] + 0.58 * gray[..., None]
    muted = 0.82 * muted + 0.18
    return np.clip(muted, 0.0, 1.0).astype(np.float32)


def _try_build_external_basemap(*, x: np.ndarray, y: np.ndarray, out_dir: str, options: Dict[str, Any]) -> Dict[str, Any]:
    from pyproj import Transformer  # type: ignore
    from rasterio.transform import from_bounds  # type: ignore
    from rasterio.warp import Resampling, reproject  # type: ignore

    crs = options["crs"]
    xmin, xmax, ymin, ymax = _spatial_extent(x, y)
    transformer = Transformer.from_crs(crs, "EPSG:3857", always_xy=True)
    mx, my = transformer.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    mw, me = float(np.nanmin(mx)), float(np.nanmax(mx))
    ms, mn = float(np.nanmin(my)), float(np.nanmax(my))
    provider_request = None
    if _is_usgs_3dep_provider(options):
        provider = _resolve_basemap_source(None, options)
        merc_extent = (mw, me, ms, mn)
        image, provider_request = _fetch_arcgis_export_image(provider, (mw, ms, me, mn), int(options.get("export_size_px", 1024)), options)
    else:
        import contextily as ctx  # type: ignore

        provider = _resolve_basemap_source(ctx, options)
        image, merc_extent = ctx.bounds2img(mw, ms, me, mn, source=provider, ll=False)
        image = _mute_basemap_rgb(image)
    h, w = image.shape[:2]
    src_left, src_right, src_bottom, src_top = [float(v) for v in merc_extent]
    src_transform = from_bounds(src_left, src_bottom, src_right, src_top, w, h)
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, w, h)
    dst = np.zeros((3, h, w), dtype=np.float32)
    for band in range(3):
        reproject(
            source=image[:, :, band],
            destination=dst[band],
            src_transform=src_transform,
            src_crs="EPSG:3857",
            dst_transform=dst_transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
        )
    basemap = np.moveaxis(dst, 0, -1)
    cache_dir = Path(out_dir) / BASEMAP_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / BASEMAP_NPZ, image=basemap, extent=np.asarray([xmin, xmax, ymin, ymax]))
    metadata = {
        "mode": "external_basemap",
        "map_mode": options.get("mode", DEFAULT_VISUALIZATION_MAP_MODE),
        "crs": crs,
        "provider": options["provider"],
        "provider_source": str(provider),
        "provider_request": provider_request,
        "hillshade_cmap": options.get("hillshade_cmap"),
        "hillshade_tint_strength": options.get("hillshade_tint_strength"),
        "extent": [xmin, xmax, ymin, ymax],
        "source_crs": "EPSG:3857",
        "alpha": options["basemap_alpha"],
        "cache_scope": options.get("cache_scope", "run_extent"),
    }
    (cache_dir / BASEMAP_METADATA_JSON).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"mode": "external_basemap", "image": basemap, "extent": (xmin, xmax, ymin, ymax), "metadata": metadata}


def _cache_matches_options(metadata: Dict[str, Any], options: Optional[Dict[str, Any]]) -> bool:
    if options is None:
        return True
    if metadata.get("crs") not in {None, options.get("crs")}:
        return False
    if metadata.get("provider") not in {None, options.get("provider")}:
        return False
    cached_map_mode = metadata.get("map_mode")
    if cached_map_mode is not None and cached_map_mode != options.get("mode"):
        return False
    return True


def _load_cached_basemap(out_dir: str, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, options: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    cache_dir = Path(out_dir) / BASEMAP_CACHE_DIRNAME
    npz_path = cache_dir / BASEMAP_NPZ
    meta_path = cache_dir / BASEMAP_METADATA_JSON
    if not meta_path.exists():
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if not _cache_matches_options(metadata, options):
            return None
        if not npz_path.exists():
            mode = metadata.get("mode")
            if mode in {"elevation_hillshade", "dem_elevation", "none"}:
                return {"mode": mode, "metadata": metadata}
            return None
        data = np.load(npz_path)
        extent = tuple(float(v) for v in data["extent"].tolist())
        if x is not None and y is not None:
            req = _spatial_extent(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
            covers = extent[0] <= req[0] and extent[1] >= req[1] and extent[2] <= req[2] and extent[3] >= req[3]
            if not covers:
                return None
        return {
            "mode": metadata.get("mode", "external_basemap"),
            "image": data["image"],
            "extent": extent,
            "metadata": metadata,
        }
    except Exception:
        return None


def _metadata_for_fallback(out_dir: str, mode: str, options: Dict[str, Any], reason: str = "", extent: Optional[Tuple[float, float, float, float]] = None) -> None:
    cache_dir = Path(out_dir) / BASEMAP_CACHE_DIRNAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": mode,
        "map_mode": options.get("mode", DEFAULT_VISUALIZATION_MAP_MODE),
        "crs": options.get("crs"),
        "provider": options.get("provider"),
        "fallback": options.get("fallback"),
        "reason": reason,
        "extent": list(extent) if extent is not None else None,
        "cache_scope": options.get("cache_scope", "run_extent"),
    }
    (cache_dir / BASEMAP_METADATA_JSON).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _cartographic_context(*, x: np.ndarray, y: np.ndarray, elevation_raw: Optional[Any], out_dir: str, visualization_config: Any) -> Dict[str, Any]:
    options = _visualization_options(visualization_config)
    elevation = _to_numpy_1d_optional(elevation_raw)
    context: Dict[str, Any] = {"options": options, "elevation": elevation, "mode": "none"}
    if not options["map_enabled"]:
        return context
    cached = _load_cached_basemap(out_dir, x=x, y=y, options=options)
    if cached is not None:
        context.update(cached)
        return context
    if options["mode"] in {"elevation_hillshade", "dem_elevation"}:
        extent = _spatial_extent(x, y)
        if elevation is not None:
            context["mode"] = options["mode"]
            _metadata_for_fallback(out_dir, options["mode"], options, reason=f"mode={options['mode']}", extent=extent)
            return context
        _metadata_for_fallback(out_dir, "none", options, reason=f"mode={options['mode']} but no elevation_raw was provided", extent=extent)
        return context
    try:
        context.update(_try_build_external_basemap(x=x, y=y, out_dir=out_dir, options=options))
        return context
    except Exception as exc:
        extent = _spatial_extent(x, y)
        if options["fallback"].strip().lower() == "elevation_hillshade" and elevation is not None:
            context["mode"] = "elevation_hillshade"
            _metadata_for_fallback(out_dir, "elevation_hillshade", options, reason=str(exc), extent=extent)
            warnings.warn(f"Basemap tile rendering failed; using elevation hillshade fallback: {exc}")
            return context
        _metadata_for_fallback(out_dir, "none", options, reason=str(exc), extent=extent)
        warnings.warn(f"Basemap rendering disabled: {exc}")
        return context


def _dem_elevation_cmap(name: str) -> Any:
    if str(name).strip().lower() == "hecras_dem":
        return mcolors.LinearSegmentedColormap.from_list(
            "hecras_dem",
            [
                (0.00, "#06451f"),
                (0.18, "#0f7f38"),
                (0.36, "#8bbd25"),
                (0.54, "#d4d03d"),
                (0.68, "#d77b24"),
                (0.82, "#8b1f0e"),
                (1.00, "#c9c9c9"),
            ],
        )
    return plt.get_cmap(name)


def _cyan_depth_cmap() -> Any:
    return mcolors.LinearSegmentedColormap.from_list(
        "cyan_depth",
        [
            (0.00, "#dffcff"),
            (0.18, "#8ff7ff"),
            (0.40, "#19d9f2"),
            (0.68, "#0284c7"),
            (1.00, "#08306b"),
        ],
    )



def _rgba(hex_color: str, alpha: float) -> Tuple[float, float, float, float]:
    return (*mcolors.to_rgb(hex_color), float(alpha))


def _error_rose_cmap() -> Any:
    return mcolors.LinearSegmentedColormap.from_list(
        "error_rose",
        [
            (0.00, _rgba("#fff7f8", 0.00)),
            (0.14, _rgba("#fff1f2", 0.10)),
            (0.35, _rgba("#fecdd3", 0.36)),
            (0.62, _rgba("#fb7185", 0.70)),
            (0.84, _rgba("#be123c", 0.92)),
            (1.00, _rgba("#4c0519", 1.00)),
        ],
    )


def _spread_violet_cmap() -> Any:
    return mcolors.LinearSegmentedColormap.from_list(
        "spread_violet",
        [
            (0.00, _rgba("#faf7ff", 0.00)),
            (0.16, _rgba("#f5f3ff", 0.10)),
            (0.38, _rgba("#ddd6fe", 0.34)),
            (0.64, _rgba("#a78bfa", 0.68)),
            (0.84, _rgba("#6d28d9", 0.90)),
            (1.00, _rgba("#2e1065", 1.00)),
        ],
    )


def _crps_indigo_cmap() -> Any:
    return mcolors.LinearSegmentedColormap.from_list(
        "crps_indigo",
        [
            (0.00, _rgba("#f0fdff", 0.00)),
            (0.16, _rgba("#ecfeff", 0.10)),
            (0.38, _rgba("#a5f3fc", 0.34)),
            (0.62, _rgba("#38bdf8", 0.68)),
            (0.82, _rgba("#4f46e5", 0.90)),
            (1.00, _rgba("#1e1b4b", 1.00)),
        ],
    )


def _resolve_field_cmap(cmap: Any) -> Any:
    if isinstance(cmap, str):
        key = cmap.strip().lower()
        if key == "cyan_depth":
            return _cyan_depth_cmap()
        if key == "error_rose":
            return _error_rose_cmap()
        if key == "spread_violet":
            return _spread_violet_cmap()
        if key == "crps_indigo":
            return _crps_indigo_cmap()
    return cmap


def _diagnostic_background_context(context: Dict[str, Any]) -> Dict[str, Any]:
    options = dict(context.get("options", _visualization_options(None)))
    diagnostic_alpha = float(options.get("diagnostic_basemap_alpha", 0.28))
    options["basemap_alpha"] = min(float(options.get("basemap_alpha", diagnostic_alpha)), diagnostic_alpha)
    return {**context, "options": options}


def _draw_cartographic_background(ax: Any, x: np.ndarray, y: np.ndarray, context: Dict[str, Any], renderer: Dict[str, Any]) -> None:
    mode = context.get("mode")
    options = context.get("options", {})
    if mode == "external_basemap" and context.get("image") is not None:
        ax.imshow(context["image"], extent=context["extent"], origin="upper", alpha=float(options.get("basemap_alpha", 0.42)), zorder=0)
        return
    if mode in {"elevation_hillshade", "dem_elevation"} and context.get("elevation") is not None:
        elev = np.asarray(context["elevation"], dtype=np.float64)
        finite = elev[np.isfinite(elev)]
        if finite.size == 0:
            return
        qlo, qhi = options.get("dem_quantiles", (0.02, 0.98))
        if mode == "elevation_hillshade":
            qlo, qhi = 0.02, 0.98
        lo, hi = np.nanquantile(finite, [qlo, qhi])
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        plot_elev = np.asarray(elev, dtype=np.float64).copy()
        plot_elev[~np.isfinite(plot_elev)] = lo
        plot_elev = np.clip(plot_elev, lo, hi)
        if mode == "dem_elevation":
            cmap = _dem_elevation_cmap(str(options.get("dem_cmap", "hecras_dem")))
            alpha = float(options.get("basemap_alpha", 1.0))
        else:
            cmap = "Greys_r"
            alpha = 0.32
        _plot_spatial_field(ax=ax, x=x, y=y, arr=plot_elev, renderer=renderer, cmap=cmap, vmin=lo, vmax=hi, alpha=alpha, zorder=0)


def _nice_scale_length(x_range: float) -> float:
    target = max(float(x_range) * 0.22, 1.0)
    for length in (50, 100, 200, 500, 1000, 2000, 5000, 10000):
        if length >= target:
            return float(length)
    return float(target)


def _add_map_annotations(ax: Any, x: np.ndarray, y: np.ndarray) -> None:
    xmin, xmax, ymin, ymax = _spatial_extent(x, y, pad_frac=0.0)
    xr = max(xmax - xmin, MIN_EPS)
    yr = max(ymax - ymin, MIN_EPS)
    halo = [patheffects.Stroke(linewidth=2.4, foreground="white"), patheffects.Normal()]

    # North arrow: upper-left, away from colorbars and the lower diagnostics row.
    nx = xmin + 0.08 * xr
    ny = ymax - 0.18 * yr
    ax.annotate(
        "N",
        xy=(nx, ny + 0.09 * yr),
        xytext=(nx, ny),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#111827",
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#111827"},
        path_effects=halo,
        zorder=7,
    )

    # Scale bar: lower-right, inside the map extent with a right-aligned label.
    scale_len = min(_nice_scale_length(xr), 0.35 * xr)
    sx1 = xmax - 0.06 * xr
    sx0 = sx1 - scale_len
    sy0 = ymin + 0.07 * yr
    ax.plot([sx0, sx1], [sy0, sy0], color="#111827", linewidth=2.2, zorder=6)
    label = f"{scale_len/1000:.1f} km" if scale_len >= 1000 else f"{int(scale_len)} m"
    ax.text(
        sx1,
        sy0 + 0.025 * yr,
        label,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#111827",
        path_effects=halo,
        zorder=7,
    )


def _mask_wd_dry_for_overlay(arr: np.ndarray, threshold: float) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64).copy()
    data[~np.isfinite(data)] = np.nan
    data[data < float(threshold)] = np.nan
    return data


def _mask_near_zero_for_overlay(arr: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64).copy()
    data[~np.isfinite(data)] = np.nan
    data[np.abs(data) <= float(threshold)] = np.nan
    return data


def _diagnostic_zero_threshold(options: Dict[str, Any], vmax: float) -> float:
    absolute = float(options.get("diagnostic_zero_threshold", 1e-10))
    fraction = max(float(options.get("diagnostic_zero_fraction", 0.03)), 0.0)
    return max(absolute, fraction * max(float(vmax), MIN_EPS))


def _draw_wet_edge(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    arr: np.ndarray,
    renderer: Dict[str, Any],
    threshold: float,
) -> List[Any]:
    values = np.asarray(arr, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0 or float(np.nanmin(finite)) > threshold or float(np.nanmax(finite)) < threshold:
        return []
    try:
        if renderer["mode"] == "structured":
            grid = _field_to_structured_grid(values, renderer)
            xc = 0.5 * (renderer["x_edges"][:-1] + renderer["x_edges"][1:])
            yc = 0.5 * (renderer["y_edges"][:-1] + renderer["y_edges"][1:])
            contour = ax.contour(xc, yc, grid, levels=[threshold], colors="#082f49", linewidths=1.05, zorder=5)
        elif renderer["mode"] == "tri":
            contour = ax.tricontour(renderer["triangulation"], values, levels=[threshold], colors="#082f49", linewidths=1.05, zorder=5)
        else:
            contour = ax.tricontour(mtri.Triangulation(x, y), values, levels=[threshold], colors="#082f49", linewidths=1.05, zorder=5)
    except Exception:
        return []

    halo = [patheffects.Stroke(linewidth=2.5, foreground="white"), patheffects.Normal()]
    try:
        contour.set_path_effects(halo)
    except Exception:
        for artist in getattr(contour, "collections", []):
            artist.set_path_effects(halo)
    if hasattr(contour, "remove"):
        return [contour]
    return list(getattr(contour, "collections", []))


def _remove_artists(artists: List[Any]) -> None:
    while artists:
        artist = artists.pop()
        try:
            artist.remove()
        except (AttributeError, ValueError):
            pass


def _plot_spatial_panel(
    *,
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    arr: np.ndarray,
    renderer: Dict[str, Any],
    context: Dict[str, Any],
    cmap: str,
    vmin: float,
    vmax: float,
    norm: Optional[Any] = None,
    is_wd_depth: bool = False,
    annotate: bool = False,
    zero_transparent: bool = False,
    alpha_override: Optional[float] = None,
) -> Tuple[Any, List[Any]]:
    options = context.get("options", _visualization_options(None))
    background_context = _diagnostic_background_context(context) if zero_transparent else context
    _draw_cartographic_background(ax, x, y, background_context, renderer)
    if is_wd_depth:
        plot_arr = _mask_wd_dry_for_overlay(arr, options["wet_edge_threshold_m"])
        cmap = options.get("wd_colormap", cmap)
        alpha = options["wd_overlay_alpha"]
    elif zero_transparent:
        plot_arr = _mask_near_zero_for_overlay(arr, _diagnostic_zero_threshold(options, vmax))
        # Diagnostic colormaps carry their own alpha ramp so low values reveal the DEM.
        alpha = None
    else:
        plot_arr = arr
        alpha = options["overlay_alpha"]
    if alpha_override is not None:
        alpha = float(alpha_override)
    artist = _plot_spatial_field(ax=ax, x=x, y=y, arr=plot_arr, renderer=renderer, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, alpha=alpha, zorder=2)
    draw_edge = bool(is_wd_depth and options.get("show_wet_edge", True))
    edge_artists = _draw_wet_edge(ax, x, y, arr, renderer, options["wet_edge_threshold_m"]) if draw_edge else []
    if annotate:
        _add_map_annotations(ax, x, y)
    return artist, edge_artists


def _save_animation_outputs(ani: Any, base_path: str, options: Dict[str, Any]) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    if options.get("write_gif", True):
        gif_path = f"{base_path}.gif"
        ani.save(gif_path, writer="pillow", fps=ANIMATION_FPS)
        outputs["gif"] = gif_path
    if options.get("write_mp4", True):
        mp4_path = f"{base_path}.mp4"
        if animation.writers.is_available("ffmpeg"):
            ani.save(mp4_path, writer="ffmpeg", fps=ANIMATION_FPS, dpi=160)
            outputs["mp4"] = mp4_path
        else:
            outputs["mp4_skipped"] = "ffmpeg writer unavailable"
    return outputs


def _safe_linear_fit_and_corr(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Fit y=a*x+b only when spread-skill samples are numerically valid."""
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size <= 2:
        return None
    if np.ptp(x_arr) <= MIN_EPS or np.ptp(y_arr) <= MIN_EPS:
        return None
    try:
        slope, intercept = np.polyfit(x_arr, y_arr, deg=1)
        corr_matrix = np.corrcoef(x_arr, y_arr)
        corr = float(corr_matrix[0, 1])
    except (FloatingPointError, np.linalg.LinAlgError, ValueError):
        return None
    if not (np.isfinite(slope) and np.isfinite(intercept) and np.isfinite(corr)):
        return None
    return corr, float(slope), float(intercept)


def _finite_spatial_values(*arrays: np.ndarray) -> np.ndarray:
    vals = []
    for arr in arrays:
        if arr is None:
            continue
        arr_np = np.asarray(arr, dtype=np.float64)
        finite = arr_np[np.isfinite(arr_np)]
        if finite.size > 0:
            vals.append(finite)
    if not vals:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(vals, axis=0)


def _robust_nonnegative_vmax(
    *arrays: np.ndarray,
    quantile: float = ROBUST_SPATIAL_COLOR_QUANTILE,
) -> float:
    vals = _finite_spatial_values(*arrays)
    if vals.size == 0:
        return MIN_EPS
    vals = vals[vals >= 0.0]
    if vals.size == 0:
        return MIN_EPS
    hard_max = float(np.nanmax(vals))
    vmax = float(np.quantile(vals, quantile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = hard_max
    return max(min(vmax, hard_max), MIN_EPS)


def _robust_symmetric_abs_vmax(
    *arrays: np.ndarray,
    quantile: float = ROBUST_SPATIAL_COLOR_QUANTILE,
) -> float:
    vals = np.abs(_finite_spatial_values(*arrays))
    if vals.size == 0:
        return MIN_EPS
    hard_max = float(np.nanmax(vals))
    vmax = float(np.quantile(vals, quantile))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = hard_max
    return max(min(vmax, hard_max), MIN_EPS)


def _wd_spatial_vmax(*arrays: np.ndarray) -> float:
    vmax = _robust_nonnegative_vmax(*arrays)
    return max(min(vmax, WD_SPATIAL_VMAX_CAP_METERS), MIN_EPS)


def _channel_vmin_vmax_cmap(
    ch: str, gt: np.ndarray, pred: np.ndarray
) -> Tuple[float, float, str]:
    """Return (vmin, vmax, cmap) for a channel (wd vs velocity-style)."""
    if ch == "wd":
        vmax = _wd_spatial_vmax(gt, pred)
        return 0.0, vmax, "viridis"
    vmax = _robust_symmetric_abs_vmax(gt, pred)
    return -vmax, vmax, "coolwarm"

def _save_generic_rollout_visuals(
    geometry: torch.Tensor,
    pred_by_channel: Dict[str, np.ndarray],
    gt_by_channel: Dict[str, np.ndarray],
    target_variables: List[str],
    out_dir: str,
    run_id: str,
    dt_seconds: float,
    elevation_raw: Optional[Any] = None,
    visualization_config: Optional[Any] = None,
) -> None:
    """Write publication maps and a comparison GIF for generic target variables."""
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    x, y = _geometry_xy(geometry)
    cartographic_context = _cartographic_context(
        x=x,
        y=y,
        elevation_raw=elevation_raw,
        out_dir=out_dir,
        visualization_config=visualization_config,
    )
    rid = run_id or "unknown"
    n_steps = next(iter(gt_by_channel.values())).shape[0]
    steps = [s for s in PUBLICATION_TIMESTEPS if 0 <= s < n_steps] or [
        min(n_steps - 1, 0)
    ]
    n_rows = len(target_variables)

    for t in steps:
        fig, axs = plt.subplots(
            n_rows, 3, figsize=(18, 5 * n_rows), dpi=250, constrained_layout=True
        )
        axs = np.atleast_2d(axs)
        for r, ch in enumerate(target_variables):
            gt_t = gt_by_channel[ch][t]
            pred_t = pred_by_channel[ch][t]
            err_t = np.abs(pred_t - gt_t)
            vmin, vmax, cmap = _channel_vmin_vmax_cmap(ch, gt_t, pred_t)
            emax = _robust_nonnegative_vmax(err_t)
            panels = [
                (f"{ch.upper()} Ground Truth", gt_t, cmap, vmin, vmax, ch),
                (f"{ch.upper()} Prediction", pred_t, cmap, vmin, vmax, ch),
                (f"{ch.upper()} Abs Error", err_t, "error_rose", 0.0, emax, "error"),
            ]
            for c, (title, arr, pcmap, pvmin, pvmax, cblabel) in enumerate(panels):
                ax = axs[r, c]
                renderer_panel = _build_spatial_renderer(x, y, figsize=(18, 5 * n_rows), dpi=250, n_rows=n_rows, n_cols=3)
                sc, _ = _plot_spatial_panel(
                    ax=ax,
                    x=x,
                    y=y,
                    arr=arr,
                    renderer=renderer_panel,
                    context=cartographic_context,
                    cmap=pcmap,
                    vmin=pvmin,
                    vmax=pvmax,
                    is_wd_depth=(ch == "wd" and "Error" not in title),
                    zero_transparent=("Abs Error" in title),
                    annotate=(r == 0 and c == 0),
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")
                cb = fig.colorbar(sc, ax=ax, fraction=CBAR_FRAC, pad=CBAR_PAD)
                cb.set_label(cblabel)
        fig.savefig(
            os.path.join(fig_dir, f"rollout_{rid}_t{t}.png"),
            bbox_inches="tight", pad_inches=0.1
        )
        plt.close(fig)

    fig, axs = plt.subplots(
        n_rows, 2, figsize=(12, 4 * n_rows), constrained_layout=True
    )
    axs = np.atleast_2d(axs)
    scatters: List[Tuple[str, Any, Any, Dict[str, Any], List[Any], List[Any], Any, Any]] = []
    for r, ch in enumerate(target_variables):
        gt0 = gt_by_channel[ch][0]
        pred0 = pred_by_channel[ch][0]
        vmin, vmax, cmap = _channel_vmin_vmax_cmap(
            ch, gt_by_channel[ch], pred_by_channel[ch]
        )
        renderer_anim = _build_spatial_renderer(x, y, figsize=(12, 4 * n_rows), dpi=100, n_rows=n_rows, n_cols=2)
        s_gt, gt_edge_artists = _plot_spatial_panel(
            ax=axs[r, 0],
            x=x,
            y=y,
            arr=gt0,
            renderer=renderer_anim,
            context=cartographic_context,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            is_wd_depth=(ch == "wd"),
            annotate=False,
        )
        s_pr, pred_edge_artists = _plot_spatial_panel(
            ax=axs[r, 1],
            x=x,
            y=y,
            arr=pred0,
            renderer=renderer_anim,
            context=cartographic_context,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            is_wd_depth=(ch == "wd"),
        )
        axs[r, 0].set_title(f"{ch.upper()} Ground Truth")
        axs[r, 1].set_title(f"{ch.upper()} Prediction")
        axs[r, 0].axis("off")
        axs[r, 1].axis("off")
        fig.colorbar(s_gt, ax=axs[r, 0], fraction=CBAR_FRAC, pad=0.03)
        fig.colorbar(s_pr, ax=axs[r, 1], fraction=CBAR_FRAC, pad=0.03)
        scatters.append((ch, s_gt, s_pr, renderer_anim, gt_edge_artists, pred_edge_artists, axs[r, 0], axs[r, 1]))

    def _animate(frame_idx: int) -> List[Any]:
        time_hours = (frame_idx + 1) * dt_seconds / 3600.0
        fig.suptitle(
            f"Rollout Comparison (Run: {rid}) - Time: {time_hours:.2f} hrs",
            fontsize=16,
        )
        artists: List[Any] = []
        options = cartographic_context.get("options", _visualization_options(visualization_config))
        for ch, s_gt, s_pr, renderer_anim, gt_edges, pred_edges, ax_gt, ax_pr in scatters:
            gt_raw = gt_by_channel[ch][frame_idx]
            pred_raw = pred_by_channel[ch][frame_idx]
            gt_frame = gt_raw
            pred_frame = pred_raw
            if ch == "wd":
                gt_frame = _mask_wd_dry_for_overlay(gt_raw, options["wet_edge_threshold_m"])
                pred_frame = _mask_wd_dry_for_overlay(pred_raw, options["wet_edge_threshold_m"])
            _update_spatial_artist(s_gt, gt_frame, renderer_anim)
            _update_spatial_artist(s_pr, pred_frame, renderer_anim)
            if ch == "wd":
                _remove_artists(gt_edges)
                _remove_artists(pred_edges)
                if options.get("show_wet_edge", True):
                    gt_edges.extend(_draw_wet_edge(ax_gt, x, y, gt_raw, renderer_anim, options["wet_edge_threshold_m"]))
                    pred_edges.extend(_draw_wet_edge(ax_pr, x, y, pred_raw, renderer_anim, options["wet_edge_threshold_m"]))
            artists.extend([s_gt, s_pr])
            artists.extend(gt_edges)
            artists.extend(pred_edges)
        return artists

    ani = animation.FuncAnimation(
        fig, _animate, frames=n_steps, interval=ANIMATION_INTERVAL_MS, blit=False
    )
    _save_animation_outputs(
        ani,
        os.path.join(out_dir, f"rollout_{rid}"),
        cartographic_context.get("options", _visualization_options(visualization_config)),
    )
    plt.close(fig)

def _adaptive_marker_size(
    x: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[float, float],
    dpi: int,
    n_rows: int = 1,
    n_cols: int = 1,
    fill_factor: float = 1.20,
) -> float:
    """
    Choose square scatter marker area to visually fill gaps for point maps.

    Uses coordinate spacing + panel geometry, not only number of points.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_points = int(x.size)
    if n_points <= 1:
        return 12.0

    xr = max(float(np.nanmax(x) - np.nanmin(x)), MIN_EPS)
    yr = max(float(np.nanmax(y) - np.nanmin(y)), MIN_EPS)
    dx = _median_positive_step(x)
    dy = _median_positive_step(y)
    if dx is None or not np.isfinite(dx):
        dx = np.sqrt((xr * yr) / max(n_points, 1))
    if dy is None or not np.isfinite(dy):
        dy = np.sqrt((xr * yr) / max(n_points, 1))

    panel_w_in = 0.82 * float(figsize[0]) / max(int(n_cols), 1)
    panel_h_in = 0.78 * float(figsize[1]) / max(int(n_rows), 1)
    px_per_x = panel_w_in * float(dpi) / xr
    px_per_y = panel_h_in * float(dpi) / yr
    step_x_px = max(dx * px_per_x, 0.0)
    step_y_px = max(dy * px_per_y, 0.0)

    # Use the larger axis spacing to prevent visible striping.
    side_px = fill_factor * max(step_x_px, step_y_px, 1.0)
    side_pt = side_px * 72.0 / float(dpi)
    marker_area_pt2 = side_pt ** 2
    return float(np.clip(marker_area_pt2, 4.0, 180.0))


def _scatter_style(marker_size: float) -> Dict[str, Any]:
    """Consistent style for continuous-looking rasterized point maps."""
    return {
        "s": marker_size,
        "marker": "s",
        "linewidths": 0,
        "edgecolors": "none",
        "antialiaseds": False,
        "rasterized": True,
    }


def _compute_cell_edges(centers: np.ndarray) -> np.ndarray:
    """Compute cell edges from sorted cell centers."""
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5], dtype=np.float64)
    mids = 0.5 * (c[:-1] + c[1:])
    left = c[0] - (mids[0] - c[0])
    right = c[-1] + (c[-1] - mids[-1])
    return np.concatenate(([left], mids, [right])).astype(np.float64)


def _build_structured_renderer(
    x: np.ndarray, y: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Recover a structured 2D grid from point coordinates for seam-free rendering.

    Returns None when geometry cannot be mapped cleanly to a unique rectilinear grid.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_points = int(x.size)
    if n_points <= 1:
        return None

    ux = np.unique(x)
    uy = np.unique(y)
    nx = int(ux.size)
    ny = int(uy.size)
    if nx < 2 or ny < 2:
        return None
    # Structured renderer is only appropriate when grid occupancy is high.
    coverage = float(n_points) / float(nx * ny)
    if coverage < 0.95:
        return None

    xr = max(float(np.nanmax(x) - np.nanmin(x)), 1.0)
    yr = max(float(np.nanmax(y) - np.nanmin(y)), 1.0)
    atol_x = 1e-10 * xr
    atol_y = 1e-10 * yr

    ix = np.searchsorted(ux, x)
    iy = np.searchsorted(uy, y)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    if not np.all(in_bounds):
        return None
    if not np.all(np.isclose(x, ux[ix], rtol=0.0, atol=atol_x)):
        return None
    if not np.all(np.isclose(y, uy[iy], rtol=0.0, atol=atol_y)):
        return None

    linear = iy * nx + ix
    if np.unique(linear).size != n_points:
        # Duplicate points in same cell -> ambiguous grid assignment.
        return None

    flat_to_point = np.full(nx * ny, -1, dtype=np.int64)
    flat_to_point[linear] = np.arange(n_points, dtype=np.int64)
    return {
        "mode": "structured",
        "nx": nx,
        "ny": ny,
        "flat_to_point": flat_to_point,
        "x_edges": _compute_cell_edges(ux),
        "y_edges": _compute_cell_edges(uy),
    }


def _build_triangulation_renderer(x: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
    """Build point-only triangulation renderer with long-edge masking."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3:
        return None
    tri = mtri.Triangulation(x, y)
    if tri.triangles.size == 0:
        return None

    tris = tri.triangles
    dx = _median_positive_step(x)
    dy = _median_positive_step(y)
    sx = dx if dx is not None and np.isfinite(dx) and dx > 0.0 else 1.0
    sy = dy if dy is not None and np.isfinite(dy) and dy > 0.0 else 1.0
    xn = x / sx
    yn = y / sy

    x0, y0 = xn[tris[:, 0]], yn[tris[:, 0]]
    x1, y1 = xn[tris[:, 1]], yn[tris[:, 1]]
    x2, y2 = xn[tris[:, 2]], yn[tris[:, 2]]
    l01 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    l12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l20 = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)
    lmax = np.maximum(l01, np.maximum(l12, l20))
    finite = np.isfinite(lmax) & (lmax > 0.0)
    if np.any(finite):
        med = float(np.median(lmax[finite]))
        cutoff = 2.5 * med
        tri.set_mask(lmax > cutoff)
    return {"mode": "tri", "triangulation": tri}


def _build_spatial_renderer(
    x: np.ndarray,
    y: np.ndarray,
    figsize: Tuple[float, float],
    dpi: int,
    n_rows: int,
    n_cols: int,
) -> Dict[str, Any]:
    """Build renderer config: structured grid first, scatter fallback."""
    structured = _build_structured_renderer(x, y)
    if structured is not None:
        return structured
    tri = _build_triangulation_renderer(x, y)
    if tri is not None:
        return tri
    marker_size = _adaptive_marker_size(
        x, y, figsize=figsize, dpi=dpi, n_rows=n_rows, n_cols=n_cols, fill_factor=1.20
    )
    return {
        "mode": "scatter",
        "marker_size": marker_size,
    }


def _field_to_structured_grid(arr: np.ndarray, renderer: Dict[str, Any]) -> np.ndarray:
    """Map 1D point values to structured 2D grid with NaN for absent cells."""
    flat_to_point = renderer["flat_to_point"]
    flat = np.full(flat_to_point.shape[0], np.nan, dtype=np.float64)
    valid = flat_to_point >= 0
    flat[valid] = np.asarray(arr, dtype=np.float64)[flat_to_point[valid]]
    return flat.reshape(int(renderer["ny"]), int(renderer["nx"]))


def _plot_spatial_field(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    arr: np.ndarray,
    renderer: Dict[str, Any],
    cmap: str,
    vmin: float,
    vmax: float,
    norm: Optional[Any] = None,
    alpha: Optional[float] = None,
    zorder: Optional[float] = None,
) -> Any:
    """Plot one spatial field with best available renderer."""
    kwargs: Dict[str, Any] = {"cmap": _resolve_field_cmap(cmap)}
    if alpha is not None:
        kwargs["alpha"] = float(alpha)
    if zorder is not None:
        kwargs["zorder"] = zorder
    if norm is not None:
        kwargs["norm"] = norm
    else:
        kwargs["vmin"] = vmin
        kwargs["vmax"] = vmax
    if renderer["mode"] == "structured":
        grid = np.ma.masked_invalid(_field_to_structured_grid(arr, renderer))
        return ax.pcolormesh(
            renderer["x_edges"],
            renderer["y_edges"],
            grid,
            shading="flat",
            antialiased=False,
            rasterized=True,
            **kwargs,
        )
    if renderer["mode"] == "tri":
        return ax.tripcolor(
            renderer["triangulation"],
            np.ma.masked_invalid(np.asarray(arr, dtype=np.float64)),
            shading="gouraud",
            rasterized=True,
            **kwargs,
        )
    return ax.scatter(
        x,
        y,
        c=np.ma.masked_invalid(np.asarray(arr, dtype=np.float64)),
        **_scatter_style(float(renderer["marker_size"])),
        **kwargs,
    )


def _update_spatial_artist(artist: Any, arr: np.ndarray, renderer: Dict[str, Any]) -> None:
    """Update an existing spatial artist for animation frame."""
    if renderer["mode"] == "structured":
        grid = np.ma.masked_invalid(_field_to_structured_grid(arr, renderer))
        artist.set_array(grid.ravel())
        return
    arr_masked = np.ma.masked_invalid(np.asarray(arr, dtype=np.float64))
    if renderer["mode"] == "tri":
        artist.set_array(arr_masked)
        return
    artist.set_array(arr_masked)


def _save_nonspatial_uq_diagnostics(
    out_dir: str,
    time_hours: np.ndarray,
    metrics: Dict[str, np.ndarray],
    reliability_bins: Dict[str, np.ndarray],
    pit_hist_counts: np.ndarray,
    pit_edges: np.ndarray,
    rank_hist_counts: np.ndarray,
    spread_skill_samples: np.ndarray,
    interval_coverage: Dict[float, np.ndarray],
    interval_width: Dict[float, np.ndarray],
    wasserstein_wd: Optional[np.ndarray],
    logger: logging.Logger,
) -> None:
    """Write non-spatial UQ figures and overall metric summary for publication use."""
    os.makedirs(out_dir, exist_ok=True)

    overall: Dict[str, Any] = {}
    for key, arr in metrics.items():
        overall[f"{key}_overall_mean"] = float(np.mean(arr))
        overall[f"{key}_overall_std"] = float(np.std(arr))
        overall[f"{key}_leadtime_mean_last"] = float(np.mean(arr[:, -1]))

    if wasserstein_wd is not None:
        overall["wasserstein_wd_overall_mean"] = float(np.mean(wasserstein_wd))
        overall["wasserstein_wd_overall_std"] = float(np.std(wasserstein_wd))

    if reliability_bins.get("count", np.array([])).sum() > 0:
        count = reliability_bins["count"]
        mean_pred = reliability_bins["mean_pred"]
        mean_obs = reliability_bins["mean_obs"]
        rel_mask = count > 0
        ece = float(
            np.sum(count[rel_mask] * np.abs(mean_pred[rel_mask] - mean_obs[rel_mask]))
            / np.sum(count[rel_mask])
        )
        overall["wd_exceed_reliability_ece"] = ece
        if "brier_overall" in reliability_bins:
            overall["wd_exceed_brier_overall"] = float(reliability_bins["brier_overall"])
        elif "all_pred" in reliability_bins and "all_obs" in reliability_bins:
            overall["wd_exceed_brier_overall"] = float(
                np.mean((reliability_bins["all_pred"] - reliability_bins["all_obs"]) ** 2)
            )

    if pit_hist_counts.sum() > 0:
        pit_pdf = pit_hist_counts / pit_hist_counts.sum()
        uniform = np.full_like(pit_pdf, 1.0 / len(pit_pdf))
        overall["pit_l1_distance"] = float(np.sum(np.abs(pit_pdf - uniform)))

    if rank_hist_counts.sum() > 0:
        rank_pdf = rank_hist_counts / rank_hist_counts.sum()
        uniform_rank = np.full_like(rank_pdf, 1.0 / len(rank_pdf))
        overall["rank_hist_l1_distance"] = float(np.sum(np.abs(rank_pdf - uniform_rank)))

    if spread_skill_samples.size > 0:
        fit = _safe_linear_fit_and_corr(spread_skill_samples[:, 0], spread_skill_samples[:, 1])
        if fit is not None:
            corr, slope, intercept = fit
            overall["spread_skill_corr"] = corr
            overall["spread_skill_slope"] = slope
            overall["spread_skill_intercept"] = intercept

    json_path = os.path.join(out_dir, UQ_OVERALL_JSON)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, sort_keys=True)
    logger.info("Saved overall UQ metrics to %s", json_path)
    for key in sorted(overall.keys()):
        if key.endswith("_overall_mean") or key in {
            "wd_exceed_reliability_ece",
            "pit_l1_distance",
            "rank_hist_l1_distance",
            "spread_skill_corr",
            "spread_skill_slope",
        }:
            logger.info("Overall UQ metric %s=%.6e", key, overall[key])

    # Reliability diagram + bin counts
    if reliability_bins.get("count", np.array([])).sum() > 0:
        centers = reliability_bins["centers"]
        count = reliability_bins["count"]
        mean_pred = reliability_bins["mean_pred"]
        mean_obs = reliability_bins["mean_obs"]
        mask = count > 0
        fig, axs = plt.subplots(2, 1, figsize=(8.2, 7.0), dpi=280, constrained_layout=True)
        ax = axs[0]
        ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0, label="Perfect calibration")
        ax.plot(mean_pred[mask], mean_obs[mask], "o-", color="#1f77b4", linewidth=1.6, markersize=4, label="Model")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Forecast probability")
        ax.set_ylabel("Empirical probability")
        ax.set_title(f"Reliability: P(wd > {UQ_EXCEEDANCE_THRESHOLD:.2f})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)
        axs[1].bar(centers, count, width=(centers[1] - centers[0]) * 0.9, color="#4c78a8")
        axs[1].set_xlabel("Forecast probability bin")
        axs[1].set_ylabel("Sample count")
        axs[1].set_title("Forecast probability histogram")
        axs[1].grid(True, axis="y", alpha=0.3)
        fig.savefig(os.path.join(out_dir, UQ_RELIABILITY_PNG), bbox_inches="tight")
        plt.close(fig)

    # PIT + rank histogram
    if pit_hist_counts.sum() > 0 and rank_hist_counts.sum() > 0:
        pit_centers = 0.5 * (pit_edges[:-1] + pit_edges[1:])
        pit_pdf = pit_hist_counts / max(pit_hist_counts.sum(), 1.0)
        rank_idx = np.arange(rank_hist_counts.size)
        rank_pdf = rank_hist_counts / max(rank_hist_counts.sum(), 1.0)
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.6), dpi=280, constrained_layout=True)
        axs[0].bar(pit_centers, pit_pdf, width=(pit_edges[1] - pit_edges[0]) * 0.9, color="#59a14f", alpha=0.9)
        axs[0].axhline(1.0 / len(pit_pdf), linestyle="--", color="gray", linewidth=1.1)
        axs[0].set_title("PIT histogram")
        axs[0].set_xlabel("PIT value")
        axs[0].set_ylabel("Density")
        axs[0].grid(True, axis="y", alpha=0.3)
        axs[1].bar(rank_idx, rank_pdf, color="#f28e2b", alpha=0.9)
        axs[1].axhline(1.0 / len(rank_pdf), linestyle="--", color="gray", linewidth=1.1)
        axs[1].set_title("Rank histogram")
        axs[1].set_xlabel("Rank bin")
        axs[1].set_ylabel("Density")
        axs[1].grid(True, axis="y", alpha=0.3)
        fig.savefig(os.path.join(out_dir, UQ_PIT_RANK_PNG), bbox_inches="tight")
        plt.close(fig)

    # Spread-skill scatter
    if spread_skill_samples.size > 0:
        x = spread_skill_samples[:, 0]
        y = spread_skill_samples[:, 1]
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size > 0:
            vmax = max(_nanmax_floor(np.quantile(x, 0.995)), _nanmax_floor(np.quantile(y, 0.995)))
            fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.6), dpi=280, constrained_layout=True)
            hb = ax.hexbin(x, y, gridsize=48, mincnt=1, bins="log", cmap="viridis")
            cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.03)
            cb.set_label("log10(count)")
            ax.plot([0, vmax], [0, vmax], "--", color="white", linewidth=1.2, label="Ideal y=x")
            fit = _safe_linear_fit_and_corr(x, y)
            if fit is not None:
                corr, slope, intercept = fit
                xx = np.linspace(0.0, vmax, 100)
                ax.plot(xx, slope * xx + intercept, color="#d62728", linewidth=1.4, label="Fit")
                ax.text(
                    0.02,
                    0.95,
                    f"corr={corr:.3f}\nslope={slope:.3f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )
            ax.set_xlim(0.0, vmax)
            ax.set_ylim(0.0, vmax)
            ax.set_xlabel("Forecast spread (std)")
            ax.set_ylabel("Absolute mean error")
            ax.set_title("Spread-skill relationship (WD)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", fontsize=8)
            fig.savefig(os.path.join(out_dir, UQ_SPREAD_SKILL_PNG), bbox_inches="tight")
            plt.close(fig)

    # Interval coverage + sharpness
    if interval_coverage:
        alphas = sorted(interval_coverage.keys())
        fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.4), dpi=280, constrained_layout=True)
        for a in alphas:
            cov = interval_coverage[a]
            width = interval_width[a]
            cov_mean = np.mean(cov, axis=0)
            cov_std = np.std(cov, axis=0)
            width_mean = np.mean(width, axis=0)
            axs[0].plot(time_hours, cov_mean, linewidth=1.3, label=f"{int(a*100)}% interval")
            axs[0].fill_between(time_hours, cov_mean - cov_std, cov_mean + cov_std, alpha=0.12)
            axs[0].axhline(a, linestyle="--", linewidth=0.9, alpha=0.6)
            axs[1].plot(time_hours, width_mean, linewidth=1.3, label=f"{int(a*100)}% interval")
        axs[0].set_title("Empirical coverage vs lead time")
        axs[0].set_xlabel("Lead time (hour)")
        axs[0].set_ylabel("Coverage")
        axs[0].set_ylim(0.0, 1.0)
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=8, ncol=2)
        axs[1].set_title("Prediction interval width (sharpness)")
        axs[1].set_xlabel("Lead time (hour)")
        axs[1].set_ylabel("Width")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=8, ncol=2)
        fig.savefig(os.path.join(out_dir, UQ_INTERVAL_COVERAGE_PNG), bbox_inches="tight")
        plt.close(fig)

    # Predictive variance decomposition (epistemic vs stochastic).
    if (
        "within_var_wd" in metrics
        and "between_var_wd" in metrics
        and "between_frac_wd" in metrics
    ):
        within = metrics["within_var_wd"]
        between = metrics["between_var_wd"]
        total = metrics.get("total_var_wd", within + between)
        frac = metrics["between_frac_wd"]
        ratio = metrics.get("between_to_within_wd", None)
        within_m = np.mean(within, axis=0)
        between_m = np.mean(between, axis=0)
        total_m = np.mean(total, axis=0)
        frac_m = np.mean(frac, axis=0)

        fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=280, constrained_layout=True)
        ax0, ax1 = axs
        ax0.plot(time_hours, total_m, linewidth=1.6, color="#111111", label="Total variance")
        ax0.plot(time_hours, within_m, linewidth=1.4, color="#1f77b4", label="Within-model (noise)")
        ax0.plot(time_hours, between_m, linewidth=1.4, color="#d62728", label="Between-model")
        ax0.set_yscale("log")
        ax0.set_xlabel("Lead time (hour)")
        ax0.set_ylabel("Variance (log scale)")
        ax0.set_title("Variance decomposition (WD)")
        ax0.grid(True, alpha=0.3)
        ax0.legend(fontsize=8, loc="upper left")

        ax1.plot(time_hours, frac_m, linewidth=1.6, color="#2ca02c", label="Between / total")
        ax1.axhline(0.5, linestyle="--", color="gray", linewidth=0.9, alpha=0.7)
        if ratio is not None:
            ratio_m = np.mean(ratio, axis=0)
            ax1_t = ax1.twinx()
            ax1_t.plot(time_hours, ratio_m, linewidth=1.2, color="#9467bd", label="Between / within")
            ax1_t.set_ylabel("Variance ratio")
            ax1_t.grid(False)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_t.get_legend_handles_labels()
            ax1_t.legend(lines + lines2, labels + labels2, fontsize=8, loc="upper right")
        else:
            ax1.legend(fontsize=8, loc="upper right")
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Lead time (hour)")
        ax1.set_ylabel("Fraction")
        ax1.set_title("Epistemic share and dominance")
        ax1.grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, UQ_VAR_DECOMP_PNG), bbox_inches="tight")
        plt.close(fig)

    # Per-hydrograph metric boxplots (time-averaged)
    small_box_data: List[np.ndarray] = []
    small_box_labels: List[str] = []
    for key in ["rmse_wd", "crps_wd", "gaussian_nll_wd", "brier_wd_exceed", "wasserstein_wd"]:
        if key in metrics:
            small_box_data.append(np.mean(metrics[key], axis=1))
            small_box_labels.append(key)
    ratio_data = (
        np.mean(metrics["spread_ratio_wd"], axis=1)
        if "spread_ratio_wd" in metrics
        else None
    )
    if small_box_data or ratio_data is not None:
        fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=280, constrained_layout=True)
        ax_small, ax_ratio = axs
        if small_box_data:
            ax_small.boxplot(
                small_box_data,
                labels=small_box_labels,
                showfliers=False,
                whis=(5, 95),
            )
            ax_small.set_yscale("log")
            ax_small.set_title("Error/score metrics (log scale)")
            ax_small.set_ylabel("Metric value")
            ax_small.grid(True, axis="y", alpha=0.3)
            ax_small.tick_params(axis="x", rotation=18)
        else:
            ax_small.set_visible(False)

        if ratio_data is not None:
            ax_ratio.boxplot(
                [ratio_data],
                labels=["spread_ratio_wd"],
                showfliers=False,
                whis=(5, 95),
            )
            ax_ratio.axhline(1.0, linestyle="--", color="gray", linewidth=1.0, alpha=0.8)
            ax_ratio.set_title("Dispersion ratio")
            ax_ratio.set_ylabel("Predicted spread / GT spread")
            lo = np.quantile(ratio_data, 0.01)
            hi = np.quantile(ratio_data, 0.99)
            margin = 0.10 * max(hi - lo, 0.1)
            ax_ratio.set_ylim(max(0.0, lo - margin), hi + margin)
            ax_ratio.grid(True, axis="y", alpha=0.3)
        else:
            ax_ratio.set_visible(False)

        fig.suptitle("Per-hydrograph time-mean UQ metrics", fontsize=12.5)
        fig.savefig(os.path.join(out_dir, UQ_BOXPLOT_PNG), bbox_inches="tight")
        plt.close(fig)


def _coerce_boundary_series_raw(boundary_series_raw: Optional[Any]) -> Optional[np.ndarray]:
    if boundary_series_raw is None:
        return None
    if hasattr(boundary_series_raw, "detach"):
        arr = boundary_series_raw.detach().cpu().numpy()
    else:
        arr = np.asarray(boundary_series_raw)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        return None
    return arr


def _boundary_channel_names(boundary_channel_names: Optional[List[str]], n_channels: int) -> List[str]:
    names = [str(name).strip() for name in (boundary_channel_names or [])]
    out = []
    for idx in range(n_channels):
        if idx < len(names) and names[idx]:
            out.append(names[idx])
        elif n_channels == 1:
            out.append("inflow")
        else:
            out.append(f"boundary_{idx}")
    return out


def _find_boundary_channel(names: List[str], *needles: str) -> Optional[int]:
    lowered = [name.lower() for name in names]
    for idx, name in enumerate(lowered):
        if any(needle in name for needle in needles):
            return idx
    return None


def _boundary_plot_kind(name: str) -> str:
    lower = str(name).lower()
    if "precip" in lower or "rain" in lower:
        return "bar"
    return "line"


def _boundary_display_label(name: str) -> str:
    lower = str(name).lower()
    if "stage" in lower:
        return "Stage"
    if "precip" in lower or "rain" in lower:
        return "Precipitation"
    if lower in {"inflow", "flow", "hydrograph", "boundary"}:
        return "Inflow / flow"
    return str(name).replace("_", " ").title()


def _diagnostic_boundary_panels(boundary_channel_names: Optional[List[str]], n_channels: int) -> List[Tuple[int, str]]:
    names = _boundary_channel_names(boundary_channel_names, n_channels)
    stage_idx = _find_boundary_channel(names, "stage")
    precip_idx = _find_boundary_channel(names, "precip", "rain")
    if stage_idx is not None and precip_idx is not None:
        return [(stage_idx, "line"), (precip_idx, "bar")]
    if n_channels <= 0:
        return []
    if n_channels == 1:
        return [(0, _boundary_plot_kind(names[0]))]
    return [(idx, _boundary_plot_kind(names[idx])) for idx in range(min(n_channels, 2))]


def _diagnostic_ylim(values: np.ndarray, *, force_zero_bottom: bool = False) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if force_zero_bottom:
        vmin = min(0.0, vmin)
    if np.isclose(vmin, vmax):
        pad = max(abs(vmax) * 0.05, 1.0 if force_zero_bottom else 0.1)
    else:
        pad = 0.08 * (vmax - vmin)
    return (vmin - pad, vmax + pad)


def _forecast_horizon_hours(
    n_steps: int,
    dt_seconds: float,
    *,
    initial_history_steps: int = DEFAULT_FORECAST_HISTORY_STEPS,
) -> np.ndarray:
    """Display rollout time relative to the forecast horizon, not raw spinup index."""
    n = max(int(n_steps), 0)
    history = max(int(initial_history_steps), 0)
    return (history + np.arange(n, dtype=np.float64)) * float(dt_seconds) / 3600.0


def _visible_rollout_boundary_series(series: np.ndarray, rollout_start_index: int) -> np.ndarray:
    start = max(int(rollout_start_index), 0)
    if start >= int(series.shape[0]):
        return series[0:0]
    return series[start:]


def _make_rollout_diagnostic_axes(
    fig: Any,
    gs: Any,
    *,
    boundary_series_raw: Optional[Any],
    boundary_channel_names: Optional[List[str]],
) -> Dict[str, Any]:
    series = _coerce_boundary_series_raw(boundary_series_raw)
    n_channels = int(series.shape[1]) if series is not None else 0
    names = _boundary_channel_names(boundary_channel_names, n_channels)
    panels = _diagnostic_boundary_panels(names, n_channels)
    boundary_axes: List[Tuple[Any, int, str]] = []
    if len(panels) == 1:
        boundary_axes.append((fig.add_subplot(gs[2, 0:2]), panels[0][0], panels[0][1]))
        rel_ax = fig.add_subplot(gs[2, 2])
    elif len(panels) >= 2:
        boundary_axes.append((fig.add_subplot(gs[2, 0]), panels[0][0], panels[0][1]))
        boundary_axes.append((fig.add_subplot(gs[2, 1]), panels[1][0], panels[1][1]))
        rel_ax = fig.add_subplot(gs[2, 2])
    else:
        rel_ax = fig.add_subplot(gs[2, :])
    return {
        "series": series,
        "names": names,
        "boundary_axes": boundary_axes,
        "relative_l2_axis": rel_ax,
    }


def _draw_boundary_diagnostic_axis(
    ax: Any,
    *,
    time_hours: np.ndarray,
    values: np.ndarray,
    name: str,
    kind: str,
    current_index: int,
) -> None:
    ax.clear()
    values = np.asarray(values, dtype=np.float64)
    current_index = int(np.clip(current_index, 0, max(values.size - 1, 0)))
    label = _boundary_display_label(name)
    color = "#2563eb" if kind != "bar" else "#0f766e"
    if kind == "bar":
        width = 0.8 * float(np.median(np.diff(time_hours))) if time_hours.size > 1 else 0.2
        width = max(width, 0.02)
        ax.bar(time_hours, values, width=width, color=color, alpha=0.18, linewidth=0)
        ax.bar(time_hours[: current_index + 1], values[: current_index + 1], width=width, color=color, alpha=0.85, linewidth=0)
        force_zero = True
    else:
        ax.plot(time_hours, values, color=color, alpha=0.24, linewidth=1.1)
        ax.plot(time_hours[: current_index + 1], values[: current_index + 1], color=color, alpha=0.95, linewidth=2.0)
        force_zero = False
    ax.axvline(time_hours[current_index], color="#111827", alpha=0.8, linewidth=1.1, linestyle="--")
    ax.set_title(label, fontsize=10.5)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel(label)
    ax.set_ylim(*_diagnostic_ylim(values, force_zero_bottom=force_zero))
    ax.grid(True, alpha=0.25, linewidth=0.5)


def _draw_relative_l2_axis(
    ax: Any,
    *,
    relative_l2: Optional[np.ndarray],
    frame_idx: int,
    dt_seconds: float,
    rollout_start_index: int,
    initial_history_steps: int = DEFAULT_FORECAST_HISTORY_STEPS,
) -> None:
    ax.clear()
    if relative_l2 is None:
        ax.set_axis_off()
        return
    rel = np.asarray(relative_l2, dtype=np.float64).reshape(-1)
    if rel.size == 0:
        ax.set_axis_off()
        return
    xh = _forecast_horizon_hours(rel.size, dt_seconds, initial_history_steps=initial_history_steps)
    current = int(np.clip(frame_idx, 0, rel.size - 1))
    ax.plot(xh, rel, color="#7c2d12", alpha=0.24, linewidth=1.1)
    ax.plot(xh[: current + 1], rel[: current + 1], color="#7c2d12", alpha=0.98, linewidth=2.0)
    ax.scatter([xh[current]], [rel[current]], s=18, color="#7c2d12", zorder=3)
    ax.axvline(xh[current], color="#111827", alpha=0.8, linewidth=1.1, linestyle="--")
    ax.set_title("WD relative L2", fontsize=10.5)
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("rel. L2")
    ax.set_ylim(*_diagnostic_ylim(rel, force_zero_bottom=True))
    ax.grid(True, alpha=0.25, linewidth=0.5)


def _draw_rollout_diagnostics(
    *,
    diag_axes: Dict[str, Any],
    frame_idx: int,
    dt_seconds: float,
    boundary_series_raw: Optional[Any],
    boundary_channel_names: Optional[List[str]],
    relative_l2: Optional[np.ndarray],
    rollout_start_index: int,
    initial_history_steps: int = DEFAULT_FORECAST_HISTORY_STEPS,
) -> None:
    series = diag_axes.get("series")
    if series is None:
        series = _coerce_boundary_series_raw(boundary_series_raw)
    if series is not None:
        names = diag_axes.get("names") or _boundary_channel_names(boundary_channel_names, series.shape[1])
        visible_series = _visible_rollout_boundary_series(series, rollout_start_index)
        if visible_series.shape[0] > 0:
            time_hours = _forecast_horizon_hours(
                visible_series.shape[0],
                dt_seconds,
                initial_history_steps=initial_history_steps,
            )
            current_rel = int(np.clip(frame_idx, 0, visible_series.shape[0] - 1))
            for ax, channel_idx, kind in diag_axes.get("boundary_axes", []):
                _draw_boundary_diagnostic_axis(
                    ax,
                    time_hours=time_hours,
                    values=visible_series[:, channel_idx],
                    name=names[channel_idx],
                    kind=kind,
                    current_index=current_rel,
                )
        else:
            for ax, _, _ in diag_axes.get("boundary_axes", []):
                ax.clear()
                ax.set_axis_off()
    _draw_relative_l2_axis(
        diag_axes["relative_l2_axis"],
        relative_l2=relative_l2,
        frame_idx=frame_idx,
        dt_seconds=dt_seconds,
        rollout_start_index=rollout_start_index,
        initial_history_steps=initial_history_steps,
    )

def _save_hydrograph_uq_figures_and_animation(
    geometry: Any,
    pred_mean_by_channel: Dict[str, np.ndarray],
    pred_std_by_channel: Dict[str, np.ndarray],
    gt_mean_by_channel: Dict[str, np.ndarray],
    gt_std_by_channel: Dict[str, np.ndarray],
    target_variables: List[str],
    out_dir: str,
    hydrograph_id: str,
    dt_seconds: float,
    n_ref_sims: int,
    n_ens: int,
    pred_prob_wd: Optional[np.ndarray] = None,
    gt_prob_wd: Optional[np.ndarray] = None,
    crps_map_wd: Optional[np.ndarray] = None,
    boundary_series_raw: Optional[Any] = None,
    boundary_channel_names: Optional[List[str]] = None,
    relative_l2_by_channel: Optional[Dict[str, np.ndarray]] = None,
    rollout_start_index: int = 0,
    elevation_raw: Optional[Any] = None,
    visualization_config: Optional[Any] = None,
) -> None:
    """Generate publication-ready UQ figures and animations per hydrograph."""
    import matplotlib as mpl

    mpl.rc("font", family="serif", size=11)
    x, y = _geometry_xy(geometry)
    cartographic_context = _cartographic_context(
        x=x,
        y=y,
        elevation_raw=elevation_raw,
        out_dir=out_dir,
        visualization_config=visualization_config,
    )
    renderer_3x2 = _build_spatial_renderer(x, y, figsize=(12.4, 14.5), dpi=320, n_rows=3, n_cols=2)
    renderer_1x3 = _build_spatial_renderer(x, y, figsize=(16.5, 5.2), dpi=320, n_rows=1, n_cols=3)
    renderer_1x1 = _build_spatial_renderer(x, y, figsize=(6.8, 5.8), dpi=320, n_rows=1, n_cols=1)
    renderer_2x3 = _build_spatial_renderer(x, y, figsize=(14.8, 9.6), dpi=260, n_rows=2, n_cols=3)
    hid = hydrograph_id or "unknown"
    uq_dir = os.path.join(out_dir, "uq_figures_per_hydrograph")
    os.makedirs(uq_dir, exist_ok=True)
    n_steps = next(iter(pred_mean_by_channel.values())).shape[0]
    steps = [s for s in PUBLICATION_TIMESTEPS if 0 <= s < n_steps] or [0]

    for t in steps:
        for ch in target_variables:
            pred_mean = pred_mean_by_channel[ch][t]
            pred_std = pred_std_by_channel[ch][t]
            gt_mean = gt_mean_by_channel[ch][t]
            gt_std = gt_std_by_channel[ch][t]
            vmin_m, vmax_m, cmap_mean = _channel_vmin_vmax_cmap(ch, gt_mean, pred_mean)
            spread_max = _robust_nonnegative_vmax(gt_std, pred_std)
            bias = pred_mean - gt_mean
            abs_err = np.abs(bias)
            bmax = _robust_symmetric_abs_vmax(bias)
            emax = _robust_nonnegative_vmax(abs_err)

            fig, axs = plt.subplots(
                3, 2, figsize=(12.4, 14.5), dpi=320, constrained_layout=True
            )
            fig.suptitle(
                f"Hydrograph {hid} | {ch.upper()} | t={t} | GT ({n_ref_sims} sims) vs Forecast ({n_ens} ens)",
                fontsize=12.5,
            )

            panels = [
                ("GT mean", gt_mean, cmap_mean, vmin_m, vmax_m, ch),
                ("Forecast mean", pred_mean, cmap_mean, vmin_m, vmax_m, ch),
                ("Mean bias (pred - gt)", bias, "coolwarm", -bmax, bmax, "bias"),
                ("Absolute error", abs_err, "error_rose", 0.0, emax, "abs err"),
                ("GT spread (std)", gt_std, "spread_violet", 0.0, spread_max, "std"),
                ("Forecast spread (std)", pred_std, "spread_violet", 0.0, spread_max, "std"),
            ]
            for ax, (title, arr, cmap, vmin, vmax, cblabel) in zip(axs.flatten(), panels):
                sc, _ = _plot_spatial_panel(
                    ax=ax,
                    x=x,
                    y=y,
                    arr=arr,
                    renderer=renderer_3x2,
                    context=cartographic_context,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    is_wd_depth=(ch == "wd" and title in {"GT mean", "Forecast mean"}),
                    zero_transparent=title in {"Absolute error", "GT spread (std)", "Forecast spread (std)"},
                    annotate=(title == "GT mean"),
                )
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.axis("off")
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cbar.set_label(cblabel)

            fig.savefig(
                os.path.join(uq_dir, f"uq_{ch}_{hid}_t{t}.png"),
                bbox_inches="tight",
                pad_inches=0.1,
            )
            plt.close(fig)

    if pred_prob_wd is not None and gt_prob_wd is not None:
        pred_prob_mean = np.mean(pred_prob_wd, axis=0)
        gt_prob_mean = np.mean(gt_prob_wd, axis=0)
        diff_abs = np.abs(pred_prob_mean - gt_prob_mean)
        fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.2), dpi=320, constrained_layout=True)
        prob_err_max = _robust_nonnegative_vmax(diff_abs)
        items = [
            (f"GT mean P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", gt_prob_mean, "viridis", 0.0, 1.0),
            (f"Forecast mean P(wd>{UQ_EXCEEDANCE_THRESHOLD:.2f})", pred_prob_mean, "viridis", 0.0, 1.0),
            ("|Probability error|", diff_abs, "error_rose", 0.0, prob_err_max),
        ]
        for ax, (title, arr, cmap, vmin, vmax) in zip(axs, items):
            sc, _ = _plot_spatial_panel(
                ax=ax,
                x=x,
                y=y,
                arr=arr,
                renderer=renderer_1x3,
                context=cartographic_context,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                zero_transparent=title == "|Probability error|",
                annotate=title.startswith("GT"),
            )
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        fig.savefig(
            os.path.join(uq_dir, f"uq_wd_timeavg_prob_{hid}.png"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

    if crps_map_wd is not None:
        crps_mean = np.mean(crps_map_wd, axis=0)
        vmax = _robust_nonnegative_vmax(crps_mean)
        fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.8), dpi=320, constrained_layout=True)
        sc, _ = _plot_spatial_panel(
            ax=ax,
            x=x,
            y=y,
            arr=crps_mean,
            renderer=renderer_1x1,
            context=cartographic_context,
            cmap="crps_indigo",
            vmin=0.0,
            vmax=vmax,
            zero_transparent=True,
            annotate=True,
        )
        ax.set_title("WD CRPS map (time-mean)")
        ax.set_aspect("equal")
        ax.axis("off")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("CRPS")
        fig.savefig(
            os.path.join(uq_dir, f"uq_wd_timeavg_crps_{hid}.png"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
        plt.close(fig)

    if "wd" in pred_mean_by_channel:
        wd_pred_mean = pred_mean_by_channel["wd"]
        wd_pred_std = pred_std_by_channel["wd"]
        wd_gt_mean = gt_mean_by_channel["wd"]
        wd_gt_std = gt_std_by_channel["wd"]
        wd_abs_err = np.abs(wd_pred_mean - wd_gt_mean)
        wd_crps = np.asarray(crps_map_wd, dtype=np.float64) if crps_map_wd is not None else wd_abs_err
        crps_title = "WD CRPS" if crps_map_wd is not None else "WD CRPS unavailable; showing |error|"
        vmax = _wd_spatial_vmax(wd_pred_mean, wd_gt_mean)
        spread_max = _robust_nonnegative_vmax(wd_pred_std, wd_gt_std)
        err_max = _robust_nonnegative_vmax(wd_abs_err)
        crps_max = _robust_nonnegative_vmax(wd_crps)

        wd_relative_l2 = None
        if relative_l2_by_channel is not None:
            wd_relative_l2 = relative_l2_by_channel.get("wd")
            if wd_relative_l2 is None and relative_l2_by_channel:
                wd_relative_l2 = next(iter(relative_l2_by_channel.values()))

        anim_figsize = (15.8, 12.6)
        anim_dpi = 240
        renderer_anim = _build_spatial_renderer(
            x, y, figsize=anim_figsize, dpi=anim_dpi, n_rows=3, n_cols=3
        )
        fig = plt.figure(figsize=anim_figsize, dpi=anim_dpi, constrained_layout=True)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.52])
        ax_gt_m = fig.add_subplot(gs[0, 0])
        ax_pr_m = fig.add_subplot(gs[0, 1])
        ax_err = fig.add_subplot(gs[0, 2])
        ax_gt_s = fig.add_subplot(gs[1, 0])
        ax_pr_s = fig.add_subplot(gs[1, 1])
        ax_crps = fig.add_subplot(gs[1, 2])
        diag_axes = _make_rollout_diagnostic_axes(
            fig,
            gs,
            boundary_series_raw=boundary_series_raw,
            boundary_channel_names=boundary_channel_names,
        )

        s_gt_m, gt_edge_artists = _plot_spatial_panel(
            ax=ax_gt_m,
            x=x,
            y=y,
            arr=wd_gt_mean[0],
            renderer=renderer_anim,
            context=cartographic_context,
            is_wd_depth=True,
            annotate=False,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        s_pr_m, pred_edge_artists = _plot_spatial_panel(
            ax=ax_pr_m,
            x=x,
            y=y,
            arr=wd_pred_mean[0],
            renderer=renderer_anim,
            context=cartographic_context,
            is_wd_depth=True,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        s_err, _ = _plot_spatial_panel(
            ax=ax_err,
            x=x,
            y=y,
            arr=wd_abs_err[0],
            renderer=renderer_anim,
            context=cartographic_context,
            cmap="error_rose",
            vmin=0.0,
            vmax=err_max,
            zero_transparent=True,
        )
        s_gt_s, _ = _plot_spatial_panel(
            ax=ax_gt_s,
            x=x,
            y=y,
            arr=wd_gt_std[0],
            renderer=renderer_anim,
            context=cartographic_context,
            cmap="spread_violet",
            vmin=0.0,
            vmax=spread_max,
            zero_transparent=True,
        )
        s_pr_s, _ = _plot_spatial_panel(
            ax=ax_pr_s,
            x=x,
            y=y,
            arr=wd_pred_std[0],
            renderer=renderer_anim,
            context=cartographic_context,
            cmap="spread_violet",
            vmin=0.0,
            vmax=spread_max,
            zero_transparent=True,
        )
        s_crps, _ = _plot_spatial_panel(
            ax=ax_crps,
            x=x,
            y=y,
            arr=wd_crps[0],
            renderer=renderer_anim,
            context=cartographic_context,
            cmap="crps_indigo",
            vmin=0.0,
            vmax=crps_max,
            zero_transparent=True,
        )
        for ax, title in [
            (ax_gt_m, f"GT mean ({n_ref_sims} sims)"),
            (ax_pr_m, f"Forecast mean ({n_ens} ens)"),
            (ax_err, "Absolute error |mean|"),
            (ax_gt_s, "GT spread (std)"),
            (ax_pr_s, "Forecast spread (std)"),
            (ax_crps, crps_title),
        ]:
            ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
        fig.colorbar(s_gt_m, ax=ax_gt_m, fraction=0.046, pad=0.02)
        fig.colorbar(s_pr_m, ax=ax_pr_m, fraction=0.046, pad=0.02)
        fig.colorbar(s_err, ax=ax_err, fraction=0.046, pad=0.02)
        fig.colorbar(s_gt_s, ax=ax_gt_s, fraction=0.046, pad=0.02)
        fig.colorbar(s_pr_s, ax=ax_pr_s, fraction=0.046, pad=0.02)
        cb_crps = fig.colorbar(s_crps, ax=ax_crps, fraction=0.046, pad=0.02)
        cb_crps.set_label("CRPS (m)")
        options = cartographic_context.get("options", _visualization_options(None))
        initial_history_steps = int(options.get("initial_history_steps", DEFAULT_FORECAST_HISTORY_STEPS))
        _draw_rollout_diagnostics(
            diag_axes=diag_axes,
            frame_idx=0,
            dt_seconds=dt_seconds,
            boundary_series_raw=boundary_series_raw,
            boundary_channel_names=boundary_channel_names,
            relative_l2=wd_relative_l2,
            rollout_start_index=rollout_start_index,
            initial_history_steps=initial_history_steps,
        )

        def _animate(frame_idx: int) -> List[Any]:
            time_hours = _forecast_horizon_hours(
                frame_idx + 1,
                dt_seconds,
                initial_history_steps=initial_history_steps,
            )[-1]
            fig.suptitle(
                f"Hydrograph {hid} | forecast horizon {time_hours:.2f} h",
                fontsize=13,
            )
            _update_spatial_artist(
                s_gt_m,
                _mask_wd_dry_for_overlay(wd_gt_mean[frame_idx], options["wet_edge_threshold_m"]),
                renderer_anim,
            )
            _update_spatial_artist(
                s_pr_m,
                _mask_wd_dry_for_overlay(wd_pred_mean[frame_idx], options["wet_edge_threshold_m"]),
                renderer_anim,
            )
            _remove_artists(gt_edge_artists)
            _remove_artists(pred_edge_artists)
            if options.get("show_wet_edge", True):
                gt_edge_artists.extend(_draw_wet_edge(ax_gt_m, x, y, wd_gt_mean[frame_idx], renderer_anim, options["wet_edge_threshold_m"]))
                pred_edge_artists.extend(_draw_wet_edge(ax_pr_m, x, y, wd_pred_mean[frame_idx], renderer_anim, options["wet_edge_threshold_m"]))

            zero_threshold = _diagnostic_zero_threshold(options, err_max)
            spread_zero_threshold = _diagnostic_zero_threshold(options, spread_max)
            crps_zero_threshold = _diagnostic_zero_threshold(options, crps_max)
            _update_spatial_artist(s_err, _mask_near_zero_for_overlay(wd_abs_err[frame_idx], zero_threshold), renderer_anim)
            _update_spatial_artist(s_gt_s, _mask_near_zero_for_overlay(wd_gt_std[frame_idx], spread_zero_threshold), renderer_anim)
            _update_spatial_artist(s_pr_s, _mask_near_zero_for_overlay(wd_pred_std[frame_idx], spread_zero_threshold), renderer_anim)
            _update_spatial_artist(s_crps, _mask_near_zero_for_overlay(wd_crps[frame_idx], crps_zero_threshold), renderer_anim)
            _draw_rollout_diagnostics(
                diag_axes=diag_axes,
                frame_idx=frame_idx,
                dt_seconds=dt_seconds,
                boundary_series_raw=boundary_series_raw,
                boundary_channel_names=boundary_channel_names,
                relative_l2=wd_relative_l2,
                rollout_start_index=rollout_start_index,
                initial_history_steps=initial_history_steps,
            )
            return [s_gt_m, s_pr_m, s_err, s_gt_s, s_pr_s, s_crps, *gt_edge_artists, *pred_edge_artists]

        ani = animation.FuncAnimation(
            fig, _animate, frames=n_steps, interval=ANIMATION_INTERVAL_MS, blit=False
        )
        _save_animation_outputs(
            ani,
            os.path.join(uq_dir, f"uq_rollout_{hid}"),
            cartographic_context.get("options", _visualization_options(None)),
        )
        plt.close(fig)
