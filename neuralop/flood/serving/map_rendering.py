"""Serving-safe spatial map rendering helpers.

This adapter reuses the evaluation renderer's map machinery so web artifacts
match the coastal calibration figures' cartographic style without pulling
ground-truth/error diagnostics into user-serving products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SERVING_VISUALIZATION_CONFIG = {
    "map": {
        "enabled": True,
        "mode": "dem_elevation",
        "crs": "EPSG:32618",
        "provider": "local_elevation",
        "fallback": "elevation_hillshade",
        "cache_scope": "run_extent",
        "dem_cmap": "hecras_dem",
        "dem_quantiles": [0.01, 0.99],
    },
    "wd": {
        "colormap": "cyan_depth",
        "overlay_alpha": 0.88,
        "show_wet_edge": False,
        "wet_edge_threshold_m": 0.05,
    },
    "diagnostics": {
        "basemap_alpha": 0.28,
        "zero_fraction": 0.03,
        "zero_threshold": 1.0e-10,
        "spread_colormap": "spread_violet_alpha_ramp",
    },
}


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
        context = eval_render._cartographic_context(
            x=x,
            y=y,
            elevation_raw=elevation_raw,
            out_dir=tmp_context.name,
            visualization_config=visualization_config or DEFAULT_SERVING_VISUALIZATION_CONFIG,
        )
        fig, ax = plt.subplots(
            1,
            1,
            figsize=fig_size,
            dpi=dpi,
            constrained_layout=True,
            facecolor="#f7fafb",
        )
        try:
            ax.set_facecolor("#f7fafb")
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

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from neuralop.flood.eval import render as eval_render

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    x = xy[:, 0]
    y = xy[:, 1]
    fig_size = (7.2, 6.1)
    dpi = int(dpi)
    renderer = eval_render._build_spatial_renderer(x, y, figsize=fig_size, dpi=dpi, n_rows=1, n_cols=1)
    context = eval_render._cartographic_context(
        x=x,
        y=y,
        elevation_raw=elevation_raw,
        out_dir=str(Path(output_path).parent),
        visualization_config=visualization_config or DEFAULT_SERVING_VISUALIZATION_CONFIG,
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
        facecolor="#f7fafb",
    )
    try:
        ax.set_facecolor("#f7fafb")
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
