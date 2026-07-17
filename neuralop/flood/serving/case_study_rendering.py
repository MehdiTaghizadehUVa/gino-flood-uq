"""Publication-style rendering primitives for static marketing evidence."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _save_clean_svg(fig, output: Path) -> None:
    """Write deterministic SVG text without Matplotlib's trailing spaces."""
    fig.savefig(output, format="svg", bbox_inches="tight", pad_inches=0.04)
    lines = output.read_text(encoding="utf-8").splitlines()
    output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class TerrainContext:
    image: np.ndarray
    extent: tuple[float, float, float, float]
    source_crs: str
    target_crs: str
    source_path: str
    viewport: tuple[float, float, float, float] | None = None


def hecras_dem_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "hecras_dem",
        [
            (0.0000, "#b9f6ff"),
            (0.4314, "#b6e500"),
            (0.4886, "#008b2d"),
            (0.5086, "#f1e51c"),
            (0.5229, "#ff8a00"),
            (0.5371, "#b00000"),
            (0.5543, "#bfbfbf"),
            (1.0000, "#f2f2f2"),
        ],
    )


def probability_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "flooduq_probability",
        ["#7BE4F0", "#1FB7C9", "#08799E", "#0B3F78", "#071B4D"],
    )


def uncertainty_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "flooduq_uncertainty",
        ["#D6C2FF", "#A879E6", "#7B3FC1", "#4C168A", "#23004A"],
    )


def depth_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "flooduq_depth",
        ["#BFF7FF", "#55D8E8", "#24AFCB", "#2677A8", "#173E73"],
    )


def arrival_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "flooduq_arrival",
        ["#092955", "#2C6E9D", "#73A9B0", "#D7B96C", "#F4D83D"],
    )


def build_spatial_triangulation(x: np.ndarray, y: np.ndarray):
    import matplotlib.tri as mtri

    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    tri = mtri.Triangulation(x_arr, y_arr)
    triangles = np.asarray(tri.triangles, dtype=np.int64)
    p0 = np.column_stack((x_arr[triangles[:, 0]], y_arr[triangles[:, 0]]))
    p1 = np.column_stack((x_arr[triangles[:, 1]], y_arr[triangles[:, 1]]))
    p2 = np.column_stack((x_arr[triangles[:, 2]], y_arr[triangles[:, 2]]))
    edge_max = np.maximum.reduce(
        (
            np.linalg.norm(p0 - p1, axis=1),
            np.linalg.norm(p1 - p2, axis=1),
            np.linalg.norm(p2 - p0, axis=1),
        )
    )
    finite = np.isfinite(edge_max) & (edge_max > 0.0)
    if np.any(finite):
        tri.set_mask(edge_max > 2.5 * np.median(edge_max[finite]))
    return tri


def load_terrain_context(
    terrain_tif: str | Path,
    *,
    target_crs: str,
    max_width: int = 1400,
) -> TerrainContext:
    """Read and explicitly reproject the external DEM for web rendering."""
    try:
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.transform import from_bounds
        from rasterio.warp import reproject, transform_bounds
    except Exception as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("Case-study export requires rasterio.") from exc

    path = Path(terrain_tif).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Terrain GeoTIFF not found: {path}")
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"Terrain GeoTIFF has no CRS: {path}")
        source_crs = src.crs.to_string()
        left, bottom, right, top = transform_bounds(
            src.crs,
            target_crs,
            *src.bounds,
            densify_pts=21,
        )
        aspect = max((right - left) / max(top - bottom, 1.0e-12), 0.1)
        width = max(320, int(max_width))
        height = max(240, int(round(width / aspect)))
        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=from_bounds(left, bottom, right, top, width, height),
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return TerrainContext(
        image=destination,
        extent=(float(left), float(right), float(bottom), float(top)),
        source_crs=source_crs,
        target_crs=str(target_crs),
        source_path=str(path),
    )


def _save_webp(fig, output_path: Path, *, quality: int = 72, tight: bool = True) -> Path:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        png_path = Path(handle.name)
    try:
        save_options = {"dpi": 130, "facecolor": fig.get_facecolor()}
        if tight:
            save_options.update({"bbox_inches": "tight", "pad_inches": 0.035})
        fig.savefig(png_path, **save_options)
        with Image.open(png_path) as image:
            image.convert("RGB").save(output_path, "WEBP", quality=int(quality), method=6)
    finally:
        png_path.unlink(missing_ok=True)
    return output_path


def render_spatial_webp(
    *,
    values: np.ndarray,
    geometry_xy: np.ndarray,
    terrain: TerrainContext,
    output_path: str | Path,
    title: str,
    colorbar_label: str,
    cmap,
    vmin: float,
    vmax: float,
    display_floor: float,
    reference_values: np.ndarray | None = None,
    reference_threshold: float | None = None,
    markers: Sequence[tuple[float, float, str, bool]] = (),
    quality: int = 72,
    show_title: bool = False,
    show_colorbar: bool = True,
) -> Path:
    """Render a title-free full-DEM map for an HTML-captioned figure."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    from neuralop.flood.serving.case_study_export import masked_triangle_face_values

    xy = np.asarray(geometry_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("geometry_xy must have shape [n_cells,2].")
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] != xy.shape[0]:
        raise ValueError("Map values must match the geometry cell count.")
    tri = build_spatial_triangulation(xy[:, 0], xy[:, 1])
    _, threshold_mask = masked_triangle_face_values(
        values=arr,
        triangles=tri.triangles,
        display_floor=display_floor,
    )
    base_mask = np.asarray(tri.mask, dtype=bool) if tri.mask is not None else np.zeros(len(tri.triangles), dtype=bool)
    overlay_tri = mtri.Triangulation(
        xy[:, 0],
        xy[:, 1],
        triangles=tri.triangles,
        mask=base_mask | threshold_mask,
    )

    image_extent = terrain.extent
    viewport = terrain.viewport or image_extent
    aspect = max((viewport[1] - viewport[0]) / max(viewport[3] - viewport[2], 1.0e-12), 0.1)
    fig_width = 8.6
    fig_height = max(5.7, fig_width / aspect + 0.75)
    # Keep the legacy arguments compatible, but let the website own all titles.
    _ = (title, show_title)
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        decorated = bool(show_colorbar)
        if decorated:
            fig, ax = plt.subplots(
                figsize=(fig_width, fig_height),
                dpi=160,
                constrained_layout=True,
                facecolor="#F8FBFD",
            )
        else:
            fig = plt.figure(figsize=(fig_width, fig_width / aspect), dpi=160, facecolor="#F8FBFD")
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        try:
            ax.imshow(
                terrain.image,
                extent=image_extent,
                origin="upper",
                cmap=hecras_dem_cmap(),
                vmin=-15.1,
                vmax=19.9,
                alpha=0.92,
                interpolation="bilinear",
                zorder=0,
            )
            low_rgba = cmap(0.0)
            transparent_low = (float(low_rgba[0]), float(low_rgba[1]), float(low_rgba[2]), 0.0)
            overlay_cmap = cmap.with_extremes(bad=transparent_low, under=transparent_low)
            artist = ax.tripcolor(
                overlay_tri,
                np.ma.masked_invalid(arr),
                shading="gouraud",
                cmap=overlay_cmap,
                vmin=float(vmin),
                vmax=float(vmax),
                edgecolors="none",
                linewidth=0.0,
                antialiaseds=False,
                alpha=0.90,
                rasterized=True,
                zorder=2,
            )
            if reference_values is not None and reference_threshold is not None:
                ref = np.asarray(reference_values, dtype=np.float64).reshape(-1)
                finite = ref[np.isfinite(ref)]
                if finite.size and finite.min() < reference_threshold < finite.max():
                    contour = ax.tricontour(
                        tri,
                        ref,
                        levels=[float(reference_threshold)],
                        colors=["#FFFFFF"],
                        linewidths=1.25,
                        zorder=4,
                    )
                    for collection in getattr(contour, "collections", []):
                        collection.set_path_effects([pe.Stroke(linewidth=3.0, foreground="#101820"), pe.Normal()])
            for marker_x, marker_y, label, selected in markers:
                edge = "#7FD6FF" if selected else "#FFFFFF"
                ax.scatter(
                    [marker_x],
                    [marker_y],
                    s=82 if selected else 60,
                    facecolor="#071B4D",
                    edgecolor=edge,
                    linewidth=2.0,
                    zorder=6,
                )
                ax.text(
                    marker_x,
                    marker_y,
                    label,
                    ha="center",
                    va="center",
                    color="#FFFFFF",
                    fontsize=7.5,
                    fontweight="bold",
                    zorder=7,
                )
            ax.set_xlim(viewport[0], viewport[1])
            ax.set_ylim(viewport[2], viewport[3])
            ax.set_aspect("equal")
            ax.axis("off")
            if show_colorbar:
                colorbar = fig.colorbar(artist, ax=ax, fraction=0.034, pad=0.018)
                colorbar.set_label(colorbar_label, color="#102027", fontsize=8.5)
                colorbar.ax.tick_params(labelsize=8, colors="#3F535B")
                colorbar.outline.set_edgecolor("#AEBFC7")
            return _save_webp(fig, Path(output_path), quality=quality, tight=decorated)
        finally:
            plt.close(fig)


def render_location_panel_svg(
    *,
    lead_time_hours: np.ndarray,
    members_wd: np.ndarray,
    exceedance_probability: np.ndarray,
    threshold_m: float,
    output_path: str | Path,
    location_label: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    lead = np.asarray(lead_time_hours, dtype=np.float64)
    members = np.asarray(members_wd, dtype=np.float64)
    probability = np.asarray(exceedance_probability, dtype=np.float64)
    q05, q50, q95 = np.quantile(members, [0.05, 0.50, 0.95], axis=0)
    arrivals: list[float] = []
    for member in members:
        indices = np.flatnonzero(member > float(threshold_m))
        if indices.size:
            arrivals.append(float(lead[int(indices[0])]))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    _ = location_label
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 8.5}):
        fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), dpi=150, facecolor="#FFFFFF")
        try:
            for member in members:
                axes[0].plot(lead, member, color="#2A7189", alpha=0.055, linewidth=0.55)
            axes[0].fill_between(lead, q05, q95, color="#7FD6FF", alpha=0.34, linewidth=0)
            axes[0].plot(lead, q50, color="#075E75", linewidth=1.65)
            axes[0].set_ylabel("Water depth (m)")
            axes[0].set_xlabel("Lead time (h)")

            axes[1].plot(lead, probability, color="#08799E", linewidth=1.8)
            axes[1].fill_between(lead, 0.0, probability, color="#7BE4F0", alpha=0.24)
            axes[1].axhline(0.5, color="#6B7280", linewidth=0.8, linestyle="--")
            axes[1].set_ylim(0.0, 1.0)
            axes[1].set_ylabel("Calibrated probability")
            axes[1].set_xlabel("Lead time (h)")

            if arrivals:
                bins = min(12, max(4, len(set(round(value, 2) for value in arrivals))))
                axes[2].hist(arrivals, bins=bins, color="#7B3FC1", alpha=0.78, edgecolor="#4C168A")
            else:
                axes[2].text(0.5, 0.5, "No member exceeds\nthe selected threshold", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_ylabel("Members")
            axes[2].set_xlabel("First exceedance lead (h)")

            for ax in axes:
                ax.grid(True, color="#DDE6EB", linewidth=0.55, alpha=0.8)
                for spine in ax.spines.values():
                    spine.set_color("#C8D5DC")
            fig.tight_layout()
            _save_clean_svg(fig, output)
        finally:
            plt.close(fig)
    return output


def render_validation_trajectory_svg(
    *,
    lead_time_hours: np.ndarray,
    p05: np.ndarray,
    p50: np.ndarray,
    p95: np.ndarray,
    reference: np.ndarray,
    threshold_m: float,
    output_path: str | Path,
    event_label: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lead = np.asarray(lead_time_hours, dtype=np.float64)
    _ = event_label
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, ax = plt.subplots(figsize=(5.6, 3.25), dpi=150, facecolor="#FFFFFF")
        try:
            ax.fill_between(lead, p05, p95, color="#94D7BF", alpha=0.42, linewidth=0, label="FGNO 5–95%")
            ax.plot(lead, p50, color="#0B8A5A", linewidth=1.75, label="FGNO median")
            ax.plot(lead, reference, color="#1F2430", linewidth=1.25, linestyle="--", label="HEC-RAS")
            ymax = max(float(np.nanmax(p95)), float(np.nanmax(reference)), 0.05) * 1.12
            ax.set_xlim(float(lead.min()), float(lead.max()))
            ax.set_ylim(0.0, min(1.0, ymax))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.set_ylabel(f"Area fraction above {threshold_m:.2f} m")
            ax.set_xlabel("Lead time (h)")
            ax.grid(True, color="#E0E7EC", linewidth=0.6, alpha=0.8)
            ax.legend(loc="upper left", frameon=False, fontsize=8)
            for spine in ax.spines.values():
                spine.set_color("#C8D5DC")
            fig.tight_layout()
            _save_clean_svg(fig, output)
        finally:
            plt.close(fig)
    return output
