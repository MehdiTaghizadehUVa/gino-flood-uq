"""Build a top-CSI historical-event FGNO comparison figure.

This script is intentionally split into two subcommands because the Rivanna
HDF5 environment used for the artifacts does not include Matplotlib, while the
plotting environment does not include h5py.
"""

import argparse
import json
from pathlib import Path

import numpy as np


EVENTS = [
    {
        "event_id": "2023_OPHELIA",
        "display_name": "Ophelia 2023",
    },
    {
        "event_id": "2016_MATTHEW",
        "display_name": "Matthew 2016",
    },
    {
        "event_id": "2019_DORIAN",
        "display_name": "Dorian 2019",
    },
    {
        "event_id": "1998_NOREASTER",
        "display_name": "1998 Nor'easter",
    },
    {
        "event_id": "2012_SANDY",
        "display_name": "Sandy 2012",
    },
]

BUNDLE_NAME = "historical_top5_csi_fgno_max_water_depth_bundle.npz"
METADATA_NAME = "historical_top5_csi_fgno_max_water_depth_metadata.json"
FIGURE_PREFIX = "fig_historical_top5_csi_fgno_max_water_depth"


def _decode_names(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size < 3:
        return float("nan")
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 1e-12:
        return float("nan")
    return float(1.0 - np.sum((y_pred - y_true) ** 2) / denom)


def extract(args):
    import h5py

    artifact_root = Path(args.artifact_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = None
    elevation = None
    wettable_mask = None
    structural_dry_mask = None
    gt_max_fields = []
    pred_max_fields = []
    error_fields = []
    boundary_time = []
    stage_series = []
    precip_series = []
    r2_values = []
    scatter_counts = []

    for event in EVENTS:
        artifact_path = artifact_root / f"Flood_coastal_HIST_{event['event_id']}.calibration_artifact.h5"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing artifact: {artifact_path}")

        with h5py.File(artifact_path, "r") as f:
            if geometry is None:
                geometry = np.asarray(f["geometry_raw"][:], dtype=np.float64)
                elevation = np.asarray(f["elevation_raw"][:], dtype=np.float64)
                wettable_mask = np.asarray(f["wettable_mask"][:], dtype=bool)
                structural_dry_mask = np.asarray(f["structural_dry_mask"][:], dtype=bool)

            ref_members = np.asarray(f["ref_members_wd"][:], dtype=np.float32)
            pred_members = np.asarray(f["pred_members_wd"][:], dtype=np.float32)
            time_hours = np.asarray(f["time_hours"][:], dtype=np.float64)

            # Maximum-depth product: max over lead time per member, then member mean.
            gt_max = np.nanmean(np.nanmax(ref_members, axis=1), axis=0).astype(np.float32)
            pred_max = np.nanmean(np.nanmax(pred_members, axis=1), axis=0).astype(np.float32)
            err = np.abs(pred_max - gt_max).astype(np.float32)

            boundary = np.asarray(f["boundary_series_raw"][:], dtype=np.float64)
            names = _decode_names(f["boundary_channel_names"][:])
            try:
                stage_idx = names.index("stage")
                precip_idx = names.index("precipitation")
            except ValueError as exc:
                raise ValueError(f"Expected stage and precipitation channels in {artifact_path}") from exc

            # The artifact stores history plus forecast forcing. The final n_time
            # samples align with the forecast lead times used by the water-depth rollout.
            n_time = int(time_hours.shape[0])
            boundary_forecast = boundary[-n_time:, :]
            gt_max_fields.append(gt_max)
            pred_max_fields.append(pred_max)
            error_fields.append(err)
            boundary_time.append(time_hours)
            stage_series.append(boundary_forecast[:, stage_idx])
            precip_series.append(boundary_forecast[:, precip_idx])

            wet_mask = (
                wettable_mask
                & ~structural_dry_mask
                & np.isfinite(gt_max)
                & np.isfinite(pred_max)
                & ((gt_max >= args.scatter_threshold_m) | (pred_max >= args.scatter_threshold_m))
                & (gt_max <= args.scatter_max_m)
                & (pred_max <= args.scatter_max_m)
            )
            r2_values.append(_r2_score(gt_max[wet_mask], pred_max[wet_mask]))
            scatter_counts.append(int(np.count_nonzero(wet_mask)))

    bundle_path = out_dir / BUNDLE_NAME
    np.savez_compressed(
        bundle_path,
        geometry=np.asarray(geometry, dtype=np.float64),
        elevation=np.asarray(elevation, dtype=np.float64),
        wettable_mask=np.asarray(wettable_mask, dtype=bool),
        structural_dry_mask=np.asarray(structural_dry_mask, dtype=bool),
        gt_max_fields=np.asarray(gt_max_fields, dtype=np.float32),
        pred_max_fields=np.asarray(pred_max_fields, dtype=np.float32),
        error_fields=np.asarray(error_fields, dtype=np.float32),
        boundary_time=np.asarray(boundary_time, dtype=np.float64),
        stage_series=np.asarray(stage_series, dtype=np.float32),
        precip_series=np.asarray(precip_series, dtype=np.float32),
        r2_values=np.asarray(r2_values, dtype=np.float64),
        scatter_counts=np.asarray(scatter_counts, dtype=np.int64),
    )

    metadata = {
        "artifact_root": str(artifact_root),
        "events": EVENTS,
        "definitions": {
            "ground_truth_max_water_depth": "mean over HEC-RAS reference members of max over forecast lead time",
            "fgno_predicted_max_water_depth": "mean over FGNO forecast members of max over forecast lead time",
            "absolute_error": "absolute difference between predicted and HEC-RAS maximum water-depth products",
            "forcing_window": "final n_time boundary samples aligned to artifact time_hours",
            "scatter_r2_mask": (
                "wettable and non-structural-dry cells with either HEC-RAS or FGNO "
                f"maximum water depth >= {args.scatter_threshold_m:.3f} m and both "
                f"maximum-depth products <= {args.scatter_max_m:.3f} m"
            ),
        },
        "scatter_threshold_m": args.scatter_threshold_m,
        "scatter_max_m": args.scatter_max_m,
        "r2_values": r2_values,
        "scatter_counts": scatter_counts,
    }
    metadata_path = out_dir / METADATA_NAME
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {bundle_path}")
    print(f"Wrote {metadata_path}")


def _hecras_dem_cmap():
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
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


def _cyan_depth_cmap():
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(
        "cyan_depth",
        ["#dffcff", "#8ff7ff", "#19d9f2", "#0284c7", "#08306b"],
    )


def _error_cmap():
    import matplotlib.colors as mcolors

    def rgba(hex_color, alpha):
        return (*mcolors.to_rgb(hex_color), float(alpha))

    return mcolors.LinearSegmentedColormap.from_list(
        "error_magenta_alpha",
        [
            (0.00, rgba("#fff7ff", 0.00)),
            (0.14, rgba("#fae8ff", 0.12)),
            (0.35, rgba("#f0abfc", 0.40)),
            (0.62, rgba("#d946ef", 0.74)),
            (0.84, rgba("#86198f", 0.93)),
            (1.00, rgba("#3b0764", 1.00)),
        ],
    )


def _build_triangulation(x, y):
    import matplotlib.tri as mtri

    tri = mtri.Triangulation(x, y)
    tris = tri.triangles
    x0, y0 = x[tris[:, 0]], y[tris[:, 0]]
    x1, y1 = x[tris[:, 1]], y[tris[:, 1]]
    x2, y2 = x[tris[:, 2]], y[tris[:, 2]]
    l01 = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    l12 = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l20 = np.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)
    lmax = np.maximum(l01, np.maximum(l12, l20))
    finite = np.isfinite(lmax) & (lmax > 0.0)
    if np.any(finite):
        tri.set_mask(lmax > 2.5 * np.median(lmax[finite]))
    return tri


def _plot_map(
    ax,
    tri,
    x,
    y,
    elevation,
    field,
    cmap,
    vmin,
    vmax,
    *,
    is_depth=False,
    threshold=0.05,
    background_alpha=1.0,
    overlay_alpha=None,
    basemap_image=None,
    basemap_extent=None,
):
    if basemap_image is not None and basemap_extent is not None:
        ax.imshow(
            basemap_image,
            extent=basemap_extent,
            origin="upper",
            alpha=background_alpha,
            zorder=0,
        )
    else:
        elev = np.asarray(elevation, dtype=np.float64)
        finite = elev[np.isfinite(elev)]
        # Match neuralop.flood.eval.render DEM-backed visualization defaults.
        lo, hi = np.nanquantile(finite, [0.01, 0.99])
        elev_plot = np.clip(elev, lo, hi)
        ax.tripcolor(
            tri,
            elev_plot,
            shading="gouraud",
            cmap=_hecras_dem_cmap(),
            vmin=lo,
            vmax=hi,
            edgecolors="none",
            linewidth=0.0,
            rasterized=True,
            alpha=background_alpha,
            zorder=0,
        )

    arr = np.asarray(field, dtype=np.float64).copy()
    arr[~np.isfinite(arr)] = np.nan
    if is_depth:
        arr[arr < threshold] = np.nan
        alpha = 0.88
    else:
        arr[np.abs(arr) <= max(1e-10, 0.03 * max(float(vmax), 1e-12))] = np.nan
        alpha = None
    if overlay_alpha is not None:
        alpha = overlay_alpha
    artist = ax.tripcolor(
        tri,
        np.ma.masked_invalid(arr),
        shading="gouraud",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        linewidth=0.0,
        rasterized=True,
        alpha=alpha,
        zorder=2,
    )
    ax.set_aspect("equal")
    pad_x = 0.025 * (np.nanmax(x) - np.nanmin(x))
    pad_y = 0.025 * (np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
    ax.set_ylim(np.nanmin(y) - pad_y, np.nanmax(y) + pad_y)
    ax.axis("off")
    return artist


def _format_forcing_axis(ax, ax_rain, row_idx):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax_rain.spines["top"].set_visible(False)
    ax.grid(True, axis="both", color="#D7DBE7", linewidth=0.55, alpha=0.65)
    ax.tick_params(axis="both", labelsize=9, colors="#303746", length=2)
    ax_rain.tick_params(axis="y", labelsize=9, colors="#6F768A", length=2)
    if row_idx < len(EVENTS) - 1:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel("Lead time (h)", fontsize=10)
    ax.set_ylabel("Stage (m)", fontsize=10)
    ax_rain.set_ylabel("Rainfall\n(mm/15 min)", fontsize=10, color="#6F768A")


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    out_dir = Path(args.out_dir)
    data = np.load(str(out_dir / BUNDLE_NAME))
    metadata = json.loads((out_dir / METADATA_NAME).read_text(encoding="utf-8"))

    geometry = data["geometry"]
    x = geometry[:, 0]
    y = geometry[:, 1]
    elevation = data["elevation"]
    gt = data["gt_max_fields"]
    pred = data["pred_max_fields"]
    err = data["error_fields"]
    boundary_time = data["boundary_time"]
    stage = data["stage_series"]
    precip = data["precip_series"]
    wettable = data["wettable_mask"].astype(bool)
    structural_dry = data["structural_dry_mask"].astype(bool)
    r2_values = data["r2_values"]
    scatter_counts = data["scatter_counts"]
    basemap_image = None
    basemap_extent = None
    if args.basemap_context:
        basemap = np.load(str(args.basemap_context))
        basemap_image = basemap["image"]
        basemap_extent = basemap["extent"]

    tri = _build_triangulation(x, y)
    depth_cmap = _cyan_depth_cmap()
    error_cmap = _error_cmap()
    depth_vmax = float(args.depth_vmax)
    if depth_vmax <= 0:
        depth_vmax = min(3.0, float(np.nanquantile(np.concatenate([gt.reshape(-1), pred.reshape(-1)]), 0.995)))
        depth_vmax = max(depth_vmax, 0.5)
    error_vmax = float(args.error_vmax)
    if error_vmax <= 0:
        error_vmax = float(np.nanquantile(err.reshape(-1), 0.995))
        error_vmax = max(error_vmax, 0.05)

    plt.rcParams.update(
        {
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FCFCFD",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "xtick.color": "#303746",
            "ytick.color": "#303746",
            "text.color": "#1F2430",
        }
    )

    fig = plt.figure(figsize=(22.0, 18.0), dpi=300)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=5,
        width_ratios=[1.45, 1.10, 1.10, 1.10, 1.05],
        wspace=0.12,
        hspace=0.16,
        left=0.075,
        right=0.955,
        top=0.940,
        bottom=0.058,
    )
    axes = [[fig.add_subplot(gs[i, j]) for j in range(5)] for i in range(5)]

    headers = [
        "Stage and rainfall",
        "HEC-RAS max water depth",
        "FGNO max water depth",
        "Absolute error",
        "Cellwise agreement",
    ]
    for j, header in enumerate(headers):
        axes[0][j].set_title(header, fontsize=16, fontweight="bold", pad=12)

    row_centers = []
    for i, event in enumerate(metadata["events"]):
        row_centers.append((axes[i][0].get_position().y0 + axes[i][0].get_position().y1) / 2.0)

        ax = axes[i][0]
        ax_rain = ax.twinx()
        t = boundary_time[i]
        rain_width = np.median(np.diff(t)) * 0.82 if t.size > 1 else 0.2
        ax.plot(t, stage[i], color="#1F4E79", linewidth=1.55, zorder=3)
        ax_rain.bar(
            t,
            precip[i],
            width=rain_width,
            color="#CC6F47",
            edgecolor="#804126",
            linewidth=0.22,
            alpha=0.72,
            zorder=2,
        )
        ax_rain.invert_yaxis()
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        ax.set_ylim(
            max(0.0, float(np.nanmin(stage[i])) - 0.12),
            float(np.nanmax(stage[i])) + 0.15,
        )
        rain_hi = max(float(np.nanmax(precip[i])) * 1.22, 0.1)
        ax_rain.set_ylim(rain_hi, 0.0)
        _format_forcing_axis(ax, ax_rain, i)
        if i == 0:
            ax.plot([], [], color="#1F4E79", linewidth=1.55, label="Stage")
            ax_rain.bar([], [], color="#CC6F47", edgecolor="#804126", linewidth=0.22, alpha=0.72, label="Rainfall")
            handles = [
                plt.Line2D([0], [0], color="#1F4E79", linewidth=1.55, label="Stage"),
                plt.Rectangle((0, 0), 1, 1, facecolor="#CC6F47", edgecolor="#804126", alpha=0.72, label="Rainfall"),
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=9, frameon=False, handlelength=1.6)

        _plot_map(
            axes[i][1],
            tri,
            x,
            y,
            elevation,
            gt[i],
            depth_cmap,
            0.0,
            depth_vmax,
            is_depth=True,
            threshold=0.05,
            background_alpha=1.0,
            basemap_image=basemap_image,
            basemap_extent=basemap_extent,
        )
        _plot_map(
            axes[i][2],
            tri,
            x,
            y,
            elevation,
            pred[i],
            depth_cmap,
            0.0,
            depth_vmax,
            is_depth=True,
            threshold=0.05,
            background_alpha=1.0,
            basemap_image=basemap_image,
            basemap_extent=basemap_extent,
        )
        _plot_map(
            axes[i][3],
            tri,
            x,
            y,
            elevation,
            err[i],
            error_cmap,
            0.0,
            error_vmax,
            is_depth=False,
            background_alpha=0.28,
            basemap_image=basemap_image,
            basemap_extent=basemap_extent,
        )

        ax_sc = axes[i][4]
        mask = (
            wettable
            & ~structural_dry
            & np.isfinite(gt[i])
            & np.isfinite(pred[i])
            & ((gt[i] >= args.scatter_threshold_m) | (pred[i] >= args.scatter_threshold_m))
            & (gt[i] <= args.scatter_max_m)
            & (pred[i] <= args.scatter_max_m)
        )
        x_sc = gt[i][mask]
        y_sc = pred[i][mask]
        ax_sc.scatter(
            x_sc,
            y_sc,
            s=6.5,
            facecolor="#A3BEFA",
            edgecolor="#2E4780",
            linewidth=0.16,
            alpha=0.46,
            rasterized=True,
        )
        lim = float(args.scatter_max_m) if args.scatter_max_m > 0 else depth_vmax
        ax_sc.plot([0, lim], [0, lim], color="#464C55", linewidth=0.8, linestyle="--")
        ax_sc.set_xlim(0, lim)
        ax_sc.set_ylim(0, lim)
        ax_sc.grid(True, color="#E6E8F0", linewidth=0.55, alpha=0.7)
        ax_sc.tick_params(labelsize=9, length=2)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.text(
            0.055,
            0.93,
            "$R^2$ = {:.2f}\n$n$ = {:,}".format(float(r2_values[i]), int(scatter_counts[i])),
            transform=ax_sc.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
        )
        if i == len(EVENTS) - 1:
            ax_sc.set_xlabel("HEC-RAS max water depth (m)", fontsize=10)
        else:
            ax_sc.tick_params(axis="x", labelbottom=False)
        ax_sc.set_ylabel("FGNO max water depth (m)", fontsize=10)

    for i, event in enumerate(metadata["events"]):
        fig.text(
            0.042,
            row_centers[i],
            "{}".format(event["display_name"]),
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
            linespacing=1.2,
        )

    depth_sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=depth_vmax), cmap=depth_cmap)
    depth_sm.set_array([])
    err_sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=error_vmax), cmap=error_cmap)
    err_sm.set_array([])
    cax_depth = fig.add_axes([0.275, 0.021, 0.245, 0.012])
    cbar_depth = fig.colorbar(depth_sm, cax=cax_depth, orientation="horizontal")
    cbar_depth.set_label("Maximum water depth (m)", fontsize=11, labelpad=4)
    cbar_depth.ax.tick_params(labelsize=10, length=2)
    cax_err = fig.add_axes([0.560, 0.021, 0.175, 0.012])
    cbar_err = fig.colorbar(err_sm, cax=cax_err, orientation="horizontal")
    cbar_err.set_label("Absolute error (m)", fontsize=11, labelpad=4)
    cbar_err.ax.tick_params(labelsize=10, length=2)

    for ext in ("png", "pdf", "svg"):
        out_path = out_dir / f"{FIGURE_PREFIX}.{ext}"
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.04)
        print(f"Wrote {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--artifact-root", required=True)
    p_extract.add_argument("--out-dir", required=True)
    p_extract.add_argument("--scatter-threshold-m", type=float, default=0.10)
    p_extract.add_argument("--scatter-max-m", type=float, default=3.0)
    p_extract.set_defaults(func=extract)

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--out-dir", required=True)
    p_plot.add_argument("--depth-vmax", type=float, default=3.0)
    p_plot.add_argument("--error-vmax", type=float, default=0.0)
    p_plot.add_argument("--scatter-threshold-m", type=float, default=0.10)
    p_plot.add_argument("--scatter-max-m", type=float, default=3.0)
    p_plot.add_argument("--basemap-context", default=None)
    p_plot.set_defaults(func=plot)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
