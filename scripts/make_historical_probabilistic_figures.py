"""Create historical-event probabilistic FGNO figures.

The script is split into extraction and plotting subcommands because the
Rivanna HDF5 environment and plotting environment are separate.
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np


H1_EVENTS = [
    {
        "event_id": "2023_OPHELIA",
        "display_name": "Ophelia 2023",
        "role": "high-skill hindcast",
    },
    {
        "event_id": "2009_NORIDA",
        "display_name": "Nor'Ida 2009",
        "role": "large compound event",
    },
    {
        "event_id": "2003_ISABEL",
        "display_name": "Isabel 2003",
        "role": "challenging event",
    },
]

EVENT_LABELS = {
    "Flood_coastal_HIST_1998_NOREASTER": "1998 Nor'easter",
    "Flood_coastal_HIST_1999_FLOYD": "Floyd 1999",
    "Flood_coastal_HIST_2003_ISABEL": "Isabel 2003",
    "Flood_coastal_HIST_2006_ERNESTO": "Ernesto 2006",
    "Flood_coastal_HIST_2006_NOV_NOREASTER": "Nov. Nor'easter 2006",
    "Flood_coastal_HIST_2006_OCT_ONSHORE": "Oct. onshore 2006",
    "Flood_coastal_HIST_2009_NORIDA": "Nor'Ida 2009",
    "Flood_coastal_HIST_2011_IRENE": "Irene 2011",
    "Flood_coastal_HIST_2012_SANDY": "Sandy 2012",
    "Flood_coastal_HIST_2016_MATTHEW": "Matthew 2016",
    "Flood_coastal_HIST_2019_DORIAN": "Dorian 2019",
    "Flood_coastal_HIST_2023_OPHELIA": "Ophelia 2023",
    "Flood_coastal_HIST_2025_HYBRID": "Hybrid 2025",
}

METHODS = [
    ("GINO_Perturbed", "GINO-Perturbed", "#6B7280", "o"),
    ("MC_dropout_GINO", "MC-dropout GINO", "#A35F35", "s"),
    ("Denoising_Diffusion_Operator", "Denoising Diffusion Operator", "#2F6BCB", "^"),
    ("FGNO", "FGNO", "#0B8A5A", "D"),
]

H1_BUNDLE_NAME = "figH1_historical_probabilistic_hindcast_bundle.npz"
H1_METADATA_NAME = "figH1_historical_probabilistic_hindcast_metadata.json"
H1_PREFIX = "figH1_historical_probabilistic_hindcast"
H2_PREFIX = "figH2_historical_probabilistic_skill_summary"
H2_SOURCE_NAME = "figH2_historical_probabilistic_skill_summary_source.csv"


def _decode_names(values):
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def extract_h1(args):
    import h5py

    artifact_root = Path(args.artifact_root)
    out_dir = _ensure_dir(args.out_dir)
    threshold = float(args.threshold_m)

    geometry = None
    elevation = None
    wettable_mask = None
    structural_dry_mask = None
    event_ids = []
    display_names = []
    roles = []
    boundary_time = []
    stage_series = []
    precip_series = []
    prob_exceed_max = []
    width90_max = []
    ref_max_depth = []
    pred_frac_p05 = []
    pred_frac_p50 = []
    pred_frac_p95 = []
    pred_frac_mean = []
    ref_frac = []
    coverage90_time = []

    for event in H1_EVENTS:
        artifact_path = artifact_root / ("Flood_coastal_HIST_%s.calibration_artifact.h5" % event["event_id"])
        if not artifact_path.exists():
            raise FileNotFoundError("Missing artifact: %s" % artifact_path)

        with h5py.File(str(artifact_path), "r") as f:
            if geometry is None:
                geometry = np.asarray(f["geometry_raw"][:], dtype=np.float64)
                elevation = np.asarray(f["elevation_raw"][:], dtype=np.float64)
                wettable_mask = np.asarray(f["wettable_mask"][:], dtype=bool)
                structural_dry_mask = np.asarray(f["structural_dry_mask"][:], dtype=bool)

            pred = np.asarray(f["pred_members_wd"][:], dtype=np.float32)
            ref = np.asarray(f["ref_members_wd"][:], dtype=np.float32)
            time_hours = np.asarray(f["time_hours"][:], dtype=np.float64)
            boundary = np.asarray(f["boundary_series_raw"][:], dtype=np.float64)
            names = _decode_names(f["boundary_channel_names"][:])
            try:
                stage_idx = names.index("stage")
                precip_idx = names.index("precipitation")
            except ValueError as exc:
                raise ValueError("Expected stage and precipitation channels in %s" % artifact_path) from exc

        mask = wettable_mask & (~structural_dry_mask)
        n_time = int(time_hours.shape[0])
        boundary_forecast = boundary[-n_time:, :]

        pred_max_member = np.nanmax(pred, axis=1)
        ref_max_member = np.nanmax(ref, axis=1)
        ref_max = np.nanmean(ref_max_member, axis=0)
        prob = np.nanmean(pred_max_member >= threshold, axis=0)
        width = np.nanpercentile(pred_max_member, 95, axis=0) - np.nanpercentile(pred_max_member, 5, axis=0)

        pred_wet_frac = np.nanmean(pred[:, :, mask] >= threshold, axis=2)
        ref_wet_frac = np.nanmean(ref[:, :, mask] >= threshold, axis=2)
        ref_wet_mean = np.nanmean(ref_wet_frac, axis=0)

        p05 = np.nanpercentile(pred_wet_frac, 5, axis=0)
        p50 = np.nanpercentile(pred_wet_frac, 50, axis=0)
        p95 = np.nanpercentile(pred_wet_frac, 95, axis=0)
        pmean = np.nanmean(pred_wet_frac, axis=0)

        q05 = np.nanpercentile(pred[:, :, mask], 5, axis=0)
        q95 = np.nanpercentile(pred[:, :, mask], 95, axis=0)
        ref_mean = np.nanmean(ref[:, :, mask], axis=0)
        coverage_t = np.nanmean((ref_mean >= q05) & (ref_mean <= q95), axis=1)

        prob[~mask] = np.nan
        width[~mask] = np.nan
        ref_max[~mask] = np.nan

        event_ids.append(event["event_id"])
        display_names.append(event["display_name"])
        roles.append(event["role"])
        boundary_time.append(time_hours)
        stage_series.append(boundary_forecast[:, stage_idx])
        precip_series.append(boundary_forecast[:, precip_idx])
        prob_exceed_max.append(prob.astype(np.float32))
        width90_max.append(width.astype(np.float32))
        ref_max_depth.append(ref_max.astype(np.float32))
        pred_frac_p05.append(p05.astype(np.float32))
        pred_frac_p50.append(p50.astype(np.float32))
        pred_frac_p95.append(p95.astype(np.float32))
        pred_frac_mean.append(pmean.astype(np.float32))
        ref_frac.append(ref_wet_mean.astype(np.float32))
        coverage90_time.append(coverage_t.astype(np.float32))

    np.savez_compressed(
        out_dir / H1_BUNDLE_NAME,
        geometry=np.asarray(geometry, dtype=np.float64),
        elevation=np.asarray(elevation, dtype=np.float64),
        wettable_mask=np.asarray(wettable_mask, dtype=bool),
        structural_dry_mask=np.asarray(structural_dry_mask, dtype=bool),
        event_ids=np.asarray(event_ids, dtype=object),
        display_names=np.asarray(display_names, dtype=object),
        roles=np.asarray(roles, dtype=object),
        boundary_time=np.asarray(boundary_time, dtype=np.float64),
        stage_series=np.asarray(stage_series, dtype=np.float32),
        precip_series=np.asarray(precip_series, dtype=np.float32),
        prob_exceed_max=np.asarray(prob_exceed_max, dtype=np.float32),
        width90_max=np.asarray(width90_max, dtype=np.float32),
        ref_max_depth=np.asarray(ref_max_depth, dtype=np.float32),
        pred_frac_p05=np.asarray(pred_frac_p05, dtype=np.float32),
        pred_frac_p50=np.asarray(pred_frac_p50, dtype=np.float32),
        pred_frac_p95=np.asarray(pred_frac_p95, dtype=np.float32),
        pred_frac_mean=np.asarray(pred_frac_mean, dtype=np.float32),
        ref_frac=np.asarray(ref_frac, dtype=np.float32),
        coverage90_time=np.asarray(coverage90_time, dtype=np.float32),
    )

    metadata = {
        "artifact_root": str(artifact_root),
        "events": H1_EVENTS,
        "threshold_m": threshold,
        "definitions": {
            "probability_map": "Fraction of FGNO members whose cellwise maximum water depth exceeds the threshold.",
            "uncertainty_width_map": "FGNO 95th minus 5th percentile width of memberwise maximum water depth.",
            "reference_contour": "HEC-RAS mean maximum water-depth contour at the same threshold.",
            "inundated_area_ratio": "Wettable-domain mesh-cell share exceeding the threshold at each lead time; this is used as an inundated-area ratio proxy on the near-uniform evaluated mesh.",
            "rainfall_rate": "The archived precipitation series stores 15-minute rainfall depth; plotted rainfall rate is depth divided by 0.25 h.",
            "coverage90_time": "Fraction of evaluated cells where HEC-RAS mean depth lies inside the FGNO 5th-95th percentile interval.",
        },
    }
    (out_dir / H1_METADATA_NAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote %s" % (out_dir / H1_BUNDLE_NAME))
    print("Wrote %s" % (out_dir / H1_METADATA_NAME))


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


def _alpha_cmap(name, stops):
    import matplotlib.colors as mcolors

    return mcolors.LinearSegmentedColormap.from_list(name, stops)


def _probability_cmap():
    return _alpha_cmap(
        "fgno_probability",
        [
            (0.00, "#7BE4F0"),
            (0.28, "#1FB7C9"),
            (0.58, "#08799E"),
            (0.80, "#0B3F78"),
            (1.00, "#071B4D"),
        ],
    )


def _width_cmap():
    return _alpha_cmap(
        "fgno_width",
        [
            (0.00, "#D6C2FF"),
            (0.30, "#A879E6"),
            (0.58, "#7B3FC1"),
            (0.80, "#4C168A"),
            (1.00, "#23004A"),
        ],
    )


def _style_rc():
    import matplotlib.pyplot as plt

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
            "axes.linewidth": 0.8,
        }
    )


def _format_forcing_axis(ax, ax_rain, col_idx, n_cols):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax_rain.spines["top"].set_visible(False)
    ax.grid(True, axis="both", color="#D7DBE7", linewidth=0.55, alpha=0.65)
    ax.tick_params(axis="both", labelsize=9, colors="#303746", length=2)
    ax_rain.tick_params(axis="y", labelsize=9, colors="#6F768A", length=2)
    ax.set_xlabel("Lead time (h)", fontsize=10)
    if col_idx == 0:
        ax.set_ylabel("Stage (m)", fontsize=10)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    if col_idx == n_cols - 1:
        ax_rain.set_ylabel("Rainfall rate\n(mm h$^{-1}$)", fontsize=10, color="#6F768A")
    else:
        ax_rain.set_ylabel("")
        ax_rain.tick_params(axis="y", labelright=False)


def _draw_basemap(ax, tri, x, y, elevation, basemap_image, basemap_extent, alpha, dem_vmin=-15.1, dem_vmax=19.9):
    if basemap_image is not None and basemap_extent is not None:
        ax.imshow(basemap_image, extent=basemap_extent, origin="upper", alpha=alpha, zorder=0)
    else:
        elev = np.asarray(elevation, dtype=np.float64)
        finite = elev[np.isfinite(elev)]
        if finite.size == 0:
            return
        lo = float(dem_vmin) if dem_vmin is not None else float(np.nanquantile(finite, 0.01))
        hi = float(dem_vmax) if dem_vmax is not None else float(np.nanquantile(finite, 0.99))
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
            alpha=alpha,
            zorder=0,
        )


def _format_map_axis(ax, x, y):
    ax.set_aspect("equal")
    pad_x = 0.025 * (np.nanmax(x) - np.nanmin(x))
    pad_y = 0.025 * (np.nanmax(y) - np.nanmin(y))
    ax.set_xlim(np.nanmin(x) - pad_x, np.nanmax(x) + pad_x)
    ax.set_ylim(np.nanmin(y) - pad_y, np.nanmax(y) + pad_y)
    ax.axis("off")


def _plot_overlay_map(
    ax,
    tri,
    x,
    y,
    elevation,
    field,
    cmap,
    vmin,
    vmax,
    basemap_image,
    basemap_extent,
    ref_max,
    threshold,
    basemap_alpha,
    clip_below=None,
):
    import matplotlib.tri as mtri

    _draw_basemap(ax, tri, x, y, elevation, basemap_image, basemap_extent, basemap_alpha)
    arr = np.asarray(field, dtype=np.float64).copy()
    triangles = np.asarray(tri.triangles)
    tri_vals = arr[triangles]
    with np.errstate(invalid="ignore"):
        face_values = np.nanmean(tri_vals, axis=1)
        face_max = np.nanmax(tri_vals, axis=1)
    base_mask = tri.mask if tri.mask is not None else np.zeros(len(triangles), dtype=bool)
    overlay_mask = np.array(base_mask, dtype=bool, copy=True)
    overlay_mask |= ~np.isfinite(face_values)
    if clip_below is not None:
        # Match the water-depth rendering convention: cells/triangles whose
        # signal is effectively zero are not drawn, leaving the DEM visible.
        overlay_mask |= ~np.isfinite(face_max) | (face_max < float(clip_below))
        face_values = np.maximum(face_values, float(clip_below))
    face_values = np.clip(face_values, float(vmin), float(vmax))

    overlay_tri = mtri.Triangulation(x, y, triangles=triangles, mask=overlay_mask)
    cmap = copy.copy(cmap)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    artist = ax.tripcolor(
        overlay_tri,
        facecolors=face_values,
        shading="flat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors="none",
        linewidth=0.0,
        rasterized=True,
        alpha=0.90,
        zorder=2,
    )
    finite_ref = ref_max[np.isfinite(ref_max)]
    if finite_ref.size and np.nanmin(finite_ref) < threshold < np.nanmax(finite_ref):
        import matplotlib.patheffects as pe

        contour = ax.tricontour(
            tri,
            ref_max,
            levels=[threshold],
            colors=["#FFFFFF"],
            linewidths=1.05,
            linestyles=["-"],
            zorder=4,
        )
        import warnings
        from matplotlib import MatplotlibDeprecationWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)
            contour_collections = list(getattr(contour, "collections", []))
        for coll in contour_collections:
            coll.set_path_effects([pe.Stroke(linewidth=2.35, foreground="#101820"), pe.Normal()])
    _format_map_axis(ax, x, y)
    return artist


def _plot_h1(args):
    """Plot the compact historical H1 figure.

    Layout: one event per row and four diagnostic columns. This matches the
    manuscript-oriented H1 design and keeps labels concise enough for print.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import matplotlib.ticker as mticker
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    _style_rc()
    out_dir = Path(args.out_dir)
    data = np.load(str(out_dir / H1_BUNDLE_NAME), allow_pickle=True)
    metadata = json.loads((out_dir / H1_METADATA_NAME).read_text(encoding="utf-8"))
    threshold = float(metadata["threshold_m"])

    geometry = data["geometry"]
    x = geometry[:, 0]
    y = geometry[:, 1]
    elevation = data["elevation"]
    tri = _build_triangulation(x, y)

    basemap_image = None
    basemap_extent = None
    if args.basemap_context:
        basemap = np.load(str(args.basemap_context))
        basemap_image = basemap["image"]
        basemap_extent = basemap["extent"]

    prob = data["prob_exceed_max"]
    width = data["width90_max"]
    prob_clip = float(args.prob_clip)
    width_clip = float(args.width_clip)
    width_vmax = float(args.width_vmax)
    if width_vmax <= 0:
        visible_width = width[np.isfinite(width) & (width >= width_clip)]
        width_vmax = max(0.20, float(np.nanquantile(visible_width, 0.985))) if visible_width.size else 1.0
    width_vmax = min(width_vmax, 1.0)

    n_events = len(metadata["events"])
    fig = plt.figure(figsize=(16.2, 9.1), dpi=300)
    gs = fig.add_gridspec(
        nrows=n_events + 1,
        ncols=4,
        width_ratios=[1.52, 1.0, 1.0, 1.18],
        height_ratios=[1.0] * n_events + [0.12],
        left=0.125,
        right=0.985,
        top=0.940,
        bottom=0.115,
        wspace=0.165,
        hspace=0.185,
    )
    axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(n_events)]
    cax_prob = fig.add_subplot(gs[n_events, 1])
    cax_width = fig.add_subplot(gs[n_events, 2])

    col_titles = [
        "Historical forcing",
        "FGNO exceedance probability",
        "FGNO 90% interval width",
        "Inundated-area ratio trajectory",
    ]
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=12.5, fontweight="bold", pad=8)

    prob_cmap = _probability_cmap()
    width_cmap = _width_cmap()
    display_names = [event["display_name"] for event in metadata["events"]]
    roles = [event["role"] for event in metadata["events"]]

    ratio_ylim = min(
        1.0,
        max(
            0.05,
            1.12
            * float(
                np.nanmax(
                    [
                        np.nanmax(data["pred_frac_p95"]),
                        np.nanmax(data["ref_frac"]),
                    ]
                )
            ),
        ),
    )

    for i in range(n_events):
        t = data["boundary_time"][i]
        stage = data["stage_series"][i]
        # Stored precipitation is 15-minute depth; convert to rate for plotting.
        precip_rate = data["precip_series"][i] * 4.0

        # Event label at the far left, outside the forcing axis.
        pos = axes[i][0].get_position()
        fig.text(
            pos.x0 - 0.050,
            0.5 * (pos.y0 + pos.y1),
            display_names[i],
            ha="right",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            linespacing=1.15,
        )

        ax = axes[i][0]
        ax_rain = ax.twinx()
        rain_width = np.median(np.diff(t)) * 0.82 if t.size > 1 else 0.2
        ax.plot(t, stage, color="#1F4E79", linewidth=1.45, zorder=3)
        ax_rain.bar(
            t,
            precip_rate,
            width=rain_width,
            color="#C56A3B",
            edgecolor="#7E3E22",
            linewidth=0.18,
            alpha=0.68,
            zorder=2,
        )
        ax_rain.invert_yaxis()
        ax.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        ax.set_ylim(max(0.0, float(np.nanmin(stage)) - 0.10), float(np.nanmax(stage)) + 0.12)
        rain_hi = max(float(np.nanmax(precip_rate)) * 1.18, 0.1)
        ax_rain.set_ylim(rain_hi, 0.0)
        _format_forcing_axis(ax, ax_rain, 0, 1)
        ax.set_xlabel("Lead time (h)", fontsize=9.5)
        ax.set_ylabel("Stage (m)", fontsize=9.5)
        ax_rain.set_ylabel("Rainfall rate\n(mm/hr)", fontsize=9.5, color="#6F768A")
        if i < n_events - 1:
            ax.set_xlabel("")
        if i == 0:
            handles = [
                Line2D([0], [0], color="#1F4E79", linewidth=1.45, label="Stage"),
                Patch(facecolor="#C56A3B", edgecolor="#7E3E22", alpha=0.68, label="Rainfall"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=8.5, frameon=False, handlelength=1.7)

        _plot_overlay_map(
            axes[i][1],
            tri,
            x,
            y,
            elevation,
            prob[i],
            prob_cmap,
            prob_clip,
            1.0,
            basemap_image,
            basemap_extent,
            data["ref_max_depth"][i],
            threshold,
            0.92,
            clip_below=prob_clip,
        )
        if i == 0:
            contour_handle = Line2D([0], [0], color="#FFFFFF", linewidth=1.25, label="HEC-RAS contour")
            contour_handle.set_path_effects([pe.Stroke(linewidth=2.8, foreground="#101820"), pe.Normal()])
            axes[i][1].legend(
                handles=[contour_handle],
                loc="lower left",
                bbox_to_anchor=(0.02, 0.02),
                frameon=True,
                framealpha=0.78,
                facecolor="#FFFFFF",
                edgecolor="#D7DBE7",
                fontsize=8.0,
                handlelength=2.0,
                borderpad=0.25,
                labelspacing=0.25,
            )
        _plot_overlay_map(
            axes[i][2],
            tri,
            x,
            y,
            elevation,
            width[i],
            width_cmap,
            width_clip,
            width_vmax,
            basemap_image,
            basemap_extent,
            data["ref_max_depth"][i],
            threshold,
            0.92,
            clip_below=width_clip,
        )

        ax_ts = axes[i][3]
        p05 = data["pred_frac_p05"][i]
        p50 = data["pred_frac_p50"][i]
        p95 = data["pred_frac_p95"][i]
        ref_frac = data["ref_frac"][i]
        ax_ts.fill_between(t, p05, p95, color="#94D7BF", alpha=0.42, linewidth=0, label="FGNO 5-95%")
        ax_ts.plot(t, p50, color="#0B8A5A", linewidth=1.55, label="FGNO median")
        ax_ts.plot(t, ref_frac, color="#1F2430", linewidth=1.15, linestyle="--", label="HEC-RAS")
        ax_ts.set_xlim(float(np.nanmin(t)), float(np.nanmax(t)))
        ax_ts.set_ylim(0.0, ratio_ylim)
        ax_ts.grid(True, color="#E6E8F0", linewidth=0.58, alpha=0.75)
        ax_ts.tick_params(axis="both", labelsize=8.7, length=2)
        ax_ts.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax_ts.set_xlabel("Lead time (h)", fontsize=9.5)
        ax_ts.set_ylabel("Inundated-area ratio > %.1f m" % threshold, fontsize=9.5)
        if i < n_events - 1:
            ax_ts.set_xlabel("")
        if i == 0:
            ax_ts.legend(loc="upper left", fontsize=8.5, frameon=False, handlelength=1.65)
        for spine in ax_ts.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_color("#D7DBE7")

    # Shared colorbars under the two map columns.
    prob_sm = ScalarMappable(norm=Normalize(vmin=prob_clip, vmax=1.0), cmap=prob_cmap)
    prob_sm.set_array([])
    cbar_prob = fig.colorbar(prob_sm, cax=cax_prob, orientation="horizontal")
    cbar_prob.set_label("P(max water depth > %.1f m)" % threshold, fontsize=9.3, labelpad=2)
    cbar_prob.ax.tick_params(labelsize=8.5, length=2)

    width_sm = ScalarMappable(norm=Normalize(vmin=width_clip, vmax=width_vmax), cmap=width_cmap)
    width_sm.set_array([])
    cbar_width = fig.colorbar(width_sm, cax=cax_width, orientation="horizontal")
    cbar_width.set_label("90%% interval width (m)", fontsize=9.3, labelpad=2)
    cbar_width.ax.tick_params(labelsize=8.5, length=2)

    for ext in ("png", "pdf", "svg"):
        out_path = out_dir / ("%s.%s" % (H1_PREFIX, ext))
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.035)
        print("Wrote %s" % out_path)
    plt.close(fig)


def _read_h2_rows(path):
    with open(str(path), newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: r["event"])
    return rows


def _metric_values(rows, method_key, metric):
    return np.asarray([float(r["%s__%s" % (method_key, metric)]) for r in rows], dtype=np.float64)


def _box_strip_panel(ax, rows, metric, ylabel, panel_label, better_text, transform=None, ylim=None, ref_line=None):
    rng = np.random.default_rng(42)
    positions = np.arange(len(METHODS), dtype=float)
    values_by_method = []
    colors = []
    labels = []
    for method_key, method_name, color, _marker in METHODS:
        vals = _metric_values(rows, method_key, metric)
        if transform is not None:
            vals = transform(vals)
        values_by_method.append(vals)
        colors.append(color)
        labels.append(method_name.replace("MC-dropout GINO", "MC-dropout\nGINO").replace("Denoising Diffusion Operator", "Denoising\nDiffusion Operator"))

    bp = ax.boxplot(
        values_by_method,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.25},
        whiskerprops={"color": "#4B5563", "linewidth": 0.85},
        capprops={"color": "#4B5563", "linewidth": 0.85},
        boxprops={"edgecolor": "#374151", "linewidth": 0.85},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)

    for x0, vals, color in zip(positions, values_by_method, colors):
        jitter = rng.uniform(-0.105, 0.105, size=len(vals))
        ax.scatter(
            np.full_like(vals, x0, dtype=float) + jitter,
            vals,
            s=20,
            facecolor=color,
            edgecolor="#1F2430",
            linewidth=0.25,
            alpha=0.86,
            zorder=3,
        )
        med = float(np.nanmedian(vals))
        ax.text(x0, med, "  %.3f" % med, va="center", ha="left", fontsize=8.4, color="#1F2430")

    if ref_line is not None:
        ax.axhline(ref_line, color="#1F2430", linestyle="--", linewidth=1.0, alpha=0.75, zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9.2)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title("(%s) %s" % (panel_label, ylabel), fontsize=12.5, fontweight="bold", loc="left", pad=8)
    if better_text:
        ax.text(
            0.985,
            0.955,
            better_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.8,
            color="#6F768A",
        )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", color="#E6E8F0", linewidth=0.7, alpha=0.85)
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=9.2, length=2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#6B7280")


def _plot_h2(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style_rc()
    out_dir = _ensure_dir(args.out_dir)
    rows = _read_h2_rows(args.metrics_csv)

    # Save the exact source rows used by this figure.
    with open(str(out_dir / H2_SOURCE_NAME), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.1), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.835, bottom=0.105, wspace=0.170, hspace=0.300)

    _box_strip_panel(
        axes[0, 0],
        rows,
        "Fair_CRPS_m",
        "CRPS (m)",
        "a",
        "lower is better",
        ylim=(0.0, 0.17),
    )
    _box_strip_panel(
        axes[0, 1],
        rows,
        "Brier_0.3m",
        "Brier score at 0.3 m",
        "b",
        "lower is better",
        ylim=(0.0, 0.075),
    )
    _box_strip_panel(
        axes[1, 0],
        rows,
        "CSI_0.3m",
        "CSI at 0.3 m",
        "c",
        "higher is better",
        ylim=(0.45, 1.0),
    )
    _box_strip_panel(
        axes[1, 1],
        rows,
        "Coverage_90",
        "90% coverage error",
        "d",
        "lower is better",
        transform=lambda v: np.abs(v - 0.90),
        ylim=(0.0, 0.90),
        ref_line=0.0,
    )

    fig.text(
        0.075,
        0.966,
        "Historical-event probabilistic skill across 13 hindcasts",
        ha="left",
        va="top",
        fontsize=15.5,
        fontweight="bold",
        color="#1F2430",
    )
    fig.text(
        0.075,
        0.932,
        (
            "Boxes show the median and interquartile range across events; points are individual historical hindcasts. "
            "Coverage error is the absolute deviation from nominal 90% interval coverage."
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        color="#6F768A",
    )

    for ext in ("png", "pdf", "svg"):
        out_path = out_dir / ("%s.%s" % (H2_PREFIX, ext))
        fig.savefig(str(out_path), dpi=300, bbox_inches="tight", pad_inches=0.04)
        print("Wrote %s" % out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    p_extract = sub.add_parser("extract-h1")
    p_extract.add_argument("--artifact-root", required=True)
    p_extract.add_argument("--out-dir", required=True)
    p_extract.add_argument("--threshold-m", type=float, default=0.30)
    p_extract.set_defaults(func=extract_h1)

    p_h1 = sub.add_parser("plot-h1")
    p_h1.add_argument("--out-dir", required=True)
    p_h1.add_argument("--basemap-context", default=None)
    p_h1.add_argument("--width-vmax", type=float, default=0.0)
    p_h1.add_argument("--prob-clip", type=float, default=0.10)
    p_h1.add_argument("--width-clip", type=float, default=0.08)
    p_h1.set_defaults(func=_plot_h1)

    p_h2 = sub.add_parser("plot-h2")
    p_h2.add_argument("--metrics-csv", required=True)
    p_h2.add_argument("--out-dir", required=True)
    p_h2.set_defaults(func=_plot_h2)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
