import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch

from neuralop.flood.eval import render
from neuralop.flood.eval.operator_app import _resolve_rollout_length_for_evaluation
from neuralop.flood.eval.render import (
    _cartographic_context,
    _diagnostic_boundary_panels,
    _mask_wd_dry_for_overlay,
    _save_animation_outputs,
    _save_hydrograph_uq_figures_and_animation,
)
from neuralop.flood.eval.rollout import _masked_relative_l2


def _tiny_rollout_fields(n_steps=3, n_cells=4):
    base = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ],
        dtype=np.float64,
    )[:n_steps, :n_cells]
    pred = base + 0.05
    return {
        "pred_mean_by_channel": {"wd": pred},
        "pred_std_by_channel": {"wd": np.full_like(pred, 0.02)},
        "gt_mean_by_channel": {"wd": base},
        "gt_std_by_channel": {"wd": np.full_like(pred, 0.01)},
    }


def test_diagnostic_boundary_panels_are_dataset_adaptive():
    assert _diagnostic_boundary_panels(["stage", "precipitation"], 2) == [
        (0, "line"),
        (1, "bar"),
    ]
    assert _diagnostic_boundary_panels(["inflow"], 1) == [(0, "line")]
    assert _diagnostic_boundary_panels(["stage"], 1) == [(0, "line")]


def test_forecast_horizon_hides_raw_rollout_start_offset():
    hours = render._forecast_horizon_hours(3, 1200.0, initial_history_steps=3)
    assert np.allclose(hours, [1.0, 4.0 / 3.0, 5.0 / 3.0])


def test_relative_l2_axis_uses_forecast_horizon_not_raw_spinup_time():
    fig, ax = render.plt.subplots()
    try:
        render._draw_relative_l2_axis(
            ax,
            relative_l2=np.array([0.1, 0.2, 0.3]),
            frame_idx=0,
            dt_seconds=1200.0,
            rollout_start_index=12,
            initial_history_steps=3,
        )
        xdata = ax.lines[0].get_xdata()
        assert np.allclose(xdata, [1.0, 4.0 / 3.0, 5.0 / 3.0])
        assert ax.get_xlabel() == "Forecast horizon (h)"
    finally:
        render.plt.close(fig)


def test_boundary_diagnostics_drop_skipped_spinup_steps():
    fig = render.plt.figure()
    try:
        gs = fig.add_gridspec(3, 3)
        boundary = np.array([[10.0], [11.0], [12.0], [13.0], [14.0]], dtype=np.float64)
        diag_axes = render._make_rollout_diagnostic_axes(
            fig,
            gs,
            boundary_series_raw=boundary,
            boundary_channel_names=["stage"],
        )
        render._draw_rollout_diagnostics(
            diag_axes=diag_axes,
            frame_idx=0,
            dt_seconds=1200.0,
            boundary_series_raw=boundary,
            boundary_channel_names=["stage"],
            relative_l2=np.array([0.1, 0.2, 0.3]),
            rollout_start_index=2,
            initial_history_steps=3,
        )
        boundary_ax = diag_axes["boundary_axes"][0][0]
        xdata = boundary_ax.lines[0].get_xdata()
        ydata = boundary_ax.lines[0].get_ydata()
        assert np.allclose(xdata, [1.0, 4.0 / 3.0, 5.0 / 3.0])
        assert np.allclose(ydata, [12.0, 13.0, 14.0])
    finally:
        render.plt.close(fig)


def test_boundary_diagnostics_plot_reference_forcing_ensemble():
    fig = render.plt.figure()
    try:
        gs = fig.add_gridspec(3, 3)
        backbone = np.array(
            [
                [1.0, 0.0],
                [1.1, 0.2],
                [1.2, 0.4],
                [1.3, 0.1],
                [1.1, 0.0],
            ],
            dtype=np.float64,
        )
        ensemble = np.array(
            [
                [[1.0, 0.0], [1.2, 0.1], [1.4, 0.4], [1.5, 0.2], [1.2, 0.0]],
                [[1.0, 0.0], [1.0, 0.3], [1.1, 0.6], [1.2, 0.1], [1.0, 0.0]],
                [[1.0, 0.0], [1.3, 0.2], [1.5, 0.5], [1.4, 0.3], [1.2, 0.0]],
            ],
            dtype=np.float64,
        )
        diag_axes = render._make_rollout_diagnostic_axes(
            fig,
            gs,
            boundary_series_raw=backbone,
            boundary_ensemble_series_raw=ensemble,
            boundary_channel_names=["stage", "precipitation"],
        )
        assert diag_axes["ensemble_series"].shape == (3, 5, 2)
        render._draw_rollout_diagnostics(
            diag_axes=diag_axes,
            frame_idx=1,
            dt_seconds=1200.0,
            boundary_series_raw=backbone,
            boundary_ensemble_series_raw=ensemble,
            boundary_channel_names=["stage", "precipitation"],
            relative_l2=np.array([0.1, 0.2, 0.3]),
            rollout_start_index=2,
            initial_history_steps=3,
        )
        stage_ax = diag_axes["boundary_axes"][0][0]
        precip_ax = diag_axes["boundary_axes"][1][0]
        assert any(line.get_label() == "GT forcing mean" for line in stage_ax.lines)
        assert any(line.get_label() == "Clean backbone" for line in stage_ax.lines)
        assert len(stage_ax.collections) >= 1
        assert len(precip_ax.patches) > 0
        stage_legend = stage_ax.get_legend()
        assert stage_legend is not None
        assert [text.get_text() for text in stage_legend.get_texts()] == [
            "GT forcing 5-95%",
            "GT forcing mean",
            "Clean backbone",
        ]
        legend_handles = getattr(stage_legend, "legend_handles", None)
        if legend_handles is None:
            legend_handles = stage_legend.legendHandles
        assert legend_handles[0].get_linewidth() >= 6.0
        precip_legend = precip_ax.get_legend()
        assert precip_legend is not None
        precip_handles = getattr(precip_legend, "legend_handles", None)
        if precip_handles is None:
            precip_handles = precip_legend.legendHandles
        assert precip_handles[0].get_facecolor()[-1] >= 0.30
    finally:
        render.plt.close(fig)


def test_visualization_map_mode_defaults_to_local_dem_elevation():
    opts = render._visualization_options({"map": {"enabled": True}})
    assert opts["mode"] == "dem_elevation"
    assert opts["provider"] == "local_elevation"
    assert opts["dem_cmap"] == "hecras_dem"
    assert opts["wd_colormap"] == "cyan_depth"
    assert opts["show_wet_edge"] is False
    assert opts["diagnostic_crps_colormap"] == "crps_indigo_alpha_ramp"


def test_visualization_map_mode_selects_mode_specific_default_providers():
    imagery = render._visualization_options({"map": {"mode": "imagery"}})
    topo = render._visualization_options({"map": {"mode": "topo"}})
    assert imagery["provider"] == "Esri.WorldImagery"
    assert topo["provider"] == "Esri.WorldTopoMap"


def test_visualization_map_mode_accepts_imagery_provider_override():
    opts = render._visualization_options({"map": {"mode": "imagery", "provider": "Esri.WorldImagery"}})
    assert opts["mode"] == "imagery"
    assert opts["provider"] == "Esri.WorldImagery"


def test_dem_elevation_mode_defaults_to_hecras_dem_and_cyan_depth():
    opts = render._visualization_options({"map": {"mode": "dem_elevation"}})
    assert opts["mode"] == "dem_elevation"
    assert opts["provider"] == "local_elevation"
    assert opts["dem_cmap"] == "hecras_dem"
    assert opts["wd_colormap"] == "cyan_depth"
    assert opts["basemap_alpha"] == 1.0
    assert opts["wd_overlay_alpha"] == 0.88


def test_cyan_depth_colormap_resolves_to_colormap_object():
    cmap = render._resolve_field_cmap("cyan_depth")
    assert cmap(0.0) != cmap(1.0)


def test_diagnostic_colormaps_resolve_to_distinct_low_competition_palettes():
    for name in ["error_rose", "spread_violet", "crps_indigo"]:
        cmap = render._resolve_field_cmap(name)
        assert cmap(0.0) != cmap(1.0)
        assert cmap(0.0)[3] == 0.0
        assert cmap(0.2)[3] < cmap(1.0)[3]


def test_near_zero_diagnostic_overlay_values_are_transparent_masked():
    arr = np.array([0.0, 1e-12, -1e-12, 1e-6, -1e-6, np.nan])
    masked = render._mask_near_zero_for_overlay(arr, threshold=1e-10)
    assert np.isnan(masked[0])
    assert np.isnan(masked[1])
    assert np.isnan(masked[2])
    assert np.isfinite(masked[3])
    assert np.isfinite(masked[4])
    assert np.isnan(masked[5])


def test_diagnostic_zero_threshold_combines_absolute_and_relative_cutoffs():
    opts = render._visualization_options({"diagnostics": {"zero_threshold": 1e-10, "zero_fraction": 0.05}})
    assert np.isclose(render._diagnostic_zero_threshold(opts, 2.0), 0.10)
    opts = render._visualization_options({"diagnostics": {"zero_threshold": 0.2, "zero_fraction": 0.05}})
    assert np.isclose(render._diagnostic_zero_threshold(opts, 2.0), 0.20)


def test_zero_transparent_panels_use_subdued_diagnostic_background(monkeypatch):
    calls = []

    def _fake_background(ax, x, y, context, renderer):
        calls.append(context)

    def _fake_plot_spatial_field(**kwargs):
        return object()

    monkeypatch.setattr(render, "_draw_cartographic_background", _fake_background)
    monkeypatch.setattr(render, "_plot_spatial_field", _fake_plot_spatial_field)
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    arr = np.array([0.0, 0.01, 0.10, 0.20])
    renderer = render._build_spatial_renderer(x, y, figsize=(3, 3), dpi=80, n_rows=1, n_cols=1)
    opts = render._visualization_options({"map": {"enabled": True, "mode": "dem_elevation"}, "diagnostics": {"basemap_alpha": 0.22}})
    fig, ax = render.plt.subplots()
    try:
        render._plot_spatial_panel(
            ax=ax,
            x=x,
            y=y,
            arr=arr,
            renderer=renderer,
            context={"mode": "dem_elevation", "options": opts, "elevation": arr},
            cmap="error_rose",
            vmin=0.0,
            vmax=0.20,
            zero_transparent=True,
        )
        assert np.isclose(calls[-1]["options"]["basemap_alpha"], 0.22)
    finally:
        render.plt.close(fig)


def test_zero_transparent_panels_do_not_flatten_colormap_alpha(monkeypatch):
    calls = []

    def _fake_plot_spatial_field(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(render, "_plot_spatial_field", _fake_plot_spatial_field)
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    arr = np.array([0.0, 0.01, 0.10, 0.20])
    renderer = render._build_spatial_renderer(x, y, figsize=(3, 3), dpi=80, n_rows=1, n_cols=1)
    fig, ax = render.plt.subplots()
    try:
        render._plot_spatial_panel(
            ax=ax,
            x=x,
            y=y,
            arr=arr,
            renderer=renderer,
            context={"mode": "none", "options": render._visualization_options({"map": {"enabled": False}})},
            cmap="error_rose",
            vmin=0.0,
            vmax=0.20,
            zero_transparent=True,
        )
        assert calls[-1]["alpha"] is None
    finally:
        render.plt.close(fig)


def test_usgs_3dep_provider_uses_arcgis_export_not_xyz_tiles():
    opts = render._visualization_options({"map": {"mode": "3dep_hillshade"}})
    source = render._resolve_basemap_source(ctx=object(), options=opts)
    url = render._arcgis_export_url(source, (-8499797.5, 4415002.7, -8497351.6, 4417448.7), 512)
    assert source.endswith("/USGSShadedReliefOnly/MapServer")
    assert "/export?" in url
    assert "/tile/" not in url
    assert "bbox=" in url
    assert "size=512%2C512" in url


def test_3dep_hillshade_defaults_to_stronger_alpha_than_tile_basemaps():
    hillshade = render._visualization_options({"map": {"mode": "3dep_hillshade"}})
    imagery = render._visualization_options({"map": {"mode": "imagery"}})
    assert hillshade["basemap_alpha"] > imagery["basemap_alpha"]
    assert hillshade["basemap_alpha"] == 0.65


def test_hillshade_enhancement_makes_low_contrast_3dep_export_legible():
    raw = np.linspace(0.933, 0.976, 100, dtype=np.float32).reshape(10, 10)
    raw_rgb = np.repeat(raw[:, :, None], 3, axis=2)
    enhanced = render._enhance_hillshade_rgb(raw_rgb, cmap_name="copper", tint_strength=0.45)
    assert enhanced.std() > raw_rgb.std() * 5.0
    assert enhanced.min() <= 0.51
    assert enhanced.max() >= 0.97
    assert not np.allclose(enhanced[..., 0], enhanced[..., 1])


def test_masked_relative_l2_uses_domain_norm_and_mask():
    pred = np.array([2.0, 100.0, 3.0])
    gt = np.array([1.0, 100.0, 1.0])
    wettable = np.array([True, False, True])
    expected = np.sqrt((1.0 ** 2) + (2.0 ** 2)) / np.sqrt((1.0 ** 2) + (1.0 ** 2))
    assert np.isclose(_masked_relative_l2(pred, gt, wettable), expected)


class _ConfigSection:
    pass


def _rollout_length_config(configured=78):
    cfg = _ConfigSection()
    cfg.data = _ConfigSection()
    cfg.data.rollout_length = configured
    return cfg


class _ListLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(message % args if args else message)

    def warning(self, message, *args):
        self.messages.append(message % args if args else message)


def test_rollout_evaluation_resolves_minus_one_to_full_available_timeseries():
    dataset = _ConfigSection()
    dataset.available_rollout_length = 85
    logger = _ListLogger()
    assert _resolve_rollout_length_for_evaluation(
        _rollout_length_config(configured=-1),
        dataset,
        logger,
    ) == 85
    assert any("full available rollout horizon=85" in msg for msg in logger.messages)


def test_rollout_evaluation_keeps_positive_configured_horizon():
    dataset = _ConfigSection()
    dataset.available_rollout_length = 85
    logger = _ListLogger()
    assert _resolve_rollout_length_for_evaluation(
        _rollout_length_config(configured=78),
        dataset,
        logger,
    ) == 78


def test_hydrograph_animation_writes_coastal_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ANIMATION_FPS", 1)
    monkeypatch.setattr(render, "ANIMATION_INTERVAL_MS", 20)
    monkeypatch.setattr(render, "PUBLICATION_TIMESTEPS", [0])
    geometry = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    fields = _tiny_rollout_fields()
    boundary = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.2],
            [1.3, 0.5],
            [1.2, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )

    _save_hydrograph_uq_figures_and_animation(
        geometry=geometry,
        target_variables=["wd"],
        out_dir=str(tmp_path),
        hydrograph_id="coastal_demo",
        dt_seconds=1200.0,
        n_ref_sims=2,
        n_ens=2,
        boundary_series_raw=boundary,
        boundary_channel_names=["stage", "precipitation"],
        relative_l2_by_channel={"wd": np.array([0.1, 0.2, 0.15])},
        crps_map_wd=np.full_like(fields["gt_mean_by_channel"]["wd"], 0.012),
        rollout_start_index=2,
        visualization_config={"map": {"enabled": False}, "output": {"write_mp4": False}},
        **fields,
    )

    gif = tmp_path / "uq_figures_per_hydrograph" / "uq_rollout_coastal_demo.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0


def test_hydrograph_animation_writes_dynamic_inflow_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr(render, "ANIMATION_FPS", 1)
    monkeypatch.setattr(render, "ANIMATION_INTERVAL_MS", 20)
    monkeypatch.setattr(render, "PUBLICATION_TIMESTEPS", [0])
    geometry = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    fields = _tiny_rollout_fields()
    boundary = np.array([[20.0], [22.0], [25.0], [23.0], [21.0]], dtype=np.float64)

    _save_hydrograph_uq_figures_and_animation(
        geometry=geometry,
        target_variables=["wd"],
        out_dir=str(tmp_path),
        hydrograph_id="dynamic_demo",
        dt_seconds=1200.0,
        n_ref_sims=2,
        n_ens=2,
        boundary_series_raw=boundary,
        boundary_channel_names=["inflow"],
        relative_l2_by_channel={"wd": np.array([0.1, 0.2, 0.15])},
        rollout_start_index=2,
        visualization_config={"map": {"enabled": False}, "output": {"write_mp4": False}},
        **fields,
    )

    gif = tmp_path / "uq_figures_per_hydrograph" / "uq_rollout_dynamic_demo.gif"
    assert gif.exists()
    assert gif.stat().st_size > 0


class _FakeAnimation:
    def __init__(self):
        self.saved = []

    def save(self, path, **kwargs):
        self.saved.append((path, kwargs))


def test_mask_wd_dry_for_overlay_uses_configured_threshold():
    arr = np.array([0.0, 0.03, 0.05, 0.20, np.nan])
    masked = _mask_wd_dry_for_overlay(arr, 0.05)
    assert np.isnan(masked[0])
    assert np.isnan(masked[1])
    assert np.isfinite(masked[2])
    assert np.isfinite(masked[3])
    assert np.isnan(masked[4])


def test_wet_edge_can_be_disabled_from_visualization_config():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    arr = np.array([0.0, 0.1, 0.2, 0.3])
    renderer = render._build_spatial_renderer(x, y, figsize=(3, 3), dpi=80, n_rows=1, n_cols=1)
    fig, ax = render.plt.subplots()
    try:
        _, edge_artists = render._plot_spatial_panel(
            ax=ax,
            x=x,
            y=y,
            arr=arr,
            renderer=renderer,
            context={
                "mode": "none",
                "options": render._visualization_options({
                    "map": {"enabled": False},
                    "wd": {"show_wet_edge": False},
                }),
            },
            cmap="viridis",
            vmin=0.0,
            vmax=0.3,
            is_wd_depth=True,
        )
        assert edge_artists == []
    finally:
        render.plt.close(fig)


def test_wet_edge_defaults_off_for_spatial_panels():
    opts = render._visualization_options({"map": {"enabled": False}})

    assert opts["show_wet_edge"] is False


def test_triangular_spatial_renderer_disables_mesh_edge_strokes():
    x = np.array([0.0, 1.0, 0.25, 0.8])
    y = np.array([0.0, 0.0, 0.9, 1.7])
    arr = np.array([0.0, 0.1, 0.2, 0.3])
    renderer = render._build_spatial_renderer(x, y, figsize=(3, 3), dpi=80, n_rows=1, n_cols=1)
    assert renderer["mode"] == "tri"

    fig, ax = render.plt.subplots()
    try:
        artist = render._plot_spatial_field(
            ax=ax,
            x=x,
            y=y,
            arr=arr,
            renderer=renderer,
            cmap="viridis",
            vmin=0.0,
            vmax=0.3,
            alpha=0.8,
        )

        assert np.allclose(np.asarray(artist.get_linewidths(), dtype=float), 0.0)
        assert artist.get_edgecolors().size == 0
    finally:
        render.plt.close(fig)


def test_wd_spatial_panel_draws_wet_edge_when_threshold_crossed():
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    arr = np.array([0.0, 0.1, 0.2, 0.3])
    renderer = render._build_spatial_renderer(x, y, figsize=(3, 3), dpi=80, n_rows=1, n_cols=1)
    fig, ax = render.plt.subplots()
    try:
        collections_before = len(ax.collections)
        _, edge_artists = render._plot_spatial_panel(
            ax=ax,
            x=x,
            y=y,
            arr=arr,
            renderer=renderer,
            context={
                "mode": "none",
                "options": render._visualization_options({
                    "map": {"enabled": False},
                    "wd": {"show_wet_edge": True},
                }),
            },
            cmap="viridis",
            vmin=0.0,
            vmax=0.3,
            is_wd_depth=True,
        )
        collections_after = len(ax.collections)
        assert edge_artists
        assert collections_after > collections_before + 1
    finally:
        render._remove_artists(edge_artists)
        render.plt.close(fig)


def test_dem_elevation_mode_uses_local_elevation_without_external_fetch(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _raise(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("external tiles should not be requested for dem_elevation mode")

    monkeypatch.setattr(render, "_try_build_external_basemap", _raise)
    context = _cartographic_context(
        x=np.array([379000.0, 379100.0, 379000.0, 379100.0]),
        y=np.array([4072000.0, 4072000.0, 4072100.0, 4072100.0]),
        elevation_raw=np.array([1.0, 2.0, np.nan, 4.0]),
        out_dir=str(tmp_path),
        visualization_config={"map": {"enabled": True, "mode": "dem_elevation"}},
    )
    assert context["mode"] == "dem_elevation"
    assert calls["n"] == 0


def test_elevation_hillshade_mode_skips_external_basemap_fetch(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _raise(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("external tiles should not be requested for elevation_hillshade mode")

    monkeypatch.setattr(render, "_try_build_external_basemap", _raise)
    context = _cartographic_context(
        x=np.array([379000.0, 379100.0, 379000.0, 379100.0]),
        y=np.array([4072000.0, 4072000.0, 4072100.0, 4072100.0]),
        elevation_raw=np.array([1.0, 2.0, 3.0, 4.0]),
        out_dir=str(tmp_path),
        visualization_config={"map": {"enabled": True, "mode": "elevation_hillshade"}},
    )
    assert context["mode"] == "elevation_hillshade"
    assert calls["n"] == 0


def test_cartographic_context_falls_back_to_elevation_hillshade(tmp_path, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("tiles unavailable")

    monkeypatch.setattr(render, "_try_build_external_basemap", _raise)
    context = _cartographic_context(
        x=np.array([379000.0, 379100.0, 379000.0, 379100.0]),
        y=np.array([4072000.0, 4072000.0, 4072100.0, 4072100.0]),
        elevation_raw=np.array([1.0, 2.0, 3.0, 4.0]),
        out_dir=str(tmp_path),
        visualization_config={"map": {"enabled": True, "mode": "3dep_hillshade", "fallback": "elevation_hillshade"}},
    )
    assert context["mode"] == "elevation_hillshade"
    assert (tmp_path / "cartographic_context" / "basemap_metadata.json").exists()


def test_basemap_cache_reused_for_same_extent(tmp_path, monkeypatch):
    calls = {"n": 0}

    def _fake_basemap(*, x, y, out_dir, options):
        calls["n"] += 1
        extent = (float(np.min(x)) - 1.0, float(np.max(x)) + 1.0, float(np.min(y)) - 1.0, float(np.max(y)) + 1.0)
        cache_dir = tmp_path / "cartographic_context"
        cache_dir.mkdir(exist_ok=True)
        np.savez_compressed(cache_dir / "basemap_context.npz", image=np.zeros((2, 2, 3), dtype=np.float32), extent=np.asarray(extent))
        metadata = {
            "mode": "external_basemap",
            "map_mode": options["mode"],
            "provider": options["provider"],
            "crs": options["crs"],
        }
        (cache_dir / "basemap_metadata.json").write_text(render.json.dumps(metadata))
        return {"mode": "external_basemap", "image": np.zeros((2, 2, 3), dtype=np.float32), "extent": extent, "metadata": metadata}

    monkeypatch.setattr(render, "_try_build_external_basemap", _fake_basemap)
    x = np.array([0.0, 1.0, 0.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    cfg = {"map": {"enabled": True, "mode": "topo", "provider": "fake", "fallback": "none"}}
    first = _cartographic_context(x=x, y=y, elevation_raw=None, out_dir=str(tmp_path), visualization_config=cfg)
    second = _cartographic_context(x=x, y=y, elevation_raw=None, out_dir=str(tmp_path), visualization_config=cfg)
    assert first["mode"] == "external_basemap"
    assert second["mode"] == "external_basemap"
    assert calls["n"] == 1


def test_save_animation_outputs_writes_gif_and_mp4_when_enabled(monkeypatch, tmp_path):
    fake = _FakeAnimation()
    monkeypatch.setattr(render.animation.writers, "is_available", lambda writer: writer == "ffmpeg")
    outputs = _save_animation_outputs(fake, str(tmp_path / "demo"), {"write_gif": True, "write_mp4": True})
    assert outputs["gif"].endswith(".gif")
    assert outputs["mp4"].endswith(".mp4")
    assert [path for path, _ in fake.saved] == [str(tmp_path / "demo.gif"), str(tmp_path / "demo.mp4")]
