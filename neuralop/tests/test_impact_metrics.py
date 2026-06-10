import json

import numpy as np

from neuralop.flood.eval.impact_metrics import (
    arrival_times,
    compute_flood_impact_crps_metrics,
    ensemble_crps_scalar,
    inundated_area_series,
    normalize_impact_metrics_config,
    peak_inundated_area_series,
    radius_neighborhoods,
)
from neuralop.flood.eval.render import _save_nonspatial_uq_diagnostics
from neuralop.flood.eval.runtime import UQ_IMPACT_CRPS_PNG, UQ_OVERALL_JSON


class _Logger:
    def info(self, *args, **kwargs):
        return None


def test_ensemble_crps_scalar_matches_hand_computed_value():
    # Fair CRPS uses 2K(K-1) for the forecast self-distance term.
    # term1 = (|1-4| + |2-4|) / 2 = 2.5; term2 = 2 / (2*2*1) = 0.5.
    assert np.isclose(ensemble_crps_scalar(np.array([1.0, 2.0]), np.array([4.0])), 2.0)


def test_inundated_area_uses_nonuniform_cell_areas():
    wd = np.array(
        [
            [[0.0, 0.2, 0.3], [0.2, 0.0, 0.0]],
            [[0.2, 0.2, 0.0], [0.0, 0.2, 0.3]],
        ],
        dtype=np.float64,
    )
    area = np.array([2.0, 3.0, 5.0], dtype=np.float64)

    out = inundated_area_series(wd, area, threshold_m=0.10)

    assert np.allclose(out, [[8.0, 2.0], [5.0, 8.0]])


def test_peak_inundated_area_accumulates_over_time():
    area_series = np.array([[2.0, 5.0], [7.0, 4.0], [3.0, 9.0]])

    assert np.allclose(
        peak_inundated_area_series(area_series),
        [[2.0, 5.0], [7.0, 5.0], [7.0, 9.0]],
    )


def test_arrival_times_include_never_wet_cells():
    wd = np.array(
        [
            [[0.0, 0.0], [0.2, 0.0]],
            [[0.2, 0.0], [0.0, 0.0]],
            [[0.3, 0.0], [0.0, 0.2]],
        ],
        dtype=np.float64,
    )

    out = arrival_times(wd, threshold_m=0.10, never_time=4.0)

    assert np.allclose(out, [[2.0, 4.0], [1.0, 3.0]])


def test_radius_neighborhoods_respect_active_metric_geometry():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float64)

    neighborhoods = radius_neighborhoods(coords, radius_m=1.5)

    assert [set(nbrs.tolist()) for nbrs in neighborhoods] == [{0, 1}, {0, 1}, {2}]


def test_compute_flood_impact_metrics_distinguishes_average_and_max_pooling():
    pred = np.array([[[0.0, 2.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float64)
    ref = np.array([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    static = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    cfg = {"inundation_threshold_m": 0.10, "pooled_radii_m": [1.5]}

    metrics = compute_flood_impact_crps_metrics(pred, ref, geometry, static, None, cfg)

    assert np.isclose(metrics["pooled_avg_crps_wd_r1p5"][0], 0.0)
    assert metrics["pooled_max_crps_wd_r1p5"][0] > metrics["pooled_avg_crps_wd_r1p5"][0]
    assert "crps_total_inundated_area_wd" in metrics
    assert "crps_total_inundated_area_fraction_wd" in metrics
    assert "crps_peak_inundated_area_wd" in metrics
    assert "crps_peak_inundated_area_fraction_wd" in metrics
    assert np.ndim(metrics["crps_arrival_time_wd"]) == 0
    assert np.ndim(metrics["crps_arrival_time_fraction_wd"]) == 0


def test_compute_flood_impact_metrics_respects_wettable_mask():
    pred = np.array([[[0.0, 2.0, 2.0], [2.0, 0.0, 2.0]]], dtype=np.float64)
    ref = np.array([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]], dtype=np.float64)
    geometry = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=np.float64)
    static = np.array([[0.0, 2.0], [0.0, 3.0], [0.0, 100.0]], dtype=np.float64)
    mask = np.array([True, True, False])
    cfg = {"inundation_threshold_m": 0.10, "pooled_radii_m": [1.5]}

    masked = compute_flood_impact_crps_metrics(pred, ref, geometry, static, mask, cfg)
    unmasked = compute_flood_impact_crps_metrics(pred, ref, geometry, static, None, cfg)

    assert masked["crps_total_inundated_area_wd"].shape == (1,)
    assert unmasked["crps_total_inundated_area_wd"][0] > masked["crps_total_inundated_area_wd"][0]


def test_compute_flood_impact_metrics_adds_dimensionless_normalized_scores():
    pred = np.array(
        [
            [[0.2, 0.0, 0.0], [0.2, 0.2, 0.0]],
            [[0.2, 0.2, 0.0], [0.2, 0.2, 0.2]],
        ],
        dtype=np.float64,
    )
    ref = np.array(
        [
            [[0.2, 0.2, 0.0], [0.0, 0.0, 0.0]],
            [[0.2, 0.2, 0.2], [0.2, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    geometry = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]], dtype=np.float64)
    static = np.array([[0.0, 2.0], [0.0, 3.0], [0.0, 5.0]], dtype=np.float64)
    cfg = {"inundation_threshold_m": 0.10, "pooled_radii_m": [150.0]}

    metrics = compute_flood_impact_crps_metrics(pred, ref, geometry, static, None, cfg)

    domain_area = 10.0
    horizon_scale = 3.0
    assert np.allclose(
        metrics["crps_total_inundated_area_fraction_wd"],
        metrics["crps_total_inundated_area_wd"] / domain_area,
    )
    assert np.allclose(
        metrics["crps_peak_inundated_area_fraction_wd"],
        metrics["crps_peak_inundated_area_wd"] / domain_area,
    )
    assert np.isclose(
        metrics["crps_arrival_time_fraction_wd"],
        metrics["crps_arrival_time_wd"] / horizon_scale,
    )
    assert 0.0 <= metrics["crps_arrival_time_fraction_wd"] <= 1.0


def test_normalize_impact_config_defaults_and_disable_flag():
    cfg = normalize_impact_metrics_config({"enabled": "false"})

    assert cfg.enabled is False
    assert np.isclose(cfg.inundation_threshold_m, 0.10)
    assert cfg.pooled_radii_m == (100.0, 250.0, 500.0, 1000.0)


def test_nonspatial_uq_diagnostics_save_scalar_arrival_and_impact_figure(tmp_path):
    metrics = {
        "crps_arrival_time_wd": np.array([1.0, 3.0], dtype=np.float64),
        "crps_arrival_time_fraction_wd": np.array([0.25, 0.75], dtype=np.float64),
        "crps_total_inundated_area_wd": np.array([[2.0, 4.0], [4.0, 6.0]], dtype=np.float64),
        "crps_total_inundated_area_fraction_wd": np.array(
            [[0.2, 0.4], [0.4, 0.6]], dtype=np.float64
        ),
        "crps_peak_inundated_area_wd": np.array([[3.0, 5.0], [5.0, 8.0]], dtype=np.float64),
        "crps_peak_inundated_area_fraction_wd": np.array(
            [[0.3, 0.5], [0.5, 0.8]], dtype=np.float64
        ),
        "pooled_avg_crps_wd_r100": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64),
        "pooled_max_crps_wd_r100": np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float64),
    }

    _save_nonspatial_uq_diagnostics(
        out_dir=str(tmp_path),
        time_hours=np.array([1.0, 2.0], dtype=np.float64),
        metrics=metrics,
        reliability_bins={},
        pit_hist_counts=np.zeros(2, dtype=np.float64),
        pit_edges=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        rank_hist_counts=np.zeros(2, dtype=np.float64),
        spread_skill_samples=np.empty((0, 2), dtype=np.float64),
        interval_coverage={},
        interval_width={},
        wasserstein_wd=None,
        logger=_Logger(),
    )

    payload = json.loads((tmp_path / UQ_OVERALL_JSON).read_text())
    assert np.isclose(payload["crps_arrival_time_wd_overall_mean"], 2.0)
    assert np.isclose(payload["crps_arrival_time_fraction_wd_overall_mean"], 0.5)
    assert np.isclose(payload["crps_total_inundated_area_fraction_wd_overall_mean"], 0.4)
    assert "crps_arrival_time_wd_leadtime_mean_last" not in payload
    assert (tmp_path / UQ_IMPACT_CRPS_PNG).exists()
