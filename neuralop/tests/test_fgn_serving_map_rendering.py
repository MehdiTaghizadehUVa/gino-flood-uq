import pytest

from neuralop.flood.eval import render as eval_render
from neuralop.flood.serving.map_rendering import (
    SERVING_DEM_BASEMAP_ALPHA,
    SERVING_DEM_VMAX_M,
    SERVING_DEM_VMIN_M,
    SERVING_MAP_FACE_COLOR,
    serving_visualization_config,
)


def test_serving_visualization_defaults_match_publication_dem_style():
    cfg = serving_visualization_config()

    assert cfg["map"]["mode"] == "dem_elevation"
    assert cfg["map"]["dem_cmap"] == "hecras_dem"
    assert cfg["map"]["alpha"] == pytest.approx(SERVING_DEM_BASEMAP_ALPHA)
    assert cfg["map"]["dem_vmin"] == pytest.approx(SERVING_DEM_VMIN_M)
    assert cfg["map"]["dem_vmax"] == pytest.approx(SERVING_DEM_VMAX_M)
    assert cfg["diagnostics"]["basemap_alpha"] == pytest.approx(SERVING_DEM_BASEMAP_ALPHA)
    assert SERVING_MAP_FACE_COLOR == "#FFFFFF"

    options = eval_render._visualization_options(cfg)
    assert options["basemap_alpha"] == pytest.approx(SERVING_DEM_BASEMAP_ALPHA)
    assert options["diagnostic_basemap_alpha"] == pytest.approx(SERVING_DEM_BASEMAP_ALPHA)
    assert options["dem_vmin"] == pytest.approx(SERVING_DEM_VMIN_M)
    assert options["dem_vmax"] == pytest.approx(SERVING_DEM_VMAX_M)


def test_serving_visualization_merges_partial_run_metadata():
    cfg = serving_visualization_config(
        {
            "map": {"enabled": True, "mode": "dem_elevation"},
            "diagnostics": {"zero_fraction": 0.08},
        }
    )

    assert cfg["map"]["alpha"] == pytest.approx(SERVING_DEM_BASEMAP_ALPHA)
    assert cfg["map"]["dem_vmin"] == pytest.approx(SERVING_DEM_VMIN_M)
    assert cfg["map"]["dem_vmax"] == pytest.approx(SERVING_DEM_VMAX_M)
    assert cfg["diagnostics"]["basemap_alpha"] == pytest.approx(SERVING_DEM_BASEMAP_ALPHA)
    assert cfg["diagnostics"]["zero_fraction"] == pytest.approx(0.08)


def test_serving_visualization_respects_explicit_overrides():
    cfg = serving_visualization_config(
        {
            "map": {"alpha": 0.22, "dem_vmin": -2.0, "dem_vmax": 8.0},
            "diagnostics": {"basemap_alpha": 0.18},
            "wd": {"overlay_alpha": 0.7},
        }
    )

    assert cfg["map"]["alpha"] == pytest.approx(0.22)
    assert cfg["map"]["dem_vmin"] == pytest.approx(-2.0)
    assert cfg["map"]["dem_vmax"] == pytest.approx(8.0)
    assert cfg["diagnostics"]["basemap_alpha"] == pytest.approx(0.18)
    assert cfg["wd"]["overlay_alpha"] == pytest.approx(0.7)
