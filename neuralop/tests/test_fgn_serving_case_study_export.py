from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from neuralop.flood.serving.case_study_export import (
    CaseStudyRunProvenance,
    masked_triangle_face_values,
    paper_domain_viewport,
    select_showcase_frames,
    terrain_viewport_contains_mesh,
    validate_case_study_provenance,
)
from neuralop.flood.serving.case_study_rendering import (
    TerrainContext,
    probability_cmap,
    render_spatial_webp,
    render_validation_trajectory_svg,
)


def test_showcase_frames_span_horizon_and_keep_scientific_milestones():
    selected = select_showcase_frames(
        n_time=94,
        frame_count=32,
        required_indices=(0, 19, 46, 66, 93),
    )

    assert len(selected) == 32
    assert selected == sorted(set(selected))
    assert {0, 19, 46, 66, 93}.issubset(selected)


def test_map_faces_below_display_floor_remain_transparent():
    triangles = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    values = np.array([0.0, 0.04, 0.09, 0.40], dtype=np.float64)

    face_values, face_mask = masked_triangle_face_values(
        values=values,
        triangles=triangles,
        display_floor=0.10,
    )

    assert face_mask.tolist() == [True, False]
    assert np.isnan(face_values[0])
    assert face_values[1] == pytest.approx((0.04 + 0.09 + 0.40) / 3.0)


def test_case_study_rejects_mixed_production_provenance():
    runs = [
        CaseStudyRunProvenance(
            label="Irene 2011",
            bundle_id="coastal-v1",
            calibration_mode="calibrated_default",
            ensemble_count=3,
            members_per_ensemble=20,
            forecast_steps=94,
            dt_seconds=900,
            mesh_hash="mesh-a",
        ),
        CaseStudyRunProvenance(
            label="Isabel 2003",
            bundle_id="coastal-v1",
            calibration_mode="calibrated_default",
            ensemble_count=1,
            members_per_ensemble=100,
            forecast_steps=94,
            dt_seconds=900,
            mesh_hash="mesh-a",
        ),
    ]

    with pytest.raises(ValueError, match="mixed model or ensemble provenance"):
        validate_case_study_provenance(runs)


def test_case_study_terrain_viewport_must_extend_beyond_mesh():
    geometry = np.asarray([[1.0, 2.0], [4.0, 2.5], [3.0, 6.0]], dtype=np.float64)

    assert terrain_viewport_contains_mesh(
        terrain_extent=(0.0, 8.0, 0.0, 9.0),
        geometry_xy=geometry,
    )
    assert not terrain_viewport_contains_mesh(
        terrain_extent=(1.0, 8.0, 0.0, 9.0),
        geometry_xy=geometry,
    )


def test_paper_domain_viewport_uses_mesh_extent_with_2p5_percent_padding():
    geometry = np.asarray([[200.0, 300.0], [800.0, 350.0], [600.0, 700.0]], dtype=np.float64)

    viewport = paper_domain_viewport(
        terrain_extent=(0.0, 1000.0, 0.0, 1000.0),
        geometry_xy=geometry,
    )

    assert viewport == pytest.approx((185.0, 815.0, 290.0, 710.0))


def test_rendered_subthreshold_forecast_leaves_dem_pixels_unchanged(tmp_path):
    terrain = TerrainContext(
        image=np.linspace(-15.1, 19.9, 60 * 80, dtype=np.float32).reshape(60, 80),
        extent=(0.0, 10.0, 0.0, 8.0),
        source_crs="EPSG:26918",
        target_crs="EPSG:32618",
        source_path="synthetic-dem.tif",
    )
    geometry = np.asarray(
        [[3.0, 2.0], [7.0, 2.0], [3.0, 6.0], [7.0, 6.0], [5.0, 4.0]],
        dtype=np.float64,
    )

    def render(name: str, values: np.ndarray):
        path = tmp_path / f"{name}.webp"
        render_spatial_webp(
            values=values,
            geometry_xy=geometry,
            terrain=terrain,
            output_path=path,
            title="Synthetic probability",
            colorbar_label="Probability / values below 0.10 hidden",
            cmap=probability_cmap(),
            vmin=0.10,
            vmax=1.0,
            display_floor=0.10,
        )
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    below_zero = render("below-zero", np.zeros(geometry.shape[0], dtype=np.float64))
    below_small = render("below-small", np.full(geometry.shape[0], 0.09, dtype=np.float64))
    visible = render("visible", np.full(geometry.shape[0], 0.80, dtype=np.float64))

    assert np.array_equal(below_zero, below_small)
    assert not np.array_equal(below_zero, visible)


def test_generated_svg_has_no_trailing_whitespace(tmp_path):
    output = tmp_path / "trajectory.svg"
    lead = np.asarray([0.0, 0.25, 0.5], dtype=np.float64)
    render_validation_trajectory_svg(
        lead_time_hours=lead,
        p05=np.asarray([0.0, 0.1, 0.2]),
        p50=np.asarray([0.0, 0.2, 0.3]),
        p95=np.asarray([0.0, 0.3, 0.4]),
        reference=np.asarray([0.0, 0.18, 0.28]),
        threshold_m=0.1,
        output_path=output,
        event_label="Synthetic event",
    )

    assert all(line == line.rstrip() for line in output.read_text(encoding="utf-8").splitlines())
