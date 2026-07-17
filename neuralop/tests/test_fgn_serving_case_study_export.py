from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from neuralop.flood.serving.case_study_export import (
    CaseStudyRunProvenance,
    arrival_time_from_depth,
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
from neuralop.flood.serving.case_study_video import encode_case_study_hero, encode_case_study_products


def test_showcase_frames_span_horizon_and_keep_scientific_milestones():
    selected = select_showcase_frames(
        n_time=94,
        frame_count=32,
        required_indices=(0, 19, 46, 66, 93),
    )

    assert len(selected) == 32
    assert selected == sorted(set(selected))
    assert {0, 19, 46, 66, 93}.issubset(selected)


def test_arrival_time_map_uses_first_physical_threshold_exceedance():
    lead_hours = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
    depth_by_time = np.asarray(
        [
            [0.00, 0.00, 0.31],
            [0.35, 0.05, 0.40],
            [0.20, 0.30, 0.45],
            [0.10, 0.20, 0.50],
        ],
        dtype=np.float64,
    )

    arrival = arrival_time_from_depth(
        depth_by_time=depth_by_time,
        lead_time_hours=lead_hours,
        threshold_m=0.30,
    )

    assert arrival[0] == pytest.approx(0.50)
    assert np.isnan(arrival[1])
    assert arrival[2] == pytest.approx(0.25)


def test_hero_encoder_rejects_an_incomplete_scientific_sequence(tmp_path):
    hero_dir = tmp_path / "hero"
    frame_dir = hero_dir / "frames"
    frame_dir.mkdir(parents=True)
    (frame_dir / "irene_001.webp").write_bytes(b"one-frame")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "flagship": {
                    "hero": {
                        "frameCount": 2,
                        "frameRate": 4,
                        "mp4Src": "/marketing/portsmouth/hero/animation.mp4",
                        "webmSrc": "/marketing/portsmouth/hero/animation.webm",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="frame count mismatch"):
        encode_case_study_hero(manifest_path=manifest_path, ffmpeg_path="unused")


def test_product_video_encoder_rejects_an_incomplete_scientific_sequence(tmp_path):
    frame_dir = tmp_path / "frames" / "probability"
    frame_dir.mkdir(parents=True)
    (frame_dir / "irene_t001.webp").write_bytes(b"one-frame")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "flagship": {
                    "products": [
                        {
                            "id": "probability",
                            "frames": [{"timeIndex": 0}, {"timeIndex": 1}],
                            "animation": {
                                "frameCount": 2,
                                "sourceFrameRate": 6,
                                "playbackFrameRate": 24,
                                "mp4Src": "/marketing/portsmouth/animations/probability.mp4",
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probability frame count mismatch"):
        encode_case_study_products(manifest_path=manifest_path, ffmpeg_path="unused")


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
            colorbar_label="Probability",
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


def test_rendered_spatial_gradient_interpolates_node_values(tmp_path):
    terrain = TerrainContext(
        image=np.zeros((80, 80), dtype=np.float32),
        extent=(0.0, 10.0, 0.0, 10.0),
        source_crs="EPSG:32618",
        target_crs="EPSG:32618",
        source_path="synthetic-dem.tif",
        viewport=(0.0, 10.0, 0.0, 10.0),
    )
    geometry = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    output = tmp_path / "smooth-gradient.webp"

    render_spatial_webp(
        values=np.asarray([0.10, 1.0, 0.10, 1.0], dtype=np.float64),
        geometry_xy=geometry,
        terrain=terrain,
        output_path=output,
        title="",
        colorbar_label="",
        cmap=probability_cmap(),
        vmin=0.10,
        vmax=1.0,
        display_floor=0.10,
        quality=100,
        show_title=False,
        show_colorbar=False,
    )

    with Image.open(output) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    center_row = rgb[rgb.shape[0] // 2, rgb.shape[1] // 8 : -(rgb.shape[1] // 8)]
    color_steps = np.linalg.norm(np.diff(center_row, axis=0), axis=1)

    # A flat-shaded mesh produces one or two large face-color blocks. The
    # publication renderer should instead preserve a continuous node-valued
    # gradient across the same triangles.
    assert np.count_nonzero(color_steps > 0.5) > center_row.shape[0] // 8
    assert np.linalg.norm(center_row[-1] - center_row[0]) > 80.0


def test_rendered_threshold_boundary_does_not_create_white_halo(tmp_path):
    terrain = TerrainContext(
        image=np.zeros((80, 80), dtype=np.float32),
        extent=(0.0, 10.0, 0.0, 10.0),
        source_crs="EPSG:32618",
        target_crs="EPSG:32618",
        source_path="synthetic-dem.tif",
        viewport=(0.0, 10.0, 0.0, 10.0),
    )
    output = tmp_path / "threshold-boundary.webp"

    render_spatial_webp(
        values=np.asarray([0.0, 0.80, 0.0, 0.80], dtype=np.float64),
        geometry_xy=np.asarray(
            [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
            dtype=np.float64,
        ),
        terrain=terrain,
        output_path=output,
        title="",
        colorbar_label="",
        cmap=probability_cmap(),
        vmin=0.10,
        vmax=1.0,
        display_floor=0.10,
        quality=100,
        show_title=False,
        show_colorbar=False,
    )

    with Image.open(output) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    near_white = np.all(rgb > 245, axis=2)
    assert not np.any(near_white)


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
