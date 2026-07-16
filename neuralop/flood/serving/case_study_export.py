"""Deterministic public-asset export for the Portsmouth case study.

The exporter is deliberately separate from request-time serving. It consumes
completed, auditable run artifacts and writes a static web package; the public
marketing page never reads HDF5 files or calls the serving API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CaseStudyRunProvenance:
    label: str
    bundle_id: str
    calibration_mode: str
    ensemble_count: int
    members_per_ensemble: int
    forecast_steps: int
    dt_seconds: int
    mesh_hash: str

    @property
    def compatibility_key(self) -> tuple[object, ...]:
        return (
            self.bundle_id,
            self.calibration_mode,
            self.ensemble_count,
            self.members_per_ensemble,
            self.forecast_steps,
            self.dt_seconds,
            self.mesh_hash,
        )


def validate_case_study_provenance(runs: Iterable[CaseStudyRunProvenance]) -> None:
    records = list(runs)
    if not records:
        raise ValueError("At least one completed run is required for case-study export.")
    expected = records[0].compatibility_key
    mismatched = [record.label for record in records[1:] if record.compatibility_key != expected]
    if mismatched:
        raise ValueError(
            "Case-study export rejected mixed model or ensemble provenance: "
            + ", ".join(mismatched)
        )


def select_showcase_frames(
    *,
    n_time: int,
    frame_count: int,
    required_indices: Sequence[int] = (),
) -> list[int]:
    """Select evenly distributed frames while preserving scientific milestones."""
    n_time = int(n_time)
    frame_count = int(frame_count)
    if n_time < 1:
        raise ValueError("n_time must be positive.")
    if frame_count < 2 or frame_count > n_time:
        raise ValueError("frame_count must be between 2 and n_time.")

    required = {int(index) for index in required_indices}
    if any(index < 0 or index >= n_time for index in required):
        raise ValueError("required frame index is outside the forecast horizon.")
    required.update((0, n_time - 1))
    if len(required) > frame_count:
        raise ValueError("Required milestones exceed the requested frame count.")

    candidates = [int(round(value)) for value in np.linspace(0, n_time - 1, frame_count)]
    selected = set(candidates)
    selected.update(required)

    while len(selected) > frame_count:
        removable = [index for index in selected if index not in required]
        if not removable:
            break
        # Remove the most redundant frame: the one with the smallest distance
        # to either immediate neighbour. Tie-breaking is deterministic.
        ordered = sorted(selected)
        redundancy: list[tuple[int, int]] = []
        for index in removable:
            pos = ordered.index(index)
            left = index - ordered[pos - 1] if pos > 0 else n_time
            right = ordered[pos + 1] - index if pos + 1 < len(ordered) else n_time
            redundancy.append((min(left, right), index))
        selected.remove(min(redundancy)[1])

    if len(selected) < frame_count:
        remaining = [index for index in range(n_time) if index not in selected]
        while len(selected) < frame_count:
            ordered = sorted(selected)
            best = max(
                remaining,
                key=lambda index: (min(abs(index - other) for other in ordered), -index),
            )
            selected.add(best)
            remaining.remove(best)
    return sorted(selected)


def masked_triangle_face_values(
    *,
    values: np.ndarray,
    triangles: np.ndarray,
    display_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat-shaded face values and a transparent-face mask.

    A triangle is visible when at least one finite vertex reaches the display
    floor, matching the publication renderer while leaving negligible signal
    fully transparent over the terrain background.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    faces = np.asarray(triangles, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("triangles must have shape [n_triangles,3].")
    if faces.size and (faces.min() < 0 or faces.max() >= arr.shape[0]):
        raise ValueError("triangle index is outside the values array.")
    tri_values = arr[faces]
    with np.errstate(invalid="ignore"):
        face_values = np.nanmean(tri_values, axis=1)
        face_max = np.nanmax(tri_values, axis=1)
    face_mask = (~np.isfinite(face_values)) | (~np.isfinite(face_max)) | (face_max < float(display_floor))
    face_values = face_values.astype(np.float64, copy=False)
    face_values[face_mask] = np.nan
    return face_values, face_mask


def terrain_viewport_contains_mesh(
    *,
    terrain_extent: Sequence[float],
    geometry_xy: np.ndarray,
) -> bool:
    """Return whether the rectangular terrain viewport extends beyond the mesh on every side."""
    extent = np.asarray(terrain_extent, dtype=np.float64).reshape(-1)
    xy = np.asarray(geometry_xy, dtype=np.float64)
    if extent.shape != (4,) or xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("Terrain extent must have four values and geometry_xy must have shape [n_cells,2].")
    if not np.all(np.isfinite(extent)) or not np.all(np.isfinite(xy)):
        raise ValueError("Terrain extent and geometry must be finite.")
    left, right, bottom, top = (float(value) for value in extent)
    if left >= right or bottom >= top:
        raise ValueError("Terrain extent is not ordered as left, right, bottom, top.")
    return bool(
        left < float(np.min(xy[:, 0]))
        and right > float(np.max(xy[:, 0]))
        and bottom < float(np.min(xy[:, 1]))
        and top > float(np.max(xy[:, 1]))
    )


def paper_domain_viewport(
    *,
    terrain_extent: Sequence[float],
    geometry_xy: np.ndarray,
    pad_frac: float = 0.025,
) -> tuple[float, float, float, float]:
    """Return the publication map extent: mesh bounds plus proportional padding.

    The external DEM remains the imagery source, but the displayed viewport
    follows the paper renderer instead of exposing the GeoTIFF's full bounds.
    """
    terrain = np.asarray(terrain_extent, dtype=np.float64).reshape(-1)
    xy = np.asarray(geometry_xy, dtype=np.float64)
    if terrain.shape != (4,) or xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("Terrain extent must have four values and geometry_xy must have shape [n_cells,2].")
    if not np.all(np.isfinite(terrain)) or not np.all(np.isfinite(xy)):
        raise ValueError("Terrain extent and geometry must be finite.")
    if not 0.0 <= float(pad_frac) <= 0.5:
        raise ValueError("pad_frac must be between 0.0 and 0.5.")
    mesh_left = float(np.min(xy[:, 0]))
    mesh_right = float(np.max(xy[:, 0]))
    mesh_bottom = float(np.min(xy[:, 1]))
    mesh_top = float(np.max(xy[:, 1]))
    x_pad = max((mesh_right - mesh_left) * float(pad_frac), 1.0)
    y_pad = max((mesh_top - mesh_bottom) * float(pad_frac), 1.0)
    left, right, bottom, top = (float(value) for value in terrain)
    viewport = (
        max(left, mesh_left - x_pad),
        min(right, mesh_right + x_pad),
        max(bottom, mesh_bottom - y_pad),
        min(top, mesh_top + y_pad),
    )
    if not terrain_viewport_contains_mesh(terrain_extent=viewport, geometry_xy=xy):
        raise ValueError("External terrain does not cover the padded publication viewport.")
    return viewport


@dataclass(frozen=True)
class CaseStudyExportConfig:
    bundle_manifest: Path
    flagship_run: Path
    historical_reference_bundle: Path
    historical_runs: tuple[tuple[str, str, Path], ...]
    output_dir: Path
    public_prefix: str = "/marketing/portsmouth"

    @classmethod
    def from_json(cls, path: str | Path) -> "CaseStudyExportConfig":
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))

        def _resolve(value: str) -> Path:
            candidate = Path(value).expanduser()
            return candidate.resolve() if candidate.is_absolute() else (config_path.parent / candidate).resolve()

        historical = tuple(
            (str(item["label"]), str(item["event_id"]), _resolve(str(item["run_root"])))
            for item in payload["historical_runs"]
        )
        return cls(
            bundle_manifest=_resolve(str(payload["bundle_manifest"])),
            flagship_run=_resolve(str(payload["flagship_run"])),
            historical_reference_bundle=_resolve(str(payload["historical_reference_bundle"])),
            historical_runs=historical,
            output_dir=_resolve(str(payload["output_dir"])),
            public_prefix=str(payload.get("public_prefix", "/marketing/portsmouth")).rstrip("/"),
        )


@dataclass
class CompletedRunData:
    root: Path
    manifest: dict[str, Any]
    provenance: CaseStudyRunProvenance
    geometry_xy: np.ndarray
    elevation_m: np.ndarray
    wettable_mask: np.ndarray
    lead_time_hours: np.ndarray
    calibrated_members_wd: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required case-study artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def load_run_provenance(run_root: str | Path) -> CaseStudyRunProvenance:
    root = Path(run_root).expanduser().resolve()
    manifest = _read_json(root / "run_manifest.json")
    run = manifest.get("run", {})
    bundle = manifest.get("bundle", {})
    return CaseStudyRunProvenance(
        label=str(run.get("label") or run.get("run_id") or root.name),
        bundle_id=str(run.get("bundle_id") or bundle.get("bundle_id") or ""),
        calibration_mode=str(run.get("calibration_mode") or ""),
        ensemble_count=int(run.get("ensemble_count", 0)),
        members_per_ensemble=int(run.get("members_per_ensemble", 0)),
        forecast_steps=int(run.get("forecast_steps", 0)),
        dt_seconds=int(bundle.get("dt_seconds", 0)),
        mesh_hash=str(bundle.get("mesh_hash") or ""),
    )


def load_completed_run(run_root: str | Path) -> CompletedRunData:
    root = Path(run_root).expanduser().resolve()
    manifest = _read_json(root / "run_manifest.json")
    geometry = _read_json(root / "geometry_meta.json")
    h5_path = root / "forecast_members.h5"
    if not h5_path.is_file():
        raise FileNotFoundError(f"Completed run is missing forecast_members.h5: {root}")
    try:
        import h5py
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("Case-study export requires h5py.") from exc
    with h5py.File(h5_path, "r") as h5:
        required = {"calibrated_members_wd", "lead_time_hours", "wettable_mask"}
        missing = sorted(required.difference(h5.keys()))
        if missing:
            raise ValueError(f"{h5_path} is missing datasets: {', '.join(missing)}")
        members = np.asarray(h5["calibrated_members_wd"], dtype=np.float32)
        lead = np.asarray(h5["lead_time_hours"], dtype=np.float32)
        wettable = np.asarray(h5["wettable_mask"], dtype=bool)
    x = np.asarray(geometry.get("x", []), dtype=np.float64)
    y = np.asarray(geometry.get("y", []), dtype=np.float64)
    elevation = np.asarray(geometry.get("elevation_m", []), dtype=np.float64)
    xy = np.column_stack((x, y))
    if members.ndim != 3 or members.shape[1] != lead.size or members.shape[2] != xy.shape[0]:
        raise ValueError(f"Run arrays and geometry are incompatible in {root}.")
    if wettable.shape != (xy.shape[0],) or elevation.shape != (xy.shape[0],):
        raise ValueError(f"Run wettable/elevation metadata are incompatible in {root}.")
    return CompletedRunData(
        root=root,
        manifest=manifest,
        provenance=load_run_provenance(root),
        geometry_xy=xy,
        elevation_m=elevation,
        wettable_mask=wettable,
        lead_time_hours=lead,
        calibrated_members_wd=np.clip(members, 0.0, None),
    )


def _bundle_paths(bundle_manifest_path: Path) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    payload = _read_json(bundle_manifest_path)
    root = bundle_manifest_path.parent
    terrain_value = payload.get("visualization", {}).get("map", {}).get("terrain_tif")
    if not terrain_value:
        raise ValueError("Model bundle does not configure visualization.map.terrain_tif.")
    terrain = (root / str(terrain_value)).resolve()
    static = (root / str(payload.get("static_tensor_path", ""))).resolve()
    coefficients = (root / str(payload.get("calibration_coefficients_path", ""))).resolve()
    isotonic = (root / str(payload.get("isotonic_curves_path", ""))).resolve()
    for required in (terrain, static, coefficients, isotonic):
        if not required.is_file():
            raise FileNotFoundError(f"Model-bundle case-study dependency is missing: {required}")
    return payload, terrain, static, coefficients, isotonic


def _calibrated_probability(
    members: np.ndarray,
    *,
    wettable_mask: np.ndarray,
    lead_time_hours: np.ndarray,
    threshold_m: float,
    calibration_adapter,
) -> np.ndarray:
    raw = np.mean(np.asarray(members, dtype=np.float64) > float(threshold_m), axis=0)
    out = np.full(raw.shape, np.nan, dtype=np.float64)
    out[:, wettable_mask] = calibration_adapter.apply_isotonic_exceedance(
        raw[:, wettable_mask],
        threshold_m=float(threshold_m),
        lead_time_hour=lead_time_hours,
        wettable_mask=wettable_mask,
    )
    return np.clip(out, 0.0, 1.0)


def _read_forcing(path: Path, *, forecast_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < forecast_steps:
        raise ValueError(f"Forcing CSV is shorter than forecast horizon: {path}")
    rows = rows[-forecast_steps:]
    time_seconds = np.asarray([float(row["time_seconds"]) for row in rows], dtype=np.float64)
    stage = np.asarray([float(row["stage"]) for row in rows], dtype=np.float64)
    precipitation = np.asarray([float(row["precipitation"]) for row in rows], dtype=np.float64)
    return (time_seconds - time_seconds[0]) / 3600.0, stage, precipitation


def _relative_asset(public_prefix: str, output_dir: Path, path: Path) -> str:
    return f"{public_prefix}/{path.relative_to(output_dir).as_posix()}"


def _safe_visible_vmax(values: np.ndarray, *, floor: float, quantile: float, cap: float, minimum: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    visible = arr[np.isfinite(arr) & (arr >= float(floor))]
    if not visible.size:
        return float(minimum)
    return float(min(cap, max(minimum, np.quantile(visible, quantile))))


def _location_indices(
    *,
    probability: np.ndarray,
    members: np.ndarray,
    wettable_mask: np.ndarray,
    disagreement_time_idx: int,
) -> list[tuple[str, str, int]]:
    max_probability = np.nanmax(probability, axis=0)
    above = probability >= 0.5
    first = np.where(above.any(axis=0), np.argmax(above, axis=0), probability.shape[0] + 1)
    peak_depth = members.mean(axis=0).max(axis=0)
    candidate = wettable_mask & (max_probability >= 0.5) & (peak_depth >= 0.30)
    candidate_indices = np.flatnonzero(candidate)
    if not candidate_indices.size:
        candidate_indices = np.flatnonzero(wettable_mask)
    early = int(candidate_indices[np.lexsort((-peak_depth[candidate_indices], first[candidate_indices]))[0]])

    width = np.quantile(members[:, disagreement_time_idx, :], 0.95, axis=0) - np.quantile(
        members[:, disagreement_time_idx, :], 0.05, axis=0
    )
    uncertainty_candidates = candidate_indices[candidate_indices != early]
    high_uncertainty = int(
        uncertainty_candidates[np.nanargmax(width[uncertainty_candidates])]
        if uncertainty_candidates.size
        else early
    )
    late_candidates = candidate_indices[(candidate_indices != early) & (candidate_indices != high_uncertainty)]
    later = int(
        late_candidates[np.lexsort((-peak_depth[late_candidates], -first[late_candidates]))[0]]
        if late_candidates.size
        else high_uncertainty
    )
    return [
        ("A", "Early onset", early),
        ("B", "Elevated member spread", high_uncertainty),
        ("C", "Later response", later),
    ]


def _render_flagship(
    *,
    config: CaseStudyExportConfig,
    run: CompletedRunData,
    output_dir: Path,
    terrain,
    area_m2: np.ndarray,
    calibration_adapter,
) -> dict[str, Any]:
    from neuralop.flood.serving.case_study_rendering import (
        depth_cmap,
        probability_cmap,
        render_location_panel_svg,
        render_spatial_webp,
        uncertainty_cmap,
    )

    members = run.calibrated_members_wd
    wettable = run.wettable_mask
    lead = run.lead_time_hours
    probability = _calibrated_probability(
        members,
        wettable_mask=wettable,
        lead_time_hours=lead,
        threshold_m=0.30,
        calibration_adapter=calibration_adapter,
    )
    mean_depth = members.mean(axis=0)
    interval_width = np.quantile(members, 0.95, axis=0) - np.quantile(members, 0.05, axis=0)
    wettable_area = float(np.sum(area_m2[wettable]))
    expected_area_m2 = np.nansum(probability[:, wettable] * area_m2[wettable][None, :], axis=1)
    expected_fraction = expected_area_m2 / max(wettable_area, 1.0e-12)
    peak_area_idx = int(np.nanargmax(expected_area_m2))
    onset_candidates = np.flatnonzero(expected_fraction >= 0.01)
    onset_idx = int(onset_candidates[0]) if onset_candidates.size else 0

    model_count = run.provenance.ensemble_count
    members_per_model = run.provenance.members_per_ensemble
    grouped = members.reshape(model_count, members_per_model, members.shape[1], members.shape[2])
    group_means = grouped.mean(axis=1)
    between_var = np.var(group_means, axis=0, ddof=0)
    within_var = np.mean(np.var(grouped, axis=1, ddof=0), axis=0)
    weighted_between = np.nansum(between_var[:, wettable] * area_m2[wettable][None, :], axis=1) / wettable_area
    disagreement_idx = int(np.nanargmax(weighted_between))
    frame_indices = select_showcase_frames(
        n_time=members.shape[1],
        frame_count=32,
        required_indices=(onset_idx, disagreement_idx, peak_area_idx, members.shape[1] - 1),
    )
    depth_vmax = _safe_visible_vmax(mean_depth, floor=0.05, quantile=0.995, cap=3.0, minimum=0.5)
    width_vmax = _safe_visible_vmax(interval_width, floor=0.08, quantile=0.985, cap=1.0, minimum=0.20)
    product_specs = {
        "probability": {
            "label": "P(WD > 0.30 m)",
            "values": probability,
            "cmap": probability_cmap(),
            "vmin": 0.10,
            "vmax": 1.0,
            "floor": 0.10,
            "colorbar": "Calibrated probability",
        },
        "meanDepth": {
            "label": "Calibrated mean depth",
            "values": mean_depth,
            "cmap": depth_cmap(),
            "vmin": 0.05,
            "vmax": depth_vmax,
            "floor": 0.05,
            "colorbar": "Mean WD (m)",
        },
        "intervalWidth": {
            "label": "90% interval width",
            "values": interval_width,
            "cmap": uncertainty_cmap(),
            "vmin": 0.08,
            "vmax": width_vmax,
            "floor": 0.08,
            "colorbar": "p95–p05 width (m)",
        },
    }
    products: list[dict[str, Any]] = []
    for product_id, spec in product_specs.items():
        frames: list[dict[str, Any]] = []
        for time_idx in frame_indices:
            frame_path = output_dir / "frames" / product_id / f"irene_t{time_idx + 1:03d}.webp"
            render_spatial_webp(
                values=spec["values"][time_idx],
                geometry_xy=run.geometry_xy,
                terrain=terrain,
                output_path=frame_path,
                title=f"Irene 2011 · {spec['label']} · lead {float(lead[time_idx]):.2f} h",
                colorbar_label=spec["colorbar"],
                cmap=spec["cmap"],
                vmin=float(spec["vmin"]),
                vmax=float(spec["vmax"]),
                display_floor=float(spec["floor"]),
                quality=76,
            )
            frames.append(
                {
                    "timeIndex": int(time_idx),
                    "leadHours": round(float(lead[time_idx]), 2),
                    "src": _relative_asset(config.public_prefix, output_dir, frame_path),
                }
            )
        products.append(
            {
                "id": product_id,
                "label": spec["label"],
                "displayFloor": float(spec["floor"]),
                "vmin": float(spec["vmin"]),
                "vmax": round(float(spec["vmax"]), 4),
                "frames": frames,
            }
        )

    hero_dir = output_dir / "hero"
    hero_frame_dir = hero_dir / "frames"
    hero_path = hero_dir / "irene_mean_depth_poster.webp"
    hero_mp4_path = hero_dir / "irene_mean_depth_propagation.mp4"
    hero_webm_path = hero_dir / "irene_mean_depth_propagation.webm"
    hero_sequence_path = hero_dir / "sequence.json"
    # The video is encoded by the lightweight case-study video tool after the
    # scientific export. Removing prior encodes prevents stale visual evidence
    # from surviving a regenerated frame sequence.
    hero_mp4_path.unlink(missing_ok=True)
    hero_webm_path.unlink(missing_ok=True)

    hero_sequence: list[dict[str, int | float | str]] = []
    for sequence_index, time_idx in enumerate(frame_indices, start=1):
        frame_path = hero_frame_dir / f"irene_{sequence_index:03d}.webp"
        render_spatial_webp(
            values=mean_depth[time_idx],
            geometry_xy=run.geometry_xy,
            terrain=terrain,
            output_path=frame_path,
            title="",
            colorbar_label="",
            cmap=depth_cmap(),
            vmin=0.05,
            vmax=depth_vmax,
            display_floor=0.05,
            quality=80,
            show_title=False,
            show_colorbar=False,
        )
        hero_sequence.append(
            {
                "sequence": sequence_index,
                "timeIndex": int(time_idx),
                "leadHours": round(float(lead[time_idx]), 2),
            }
        )
    hero_sequence_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "eventId": "2011_IRENE",
                "product": "Calibrated mean water depth",
                "frameRate": 4,
                "frameCount": len(hero_sequence),
                "frames": hero_sequence,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    render_spatial_webp(
        values=mean_depth[peak_area_idx],
        geometry_xy=run.geometry_xy,
        terrain=terrain,
        output_path=hero_path,
        title="",
        colorbar_label="",
        cmap=depth_cmap(),
        vmin=0.05,
        vmax=depth_vmax,
        display_floor=0.05,
        quality=84,
        show_title=False,
        show_colorbar=False,
    )

    locations = _location_indices(
        probability=probability,
        members=members,
        wettable_mask=wettable,
        disagreement_time_idx=disagreement_idx,
    )
    location_payload: list[dict[str, Any]] = []
    marker_tuples = [
        (float(run.geometry_xy[index, 0]), float(run.geometry_xy[index, 1]), label, False)
        for label, _, index in locations
    ]
    for label, interpretation, cell_index in locations:
        selected_markers = [(x, y, marker_label, marker_label == label) for x, y, marker_label, _ in marker_tuples]
        map_path = output_dir / "locations" / f"location_{label.lower()}_map.webp"
        render_spatial_webp(
            values=probability[peak_area_idx],
            geometry_xy=run.geometry_xy,
            terrain=terrain,
            output_path=map_path,
            title=f"Location {label} · {interpretation}",
            colorbar_label="Calibrated P(WD > 0.30 m)",
            cmap=probability_cmap(),
            vmin=0.10,
            vmax=1.0,
            display_floor=0.10,
            markers=selected_markers,
            quality=78,
        )
        panel_path = output_dir / "locations" / f"location_{label.lower()}_evidence.svg"
        render_location_panel_svg(
            lead_time_hours=lead,
            members_wd=members[:, :, cell_index],
            exceedance_probability=probability[:, cell_index],
            threshold_m=0.30,
            output_path=panel_path,
            location_label=label,
        )
        location_payload.append(
            {
                "id": label.lower(),
                "label": f"Location {label}",
                "interpretation": interpretation,
                "cellIndex": int(cell_index),
                "coordinates": {
                    "easting": round(float(run.geometry_xy[cell_index, 0]), 1),
                    "northing": round(float(run.geometry_xy[cell_index, 1]), 1),
                },
                "mapSrc": _relative_asset(config.public_prefix, output_dir, map_path),
                "panelSrc": _relative_asset(config.public_prefix, output_dir, panel_path),
            }
        )

    between_sd = np.sqrt(np.clip(between_var[disagreement_idx], 0.0, None))
    within_sd = np.sqrt(np.clip(within_var[disagreement_idx], 0.0, None))
    shared_vmax = _safe_visible_vmax(
        np.concatenate((between_sd[wettable], within_sd[wettable])),
        floor=0.02,
        quantile=0.995,
        cap=1.0,
        minimum=0.10,
    )
    decomposition_maps: list[dict[str, Any]] = []
    for map_id, label, values in (
        ("betweenModel", "Epistemic uncertainty", between_sd),
        ("withinModel", "Aleatoric uncertainty", within_sd),
    ):
        map_path = output_dir / "decomposition" / f"{map_id}.webp"
        render_spatial_webp(
            values=np.where(wettable, values, np.nan),
            geometry_xy=run.geometry_xy,
            terrain=terrain,
            output_path=map_path,
            title=f"{label} · lead {float(lead[disagreement_idx]):.2f} h",
            colorbar_label="Standard deviation (m)",
            cmap=uncertainty_cmap(),
            vmin=0.02,
            vmax=shared_vmax,
            display_floor=0.02,
            quality=80,
        )
        decomposition_maps.append(
            {"id": map_id, "label": label, "src": _relative_asset(config.public_prefix, output_dir, map_path)}
        )
    between_total = float(np.nansum(between_var[disagreement_idx, wettable] * area_m2[wettable]))
    within_total = float(np.nansum(within_var[disagreement_idx, wettable] * area_m2[wettable]))
    between_share = between_total / max(between_total + within_total, 1.0e-12)

    timing = _read_json(run.root / "performance_timing.json")
    phases = timing.get("phases", {})
    phase_total = phases.get("total", {}) if isinstance(phases, Mapping) else {}
    total_seconds = float(
        timing.get(
            "total_seconds",
            phase_total.get("seconds", phases.get("total_seconds", 0.0))
            if isinstance(phase_total, Mapping)
            else phases.get("total_seconds", 0.0),
        )
    )
    if total_seconds <= 0.0:
        total_seconds = float(timing.get("total", 0.0))
    peak_width = float(
        np.nansum(interval_width[peak_area_idx, wettable] * area_m2[wettable]) / max(wettable_area, 1.0e-12)
    )
    return {
        "eventId": "2011_IRENE",
        "label": "Irene 2011",
        "thresholdM": 0.30,
        "metrics": [
            {
                "value": f"{float(expected_area_m2[peak_area_idx] / 1_000_000.0):.2f} km²",
                "label": "Peak expected footprint above 0.30 m",
            },
            {"value": f"+{float(lead[peak_area_idx]):.2f} h", "label": "Lead to peak expected footprint"},
            {"value": f"{peak_width:.2f} m", "label": "Area-weighted 90% interval width at peak"},
            {"value": f"{total_seconds / 60.0:.1f} min", "label": "Measured full workflow on RTX 4090"},
        ],
        "peakAreaTimeIndex": peak_area_idx,
        "peakDisagreementTimeIndex": disagreement_idx,
        "posterSrc": products[0]["frames"][0]["src"],
        "hero": {
            "src": _relative_asset(config.public_prefix, output_dir, hero_path),
            "posterSrc": _relative_asset(config.public_prefix, output_dir, hero_path),
            "mp4Src": _relative_asset(config.public_prefix, output_dir, hero_mp4_path),
            "webmSrc": _relative_asset(config.public_prefix, output_dir, hero_webm_path),
            "sequenceSrc": _relative_asset(config.public_prefix, output_dir, hero_sequence_path),
            "frameCount": len(hero_sequence),
            "frameRate": 4,
            "durationSeconds": round(len(hero_sequence) / 4.0, 2),
            "product": "Calibrated mean water depth",
            "leadHours": round(float(lead[peak_area_idx]), 2),
            "displayFloorM": 0.05,
            "selection": "Peak expected footprint above 0.30 m",
        },
        "products": products,
        "snapshot": [
            {
                "id": item["id"],
                "label": item["label"],
                "src": next(frame["src"] for frame in item["frames"] if frame["timeIndex"] == peak_area_idx),
            }
            for item in products
        ],
        "locations": location_payload,
        "decomposition": {
            "leadHours": round(float(lead[disagreement_idx]), 2),
            "betweenVarianceShare": round(float(between_share), 4),
            "displayFloorM": 0.02,
            "sharedVmaxM": round(float(shared_vmax), 4),
            "maps": decomposition_maps,
        },
    }


def _historical_reference_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Historical reference bundle not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _render_historical_validation(
    *,
    config: CaseStudyExportConfig,
    run_specs: Sequence[tuple[str, str, Path]],
    reference: Mapping[str, np.ndarray],
    output_dir: Path,
    terrain,
    area_m2: np.ndarray,
) -> list[dict[str, Any]]:
    from neuralop.flood.serving.case_study_rendering import (
        probability_cmap,
        render_spatial_webp,
        render_validation_trajectory_svg,
        uncertainty_cmap,
    )

    display_names = [str(value) for value in reference["display_names"].tolist()]
    event_ids = [str(value) for value in reference["event_ids"].tolist()]
    lookup = {event_id: index for index, event_id in enumerate(event_ids)}
    computed: list[dict[str, Any]] = []
    for label, event_id, root in run_specs:
        if event_id not in lookup:
            raise ValueError(f"Historical reference bundle does not contain {event_id}.")
        run = load_completed_run(root)
        ref_idx = lookup[event_id]
        ref_geometry = np.asarray(reference["geometry"], dtype=np.float64)
        if run.geometry_xy.shape != ref_geometry.shape or not np.allclose(run.geometry_xy, ref_geometry, atol=0.05):
            raise ValueError(f"HEC-RAS reference geometry does not align with {label}.")
        members = run.calibrated_members_wd
        max_members = np.max(members, axis=1)
        probability = np.mean(max_members > 0.10, axis=0)
        interval_width = np.quantile(max_members, 0.95, axis=0) - np.quantile(max_members, 0.05, axis=0)
        wettable = run.wettable_mask
        member_area_fraction = (
            np.sum((members[:, :, wettable] > 0.10) * area_m2[wettable][None, None, :], axis=2)
            / max(float(np.sum(area_m2[wettable])), 1.0e-12)
        )
        computed.append(
            {
                "label": label,
                "event_id": event_id,
                "reference_index": ref_idx,
                "run": run,
                "probability": np.where(wettable, probability, np.nan),
                "interval_width": np.where(wettable, interval_width, np.nan),
                "p05": np.quantile(member_area_fraction, 0.05, axis=0),
                "p50": np.quantile(member_area_fraction, 0.50, axis=0),
                "p95": np.quantile(member_area_fraction, 0.95, axis=0),
            }
        )
    width_vmax = _safe_visible_vmax(
        np.concatenate([item["interval_width"][np.isfinite(item["interval_width"])] for item in computed]),
        floor=0.08,
        quantile=0.985,
        cap=1.0,
        minimum=0.20,
    )
    payload: list[dict[str, Any]] = []
    for item in computed:
        run = item["run"]
        ref_idx = int(item["reference_index"])
        ref_max = np.asarray(reference["ref_max_depth"][ref_idx], dtype=np.float64)
        prob_path = output_dir / "validation" / item["event_id"].lower() / "probability.webp"
        width_path = output_dir / "validation" / item["event_id"].lower() / "interval_width.webp"
        trajectory_path = output_dir / "validation" / item["event_id"].lower() / "extent_trajectory.svg"
        render_spatial_webp(
            values=item["probability"],
            geometry_xy=run.geometry_xy,
            terrain=terrain,
            output_path=prob_path,
            title=f"{item['label']} · ensemble P(max WD > 0.10 m)",
            colorbar_label="Probability",
            cmap=probability_cmap(),
            vmin=0.10,
            vmax=1.0,
            display_floor=0.10,
            reference_values=ref_max,
            reference_threshold=0.10,
            quality=80,
        )
        render_spatial_webp(
            values=item["interval_width"],
            geometry_xy=run.geometry_xy,
            terrain=terrain,
            output_path=width_path,
            title=f"{item['label']} · maximum 90% interval width",
            colorbar_label="p95–p05 width (m)",
            cmap=uncertainty_cmap(),
            vmin=0.08,
            vmax=width_vmax,
            display_floor=0.08,
            reference_values=ref_max,
            reference_threshold=0.10,
            quality=80,
        )
        reference_fraction = np.asarray(reference["ref_frac"][ref_idx], dtype=np.float64)
        if reference_fraction.shape != run.lead_time_hours.shape:
            raise ValueError(f"HEC-RAS trajectory length does not match {item['label']}.")
        render_validation_trajectory_svg(
            lead_time_hours=run.lead_time_hours,
            p05=item["p05"],
            p50=item["p50"],
            p95=item["p95"],
            reference=reference_fraction,
            threshold_m=0.10,
            output_path=trajectory_path,
            event_label=item["label"],
        )
        payload.append(
            {
                "eventId": item["event_id"],
                "label": item["label"],
                "thresholdM": 0.10,
                "probabilitySrc": _relative_asset(config.public_prefix, output_dir, prob_path),
                "intervalWidthSrc": _relative_asset(config.public_prefix, output_dir, width_path),
                "trajectorySrc": _relative_asset(config.public_prefix, output_dir, trajectory_path),
                "runId": run.manifest["run"]["run_id"],
            }
        )
    return payload


def export_portsmouth_case_study(config: CaseStudyExportConfig) -> Path:
    from neuralop.flood.serving.calibration import CalibrationAdapter
    from neuralop.flood.serving.case_study_rendering import load_terrain_context

    bundle, terrain_path, static_path, coefficients_path, isotonic_path = _bundle_paths(config.bundle_manifest)
    run_roots = [config.flagship_run, *(root for _, _, root in config.historical_runs)]
    provenances = [load_run_provenance(root) for root in run_roots]
    validate_case_study_provenance(provenances)
    expected_bundle = str(bundle.get("bundle_id", ""))
    if not expected_bundle or any(item.bundle_id != expected_bundle for item in provenances):
        raise ValueError("Run provenance does not match the configured production model bundle.")
    if any(item.ensemble_count != 3 or item.members_per_ensemble != 20 for item in provenances):
        raise ValueError("Portsmouth public evidence requires the production 3 x 20 member policy.")

    static = np.asarray(np.load(static_path), dtype=np.float64)
    if static.ndim != 2 or static.shape[1] < 2:
        raise ValueError("Model-bundle static tensor does not contain cell area in column 1.")
    area_m2 = static[:, 1]
    if not np.all(np.isfinite(area_m2)) or np.any(area_m2 <= 0.0):
        raise ValueError("Model-bundle cell areas must be finite and positive.")
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain = load_terrain_context(terrain_path, target_crs=str(bundle.get("crs", "EPSG:32618")))
    calibration_adapter = CalibrationAdapter.from_files(coefficients_path, isotonic_path)
    flagship_run = load_completed_run(config.flagship_run)
    if area_m2.shape[0] != flagship_run.geometry_xy.shape[0]:
        raise ValueError("Model-bundle cell areas do not match completed-run geometry.")
    terrain_source_extent = terrain.extent
    terrain_viewport = paper_domain_viewport(
        terrain_extent=terrain_source_extent,
        geometry_xy=flagship_run.geometry_xy,
    )
    terrain = replace(terrain, viewport=terrain_viewport)
    mesh_extent = [
        float(np.min(flagship_run.geometry_xy[:, 0])),
        float(np.max(flagship_run.geometry_xy[:, 0])),
        float(np.min(flagship_run.geometry_xy[:, 1])),
        float(np.max(flagship_run.geometry_xy[:, 1])),
    ]
    flagship = _render_flagship(
        config=config,
        run=flagship_run,
        output_dir=output_dir,
        terrain=terrain,
        area_m2=area_m2,
        calibration_adapter=calibration_adapter,
    )
    reference = _historical_reference_payload(config.historical_reference_bundle)
    historical = _render_historical_validation(
        config=config,
        run_specs=config.historical_runs,
        reference=reference,
        output_dir=output_dir,
        terrain=terrain,
        area_m2=area_m2,
    )
    run_manifest = flagship_run.manifest
    initial_condition = run_manifest.get("initial_condition", {})
    manifest = {
        "schemaVersion": 1,
        "caseStudyId": "portsmouth-irene-2011",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "title": "From historical forcing to calibrated flood probabilities.",
        "eyebrow": "PORTSMOUTH, VIRGINIA / DEPLOYMENT PROOF",
        "intro": (
            "A named historical event shows how a domain-specific FloodUQ deployment turns coastal forcing "
            "into probability, timing, extent, and uncertainty evidence. Portsmouth is the proof case; the "
            "service pattern is repeated through a new domain's own terrain and reference simulations."
        ),
        "flagship": flagship,
        "historicalValidation": {
            "thresholdM": 0.10,
            "events": historical,
            "note": (
                "The historical validation threshold is 0.10 m and is not directly compared with the "
                "0.30 m product view. Maximum-over-horizon maps use ensemble frequencies from calibrated "
                "depth members; no separate maximum-event isotonic mapping is implied."
            ),
        },
        "performance": {
            "workflow": {
                "hardware": "NVIDIA RTX 4090",
                "scope": "Complete 60-member workflow including HDF5, summaries, maps, animation, and scrub frames",
                "event": "Irene 2011",
            },
            "comparison": {
                "sample": "50 held-out coastal events",
                "ensembleBudget": "20 members per method",
                "hardware": "NVIDIA A100",
                "timingScope": "Forward-only inference per event",
                "fairCrpsUnit": "meters",
                "brierThresholdM": 0.30,
                "sourceArtifact": (
                    "coastal_uq_model_comparison/skill_cost_pareto_3x20_50events_optimized_20260613_034635/"
                    "skill_cost_metrics_by_size.csv"
                ),
            },
        },
        "provenance": {
            "bundleId": expected_bundle,
            "bundleGitCommit": str(bundle.get("git_commit", "")),
            "calibrationMode": flagship_run.provenance.calibration_mode,
            "ensemblePolicy": "3 checkpoints x 20 members",
            "seed": int(run_manifest.get("run", {}).get("seed", 0)),
            "initialConditionLibraryId": str(initial_condition.get("library_id", "")),
            "initialConditionReferenceScope": str(initial_condition.get("reference_scope", "")),
            "meshHash": flagship_run.provenance.mesh_hash,
            "terrainSha256": _sha256(terrain_path),
            "terrainSourceCrs": terrain.source_crs,
            "terrainTargetCrs": terrain.target_crs,
            "historicalReferenceSha256": _sha256(config.historical_reference_bundle),
            "dtSeconds": flagship_run.provenance.dt_seconds,
            "forecastSteps": flagship_run.provenance.forecast_steps,
        },
        "displayPolicy": {
            "demRangeM": [-15.1, 19.9],
            "demAlpha": 0.92,
            "overlayAlpha": 0.90,
            "meanDepthFloorM": 0.05,
            "probabilityFloor": 0.10,
            "intervalWidthFloorM": 0.08,
            "decompositionSdFloorM": 0.02,
            "longTriangleEdgeFactor": 2.5,
            "terrainSourceExtent": [float(value) for value in terrain_source_extent],
            "terrainViewport": [float(value) for value in terrain_viewport],
            "meshExtent": mesh_extent,
            "terrainExtendsBeyondMesh": True,
            "viewportPolicy": "mesh_bounds_plus_2p5_percent",
        },
        "researchDisclaimer": "Research only; not for emergency or operational decision use.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the static Portsmouth marketing evidence package.")
    parser.add_argument("--config", required=True, help="Path to a case-study export JSON configuration.")
    args = parser.parse_args(argv)
    path = export_portsmouth_case_study(CaseStudyExportConfig.from_json(args.config))
    print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
