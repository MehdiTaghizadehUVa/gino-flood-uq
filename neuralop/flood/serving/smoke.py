"""Serving smoke runner for the real coastal FGN bundle."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from neuralop.flood.serving.calibration import CalibrationAdapter
from neuralop.flood.serving.forcing import parse_forcing_csv
from neuralop.flood.serving.inference import ProductionFGNInferenceService
from neuralop.flood.serving.model_bundle import load_model_bundle
from neuralop.flood.serving.products import ForecastProductBuilder
from neuralop.flood.serving.run_spec import RunSpec


def _generated_forcing_csv(*, rows: int, dt_seconds: int) -> str:
    lines = ["time_seconds,stage,precipitation"]
    for i in range(int(rows)):
        stage = 0.25 + 0.002 * i
        precipitation = 0.0
        lines.append(f"{i * int(dt_seconds)},{stage:.6f},{precipitation:.6f}")
    return "\n".join(lines) + "\n"


def run_smoke(args: argparse.Namespace) -> Path:
    bundle = load_model_bundle(args.bundle_path)
    steps = int(args.forecast_steps)
    if args.forcing_csv:
        forcing_payload = Path(args.forcing_csv).read_text(encoding="utf-8")
    else:
        rows = bundle.skip_before_timestep + bundle.n_history + steps
        forcing_payload = _generated_forcing_csv(rows=rows, dt_seconds=bundle.dt_seconds)
    forcing = parse_forcing_csv(forcing_payload, bundle=bundle, requested_forecast_steps=steps)
    spec = RunSpec.new(
        user_id="smoke",
        bundle_id=bundle.bundle_id,
        input_hash=forcing.input_hash,
        forecast_steps=forcing.forecast_steps,
        label="serving-smoke",
        request_full_hdf5=False,
        seed=int(args.seed),
    )
    t0 = time.time()
    inference = ProductionFGNInferenceService(bundle, device=args.device, member_chunk_size=args.member_chunk_size)
    raw = inference.run(spec, forcing)
    calibration = CalibrationAdapter.from_files(bundle.calibration_coefficients_path, bundle.isotonic_curves_path)
    calibrated = calibration.apply(raw)
    builder = ForecastProductBuilder()
    raw_summary = builder.build_summary(raw, label="raw")
    calibrated_summary = builder.build_summary(
        calibrated,
        label="calibrated",
        calibration_adapter=calibration,
    )
    elapsed = time.time() - t0
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "forcing.csv").write_text(forcing_payload, encoding="utf-8")
    (out_dir / "raw_summary.json").write_text(json.dumps(raw_summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "calibrated_summary.json").write_text(json.dumps(calibrated_summary, indent=2, sort_keys=True), encoding="utf-8")
    map_paths = []
    map_paths.extend(builder.write_map_pngs(raw, output_dir=out_dir, label="raw", max_times=2))
    map_paths.extend(
        builder.write_map_pngs(
            calibrated,
            output_dir=out_dir,
            label="calibrated",
            max_times=2,
            calibration_adapter=calibration,
        )
    )
    smoke_summary = {
        "bundle_id": bundle.bundle_id,
        "forecast_steps": steps,
        "members_shape": list(raw.members_wd.shape),
        "calibrated_members_shape": list(calibrated.members_wd.shape),
        "finite_raw": bool(np.isfinite(raw.members_wd).all()),
        "finite_calibrated": bool(np.isfinite(calibrated.members_wd).all()),
        "min_raw_wd_m": float(np.min(raw.members_wd)),
        "max_raw_wd_m": float(np.max(raw.members_wd)),
        "min_calibrated_wd_m": float(np.min(calibrated.members_wd)),
        "max_calibrated_wd_m": float(np.max(calibrated.members_wd)),
        "elapsed_seconds": elapsed,
        "device": args.device,
        "seed": int(args.seed),
        "map_png_count": len(map_paths),
    }
    if raw.wettable_mask is not None:
        smoke_summary["n_wettable"] = int(np.asarray(raw.wettable_mask, dtype=bool).sum())
    (out_dir / "smoke_summary.json").write_text(json.dumps(smoke_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(smoke_summary, indent=2, sort_keys=True))
    return out_dir / "smoke_summary.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a small real coastal FGN serving smoke test.")
    parser.add_argument("--bundle-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forecast-steps", type=int, default=2)
    parser.add_argument("--forcing-csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--member-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    run_smoke(args)


if __name__ == "__main__":  # pragma: no cover
    main()
