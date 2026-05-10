"""User forcing CSV contract for coastal FGN serving."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np

from neuralop.flood.serving.model_bundle import FGNModelBundle


class ForcingValidationError(ValueError):
    """Raised when uploaded stage/precipitation forcing is invalid."""


_TIME_ALIASES = ("time_seconds", "time_s", "seconds", "time", "t")
_TIME_HOUR_ALIASES = ("time_hours", "time_h", "hours", "hour")
_STAGE_ALIASES = ("stage", "water_stage", "surge_stage")
_PRECIP_ALIASES = ("precipitation", "precip", "rainfall", "rain")


def build_forcing_template_csv(bundle: FGNModelBundle, *, forecast_steps: int = 1) -> str:
    """Return a minimally valid forcing CSV template for the deployed bundle."""
    rows = int(bundle.skip_before_timestep) + int(bundle.n_history) + max(1, int(forecast_steps))
    lines = ["time_seconds,stage,precipitation"]
    for idx in range(rows):
        # Mild synthetic forcing values keep the template within validation ranges
        # while making the required 20-minute cadence visible to users.
        stage = 0.10 + 0.01 * idx
        precipitation = 0.0
        lines.append(f"{idx * int(bundle.dt_seconds)},{stage:.4f},{precipitation:.4f}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ForcingInput:
    """Canonical forcing series accepted by the coastal FGN serving path."""

    stage: np.ndarray
    precipitation: np.ndarray
    dt_seconds: int
    forecast_steps: int
    input_hash: str
    source_name: str = "upload.csv"

    @property
    def n_rows(self) -> int:
        return int(self.stage.shape[0])

    def as_boundary_matrix(self) -> np.ndarray:
        return np.stack([self.stage, self.precipitation], axis=-1).astype(np.float32, copy=False)

    def summary(self) -> dict:
        return {
            "source_name": self.source_name,
            "input_hash": self.input_hash,
            "n_rows": self.n_rows,
            "dt_seconds": self.dt_seconds,
            "forecast_steps": self.forecast_steps,
            "stage_min": float(np.min(self.stage)),
            "stage_max": float(np.max(self.stage)),
            "precipitation_min": float(np.min(self.precipitation)),
            "precipitation_max": float(np.max(self.precipitation)),
        }


def _canonicalize_headers(headers: Iterable[str]) -> dict[str, str]:
    return {str(h).strip().lower(): str(h).strip() for h in headers}


def _find_column(headers: Mapping[str, str], aliases: Iterable[str], label: str) -> str:
    for alias in aliases:
        if alias in headers:
            return headers[alias]
    raise ForcingValidationError(
        f"Missing required {label} column. Accepted aliases: {', '.join(aliases)}."
    )


def _read_csv_text(data: str | bytes | Path) -> tuple[str, str]:
    if isinstance(data, Path):
        text = data.read_text(encoding="utf-8")
        return text, data.name
    if isinstance(data, bytes):
        return data.decode("utf-8-sig"), "upload.csv"
    return str(data), "upload.csv"


def parse_forcing_csv(
    data: str | bytes | Path,
    *,
    bundle: FGNModelBundle,
    requested_forecast_steps: Optional[int] = None,
    stage_range: tuple[float, float] = (-20.0, 20.0),
    precipitation_range: tuple[float, float] = (0.0, 500.0),
) -> ForcingInput:
    """Parse and validate a user forcing CSV."""
    text, source_name = _read_csv_text(data)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ForcingValidationError("Forcing CSV must include a header row.")
    headers = _canonicalize_headers(reader.fieldnames)
    stage_col = _find_column(headers, _STAGE_ALIASES, "stage")
    precip_col = _find_column(headers, _PRECIP_ALIASES, "precipitation")
    time_col = None
    time_scale = 1.0
    for alias in _TIME_ALIASES:
        if alias in headers:
            time_col = headers[alias]
            time_scale = 1.0
            break
    if time_col is None:
        for alias in _TIME_HOUR_ALIASES:
            if alias in headers:
                time_col = headers[alias]
                time_scale = 3600.0
                break

    stage_vals = []
    precip_vals = []
    time_vals = []
    for row_idx, row in enumerate(reader, start=2):
        try:
            stage_vals.append(float(row[stage_col]))
            precip_vals.append(float(row[precip_col]))
            if time_col is not None:
                time_vals.append(float(row[time_col]) * time_scale)
        except (KeyError, TypeError, ValueError) as exc:
            raise ForcingValidationError(f"Invalid numeric value at CSV row {row_idx}.") from exc

    if not stage_vals:
        raise ForcingValidationError("Forcing CSV contains no data rows.")
    stage = np.asarray(stage_vals, dtype=np.float64)
    precip = np.asarray(precip_vals, dtype=np.float64)
    if not np.all(np.isfinite(stage)) or not np.all(np.isfinite(precip)):
        raise ForcingValidationError("Forcing CSV contains NaN or infinite values.")
    if np.any(stage < stage_range[0]) or np.any(stage > stage_range[1]):
        raise ForcingValidationError(f"Stage values must be within [{stage_range[0]}, {stage_range[1]}].")
    if np.any(precip < precipitation_range[0]) or np.any(precip > precipitation_range[1]):
        raise ForcingValidationError(
            f"Precipitation values must be within [{precipitation_range[0]}, {precipitation_range[1]}]."
        )

    if time_col is not None:
        time = np.asarray(time_vals, dtype=np.float64)
        deltas = np.diff(time)
        if deltas.size and not np.allclose(deltas, float(bundle.dt_seconds), rtol=0.0, atol=1e-6):
            raise ForcingValidationError(
                f"Forcing timestep must be exactly {bundle.dt_seconds} seconds. "
                f"Observed deltas include {np.unique(np.round(deltas, 6))[:5].tolist()}."
            )

    min_rows = bundle.min_required_forcing_rows
    if stage.shape[0] < min_rows:
        raise ForcingValidationError(
            f"Forcing CSV is too short: got {stage.shape[0]} rows, need at least {min_rows} "
            "for spin-up/history plus one forecast step."
        )
    available_steps = int(stage.shape[0]) - int(bundle.skip_before_timestep) - int(bundle.n_history)
    if requested_forecast_steps is None:
        forecast_steps = min(available_steps, int(bundle.max_forecast_steps))
    else:
        forecast_steps = int(requested_forecast_steps)
        if forecast_steps < 1:
            raise ForcingValidationError("requested_forecast_steps must be >= 1.")
        if forecast_steps > available_steps:
            raise ForcingValidationError(
                f"Requested {forecast_steps} forecast steps but forcing only supports {available_steps}."
            )
        if forecast_steps > bundle.max_forecast_steps:
            raise ForcingValidationError(
                f"Requested {forecast_steps} forecast steps exceeds validated bundle horizon {bundle.max_forecast_steps}."
            )
    return ForcingInput(
        stage=stage.astype(np.float32),
        precipitation=precip.astype(np.float32),
        dt_seconds=int(bundle.dt_seconds),
        forecast_steps=forecast_steps,
        input_hash=digest,
        source_name=source_name,
    )
