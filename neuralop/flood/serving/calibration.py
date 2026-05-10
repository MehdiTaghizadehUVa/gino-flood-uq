"""Calibration adapter for serving-time FGN forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from neuralop.flood.eval.scientific_calibration import (
    apply_crps_mbm_to_wd_members,
    apply_isotonic_exceedance_probability,
)
from neuralop.flood.serving.products import ForecastResult


class CalibrationAdapter:
    """Apply CRPS-MBM and isotonic calibration artifacts to forecast products."""

    def __init__(self, *, crps_mbm: Dict[str, Any], isotonic: Dict[str, Any] | None = None) -> None:
        self.crps_mbm = crps_mbm
        self.isotonic = isotonic or {}

    @classmethod
    def from_files(cls, coefficients_path: str | Path, isotonic_path: str | Path) -> "CalibrationAdapter":
        with Path(coefficients_path).open("r", encoding="utf-8") as handle:
            coeff = json.load(handle)
        with Path(isotonic_path).open("r", encoding="utf-8") as handle:
            iso = json.load(handle)
        return cls(crps_mbm=coeff, isotonic=iso)

    def has_isotonic_curves(self) -> bool:
        """True when isotonic exceedance curves are loaded and applicable."""
        return bool(self.isotonic) and bool(self.isotonic.get("curves"))

    def apply(self, raw_forecast: ForecastResult) -> ForecastResult:
        raw_forecast = raw_forecast.validate()
        members = np.asarray(raw_forecast.members_wd, dtype=np.float64)
        calibrated = np.empty_like(members)
        wettable = raw_forecast.wettable_mask
        for t, lead_hour in enumerate(raw_forecast.lead_time_hours):
            calibrated[:, t, :] = apply_crps_mbm_to_wd_members(
                members[:, t, :],
                lead_time_hour=float(lead_hour),
                calibration_model=self.crps_mbm,
                wettable_mask=wettable,
            )
        return ForecastResult(
            members_wd=np.clip(calibrated, 0.0, None).astype(np.float32),
            lead_time_hours=np.asarray(raw_forecast.lead_time_hours, dtype=np.float32),
            wettable_mask=raw_forecast.wettable_mask,
            metadata={**(raw_forecast.metadata or {}), "calibration": "crps_member_by_member"},
        )

    def apply_isotonic_exceedance(
        self,
        raw_probability_wettable: np.ndarray,
        *,
        threshold_m: float,
        lead_time_hour: float | np.ndarray,
        wettable_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Apply isotonic calibration to a per-wettable-cell exceedance probability map.

        The input array contains raw exceedance probabilities for the wettable cells
        only (the canonical product output). When isotonic curves are unavailable,
        the input is returned unchanged.

        - 1D input: shape ``[n_wettable_cells]``; ``lead_time_hour`` is a scalar.
        - 2D input: shape ``[n_time, n_wettable_cells]``; ``lead_time_hour`` is an
          array of length ``n_time``.

        ``wettable_mask`` indexes ``crps_mbm["wet_frequency_by_cell"]`` to pull the
        per-wettable-cell wet-frequency required by the bin lookup.
        """
        raw = np.asarray(raw_probability_wettable, dtype=np.float64)
        if not self.has_isotonic_curves():
            return raw.copy()
        wet_freq_full = np.asarray(
            self.crps_mbm.get("wet_frequency_by_cell", []),
            dtype=np.float64,
        )
        if wet_freq_full.size == 0:
            return raw.copy()
        if wettable_mask is None:
            wet_freq = wet_freq_full
        else:
            mask = np.asarray(wettable_mask, dtype=bool)
            if mask.shape[0] != wet_freq_full.shape[0]:
                raise ValueError(
                    "wettable_mask length must match crps_mbm wet_frequency_by_cell length."
                )
            wet_freq = wet_freq_full[mask]
        n_wettable = raw.shape[-1]
        if wet_freq.shape[0] != n_wettable:
            raise ValueError(
                "raw_probability_wettable last dimension must match wettable cell count."
            )
        if raw.ndim == 1:
            return apply_isotonic_exceedance_probability(
                raw,
                threshold_m=float(threshold_m),
                lead_time_hour=float(lead_time_hour),
                wet_frequency_by_cell=wet_freq,
                isotonic_model=self.isotonic,
            )
        if raw.ndim != 2:
            raise ValueError(
                f"raw_probability_wettable must be 1D or 2D; got ndim={raw.ndim}."
            )
        lead_arr = np.asarray(lead_time_hour, dtype=np.float64).reshape(-1)
        if lead_arr.shape[0] != raw.shape[0]:
            raise ValueError(
                "lead_time_hour must have length equal to raw_probability_wettable axis 0 for 2D input."
            )
        out = np.empty_like(raw)
        for t_idx in range(raw.shape[0]):
            out[t_idx] = apply_isotonic_exceedance_probability(
                raw[t_idx],
                threshold_m=float(threshold_m),
                lead_time_hour=float(lead_arr[t_idx]),
                wet_frequency_by_cell=wet_freq,
                isotonic_model=self.isotonic,
            )
        return out
