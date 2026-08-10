"""Per-family HEC-RAS reference dispersion, for supervising the aleatory channel.

Fair CRPS is proper, so under misspecification it correctly over-disperses: a
forecast with location error ``b`` against truth spread ``tau`` has a
CRPS-optimal scale satisfying

    sigma / sqrt(sigma^2 + tau^2) = exp(b^2 / (2(sigma^2 + tau^2))) / sqrt(2),

which returns ``sigma = tau`` only at ``b = 0``.  Stage-1 carries a single
dispersion channel, so that error-covering variance was booked as aleatory and
the epistemic channel kept ~0.5% of total variance.  No proper scoring rule on
the *pooled* predictive can separate the two -- the reallocation study moved the
epistemic share by more than 10x and shifted crossed CRPS by 1.4-3.0% -- so the
identifying information has to come from outside the score.

The reference ensembles are that information.  This module serves the table
built by ``scripts/build_reference_dispersion_table.py``:

    D_ref[family, t, cell] = E|H - H'| = (1/(R(R-1))) sum_{r != r'} |H_r - H_r'|

which is exactly the functional fair CRPS already uses as its self-distance,
and which the model side estimates without bias at K=2 by ``|X_1 - X_2|``.

The table is ~1.3 GiB, so it is deliberately NOT an ``nn.Module`` buffer: it
must never be swept into ``state_dict`` (checkpoint bloat) or moved to GPU by a
stray ``.to(device)``.  It stays on CPU and lookups return small ``[B, Nv]``
slices that the caller transfers.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

SCHEMA = "flood_reference_dispersion_v1"

# Measured on the held-out references (probe A5): sigma / E|H-H'| = 0.9705,
# stable across regimes (0.97049 all cells, 0.97039 wet).  The Gaussian value
# 0.8862 is wrong by 9% on these skewed, non-negative depths -- do not use it.
SIGMA_OVER_MEAN_ABS_DIFFERENCE = 0.9705

# Measured in probe A5: training families carry 1.070x the held-out dispersion
# (1.087 on wet cells).  Recorded as an observation ONLY -- it is deliberately
# NOT applied by default.
#
# An earlier version divided the training target by this ratio so the pinned
# scale matched the held-out scale.  That is target leakage: it lets a property
# of the evaluation set determine a training target.  It is also wrong on its
# own terms -- the aleatory channel should reproduce the aleatory law of the
# data it is modelling, which for training families is their own dispersion.
# Any run that needs the held-out scale must pass `scale` explicitly and cannot
# then be reported as held-out evidence.
TRAIN_OVER_TEST_DISPERSION = 1.070


def load_reference_dispersion_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path).resolve(), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Reference-dispersion artifact must be a dict, got {type(payload)!r}.")
    for key in ("family_ids", "dispersion", "cell_count", "n_time"):
        if key not in payload:
            raise KeyError(f"Reference-dispersion artifact is missing {key!r}.")
    if str(payload.get("schema", SCHEMA)) != SCHEMA:
        raise ValueError(f"Unexpected artifact schema {payload.get('schema')!r}; expected {SCHEMA!r}.")
    payload["dispersion"] = payload["dispersion"].to(dtype=torch.float32, device="cpu")
    for key in ("reference_mean", "reference_mean_variance"):
        if payload.get(key) is not None:
            payload[key] = payload[key].to(dtype=torch.float32, device="cpu")
    return payload


def validate_reference_dispersion_artifact(
    artifact: dict[str, Any],
    *,
    expected_cell_count: int | None = None,
    expected_family_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    disp = artifact["dispersion"]
    families = [str(v) for v in artifact["family_ids"]]
    if disp.ndim != 3:
        raise ValueError(f"dispersion must be [F, T, Nv], got {tuple(disp.shape)}.")
    if disp.shape[0] != len(families):
        raise ValueError(f"dispersion has {disp.shape[0]} families but {len(families)} ids.")
    if len(set(families)) != len(families):
        raise ValueError("Reference-dispersion artifact has duplicate family ids.")
    if int(disp.shape[2]) != int(artifact["cell_count"]):
        raise ValueError(f"dispersion cell axis {disp.shape[2]} != cell_count {artifact['cell_count']}.")
    if int(disp.shape[1]) != int(artifact["n_time"]):
        raise ValueError(f"dispersion time axis {disp.shape[1]} != n_time {artifact['n_time']}.")
    if torch.any(disp < 0):
        raise ValueError("Reference dispersion must be non-negative.")
    if not torch.isfinite(disp).all():
        raise ValueError("Reference dispersion contains non-finite values.")
    for key in ("reference_mean", "reference_mean_variance"):
        value = artifact.get(key)
        if value is None:
            continue
        if tuple(value.shape) != tuple(disp.shape):
            raise ValueError(
                f"{key} must match dispersion shape {tuple(disp.shape)}, "
                f"got {tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains non-finite values.")
    mean_variance = artifact.get("reference_mean_variance")
    if mean_variance is not None and torch.any(mean_variance < 0):
        raise ValueError("reference_mean_variance must be non-negative.")
    if expected_cell_count is not None and int(artifact["cell_count"]) != int(expected_cell_count):
        raise ValueError(
            f"Reference-dispersion cell_count={artifact['cell_count']} "
            f"does not match expected {expected_cell_count}."
        )
    if expected_family_ids is not None:
        missing = sorted(set(str(f) for f in expected_family_ids) - set(families))
        if missing:
            raise ValueError(
                f"Reference-dispersion artifact is missing {len(missing)} required families "
                f"(e.g. {missing[:3]})."
            )
    return {
        "n_families": len(families),
        "n_time": int(artifact["n_time"]),
        "cell_count": int(artifact["cell_count"]),
        "dispersion_mean_m": float(disp.mean()),
    }


class ReferenceDispersionTable:
    """Family/time-indexed lookup of ``E|H - H'|``, held on CPU.

    ``scale`` multiplies the stored dispersion and defaults to 1.0: the target
    is each training family's OWN reference dispersion.  Rescaling toward the
    held-out population would be target leakage -- see the note on
    ``TRAIN_OVER_TEST_DISPERSION``.
    """

    def __init__(
        self,
        *,
        family_ids: Sequence[str],
        dispersion: torch.Tensor,
        reference_mean: torch.Tensor | None = None,
        reference_mean_variance: torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> None:
        self.family_ids = [str(v) for v in family_ids]
        self._index = {f: i for i, f in enumerate(self.family_ids)}
        self.dispersion = dispersion.to(dtype=torch.float32, device="cpu")
        # The reference mean defines the wetness strata.  Training must use the
        # same stratification as the offline calibration, otherwise the penalty
        # being tuned is not the penalty being optimised; stratifying on the
        # single sampled target member would use a noisy draw instead.
        self.reference_mean = (
            None if reference_mean is None
            else reference_mean.to(dtype=torch.float32, device="cpu")
        )
        self.reference_mean_variance = (
            None if reference_mean_variance is None
            else reference_mean_variance.to(dtype=torch.float32, device="cpu")
        )
        self.scale = float(scale)
        self.n_time = int(self.dispersion.shape[1])
        self.cell_count = int(self.dispersion.shape[2])

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any], **kwargs) -> "ReferenceDispersionTable":
        return cls(
            family_ids=artifact["family_ids"],
            dispersion=artifact["dispersion"],
            reference_mean=artifact.get("reference_mean"),
            reference_mean_variance=artifact.get("reference_mean_variance"),
            **kwargs,
        )

    def lookup(
        self,
        family_ids: Sequence[str],
        time_index: torch.Tensor | Sequence[int],
        *,
        step: int = 0,
    ) -> torch.Tensor:
        """Return ``[B, Nv]`` reference dispersion for each (family, t + step).

        ``target_sequence[:, s]`` is the frame at raw HDF time ``time_index + s``
        (``__getitem__`` reads ``wd[n_history + s]`` from a slice starting at
        ``target_t - n_history``), so ``step`` indexes AR rollout steps directly.
        """
        rows, times = self._rows_and_times(family_ids, time_index, step)
        return self.dispersion[rows, times] * self.scale

    def _rows_and_times(self, family_ids, time_index, step):
        try:
            rows = torch.tensor([self._index[str(v)] for v in family_ids], dtype=torch.long)
        except KeyError as exc:
            raise KeyError(f"Unknown family id in reference dispersion table: {exc.args[0]!r}.") from exc
        times = torch.as_tensor(
            [int(v) for v in (time_index.tolist() if torch.is_tensor(time_index) else time_index)],
            dtype=torch.long,
        ) + int(step)
        if rows.numel() != times.numel():
            raise ValueError(f"family_ids ({rows.numel()}) and time_index ({times.numel()}) disagree.")
        if torch.any(times < 0) or torch.any(times >= self.n_time):
            bad = times[(times < 0) | (times >= self.n_time)][:3].tolist()
            raise IndexError(f"time index out of range [0, {self.n_time}) for this table: {bad}")
        return rows, times

    def lookup_reference_mean(
        self,
        family_ids: Sequence[str],
        time_index: torch.Tensor | Sequence[int],
        *,
        step: int = 0,
    ) -> torch.Tensor | None:
        """``[B, Nv]`` reference-ensemble mean, for wetness stratification."""
        if self.reference_mean is None:
            return None
        rows, times = self._rows_and_times(family_ids, time_index, step)
        return self.reference_mean[rows, times]

    def lookup_reference_mean_variance(
        self,
        family_ids: Sequence[str],
        time_index: torch.Tensor | Sequence[int],
        *,
        step: int = 0,
    ) -> torch.Tensor | None:
        """Return the exact sample-mean variance estimate ``s^2/R`` in m2."""
        if self.reference_mean_variance is None:
            return None
        rows, times = self._rows_and_times(family_ids, time_index, step)
        return self.reference_mean_variance[rows, times]
