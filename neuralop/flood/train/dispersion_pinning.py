"""Dispersion pinning: supervise the aleatory channel against the references.

The diagnosis (Phase A) is that the FGNO's predictive spread is correctly
*sized* and wrongly *attributed*.  Fair CRPS is proper, so a model with location
error is supposed to over-disperse; with a single dispersion channel that
error-covering variance lands in the aleatory term, leaving the epistemic
channel with ~0.5% of total variance and covering the true conditional mean only
17% of the time at nominal 95%.  Because a proper scoring rule on the pooled
predictive is blind to the split (>10x change in epistemic share moved crossed
CRPS by 1.4-3.0%), no reweighting of the score alone can fix this.

This adds the missing information: the HEC-RAS reference ensembles know the true
aleatory dispersion.  Pinning the *within-particle* channel to it, while the
CRPS term still pulls the total toward the CRPS-optimal scale, forces the
remaining variance between particles, where it belongs.

Measured headroom (Phase A5), all in matched RMS convention:
    current within-particle E|X_1 - X_2| = 0.1382 m
    reference target                     = 0.1038 m   (shrink ~25%)
    variance freed                       = 0.0885 m
    epistemic sigma needed (~RMSE)       = 0.0618 m
so pinning frees more than enough variance to calibrate the epistemic channel.

Design notes
------------
* At K=2 the model-side estimator ``|X_1 - X_2|`` is *unbiased* for E|X - X'|;
  no variance estimator is involved, which matters because a 1-dof variance
  estimate would be hopeless.
* Within a stratum the signed deviation is aggregated over CELLS and squared
  once, rather than squared per cell.  Squaring per cell would penalise the
  K=2 sampling noise (large and irreducible) instead of the dispersion
  mismatch.  The deliberate cost is that equal and opposite errors across cells
  in one stratum cancel; the strata are what keep that from hiding structure.
* The square is taken PER PARTICLE and then averaged over particles.  Pooling
  the particle axis before squaring would let one particle's over-dispersion
  cancel another's under-dispersion, so the objective would not actually pin
  each particle's aleatory law -- and it would fail exactly when the method
  starts working and the particles diverge.  The cell-axis argument above does
  not extend to the particle axis: averaging over cells has already removed the
  K=2 noise, so squaring per particle costs nothing.
* Penalty units are m^2.  Calibrate lambda; do not reason about it from the
  CRPS *metric*.  The optimised objective is avg_loss ~1.3e-4, NOT the reported
  train_err ~2.4e-2 -- confusing the two put an earlier estimate about 190x too
  high, and at lambda ~1 the penalty would be roughly 200x the objective and
  would destroy the model on the first step.  Measured penalty is ~5e-3 m^2, so
  a 10-20% share corresponds to lambda of order 1e-3, not order 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Wetness regimes, matching the dispersion audit's strata.
DEFAULT_WET_THRESHOLDS = (0.01, 0.10)


@dataclass
class DispersionPenaltyResult:
    penalty: torch.Tensor
    per_stratum_mean_deviation_m: dict[str, torch.Tensor] = field(default_factory=dict)
    per_stratum_count: dict[str, int] = field(default_factory=dict)


def mean_abs_pairwise(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Unbiased estimate of E|X - X'| from K draws along ``dim``.

    Sums over ORDERED pairs and divides by K(K-1); the i == j terms are zero and
    harmless.  At K=2 this reduces exactly to ``|x_0 - x_1|``.
    """
    k = int(x.shape[dim])
    if k < 2:
        raise ValueError(f"need at least 2 draws along dim={dim} to estimate dispersion, got {k}")
    diff = (x.unsqueeze(dim) - x.unsqueeze(dim + 1)).abs()
    return diff.sum(dim=(dim, dim + 1)) / (k * (k - 1))


def _normalise_mask(mask: torch.Tensor | None, shape: tuple[int, int]) -> torch.Tensor | None:
    if mask is None:
        return None
    m = mask.to(dtype=torch.bool)
    if m.ndim == 1:
        m = m.unsqueeze(0).expand(shape)
    if tuple(m.shape) != tuple(shape):
        raise ValueError(f"structural_dry_mask shape {tuple(m.shape)} incompatible with {shape}")
    return m


def dispersion_pinning_penalty(
    predictions: torch.Tensor,
    target: torch.Tensor,
    reference_dispersion: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor | None = None,
    wet_thresholds: tuple[float, float] = DEFAULT_WET_THRESHOLDS,
    channel: int = 0,
    stratify_by: torch.Tensor | None = None,
) -> DispersionPenaltyResult:
    """Squared per-stratum mismatch between model and reference dispersion.

    Parameters
    ----------
    predictions : ``[M, K, B, N, C]`` nested particle/aleatory predictions.
    target : ``[B, N, C]`` observed frame, used only to assign wetness strata.
    reference_dispersion : ``[B, N]`` E|H - H'| for each (family, time).
    """
    if predictions.ndim != 5:
        raise ValueError("predictions must have shape [M, K, B, N, C].")
    m, _, b, n, _ = predictions.shape
    if target.shape[0] != b or target.shape[1] != n:
        raise ValueError(f"target {tuple(target.shape)} incompatible with predictions [B={b}, N={n}].")
    if tuple(reference_dispersion.shape) != (b, n):
        raise ValueError(
            f"reference_dispersion must be [B, N] = {(b, n)}, got {tuple(reference_dispersion.shape)}."
        )

    x = predictions[..., channel]                                   # [M, K, B, N]
    model_dispersion = mean_abs_pairwise(x, dim=1)                  # [M, B, N]
    ref = reference_dispersion.to(device=x.device, dtype=x.dtype).unsqueeze(0)
    deviation = model_dispersion - ref                              # [M, B, N]

    # Strata come from the reference-ensemble MEAN when it is supplied, which is
    # what the offline calibration uses.  Falling back to the single sampled
    # target member would stratify on a noisy draw, so the penalty being tuned
    # would not be the penalty being optimised.
    y = target[..., channel] if stratify_by is None else stratify_by
    valid = torch.isfinite(y)
    dry = _normalise_mask(structural_dry_mask, (b, n))
    if dry is not None:
        valid = valid & (~dry)

    lo, hi = float(wet_thresholds[0]), float(wet_thresholds[1])
    strata = {
        "dry": y <= lo,
        "front": (y > lo) & (y <= hi),
        "deep": y > hi,
    }

    penalty = predictions.new_zeros(())
    means: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    for name, selector in strata.items():
        cell_weight = (selector & valid).to(deviation.dtype)              # [B, N]
        weight = cell_weight.unsqueeze(0).expand_as(deviation)            # [M, B, N]
        # Aggregate over CELLS within each particle, square PER PARTICLE, then
        # average over particles.  Pooling particles before squaring would let
        # them cancel each other and would not pin any particle's aleatory law.
        per_particle_count = weight.sum(dim=(1, 2))                       # [M]
        per_particle_mean = (deviation * weight).sum(dim=(1, 2)) / per_particle_count.clamp_min(1.0)
        penalty = penalty + per_particle_mean.square().mean()
        means[name] = per_particle_mean.mean().detach()
        counts[name] = int(cell_weight.sum().detach().item())

    return DispersionPenaltyResult(
        penalty=penalty,
        per_stratum_mean_deviation_m=means,
        per_stratum_count=counts,
    )
