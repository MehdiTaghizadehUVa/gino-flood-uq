"""Evaluation helpers for NEON Stage-2 FGNO predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from neuralop.flood.neon import (
    anova_corrected_epistemic_variance,
    flatten_nested_predictions,
    nested_member_metadata,
    nested_variance_components,
)


@dataclass
class NEONFlattenedArtifacts:
    """Legacy-compatible flattened ensemble plus nested member metadata."""

    prediction: torch.Tensor
    member_id: torch.Tensor
    member_epistemic_id: torch.Tensor
    member_aleatory_id: torch.Tensor

    def metadata_numpy(self) -> dict[str, np.ndarray]:
        return {
            "member_id": self.member_id.detach().cpu().numpy(),
            "member_epistemic_id": self.member_epistemic_id.detach().cpu().numpy(),
            "member_aleatory_id": self.member_aleatory_id.detach().cpu().numpy(),
        }


def flatten_for_legacy_metrics(prediction: torch.Tensor) -> NEONFlattenedArtifacts:
    """Flatten ``[B, M, K, T, Nv, C]`` predictions for existing metric code."""

    if prediction.ndim != 6:
        raise ValueError(
            "prediction must have shape [B, M, K, T, Nv, C], "
            f"got {tuple(prediction.shape)}."
        )
    _, M, K, _, _, _ = prediction.shape
    metadata = nested_member_metadata(int(M), int(K))
    return NEONFlattenedArtifacts(
        prediction=flatten_nested_predictions(prediction),
        member_id=metadata["member_id"].to(prediction.device),
        member_epistemic_id=metadata["member_epistemic_id"].to(prediction.device),
        member_aleatory_id=metadata["member_aleatory_id"].to(prediction.device),
    )


def neon_variance_summary(prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute nested variance fields for NEON-FGNO diagnostics."""

    components = nested_variance_components(prediction)
    corrected = anova_corrected_epistemic_variance(prediction)
    ratio = corrected / (components.aleatory + corrected).clamp_min(1.0e-12)
    return {
        "variance_aleatory": components.aleatory,
        "variance_epistemic": components.epistemic,
        "variance_epistemic_anova_corrected": corrected,
        "variance_total": components.total,
        "variance_epistemic_fraction_anova_corrected": ratio,
    }


def domain_average_variance_summary(
    prediction: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Average NEON variance fields over mesh/time/channel dimensions."""

    summary = neon_variance_summary(prediction)
    if weights is None:
        return {key: value.mean(dim=(1, 2, 3)) for key, value in summary.items()}
    weights = weights.to(device=prediction.device, dtype=prediction.dtype)
    if weights.ndim == 3:
        weights = weights.unsqueeze(0)
    if weights.ndim != 4:
        raise ValueError(
            "weights must have shape [T, Nv, C] or [B, T, Nv, C], "
            f"got {tuple(weights.shape)}."
        )
    denom = weights.sum(dim=(1, 2, 3)).clamp_min(1.0e-12)
    out = {}
    for key, value in summary.items():
        if weights.shape[0] == 1 and value.shape[0] > 1:
            w = weights.expand(value.shape[0], -1, -1, -1)
            d = denom.expand(value.shape[0])
        else:
            w = weights
            d = denom
        out[key] = (value * w).sum(dim=(1, 2, 3)) / d
    return out


def _as_numpy_array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_nested_forecast_artifact(
    path,
    *,
    hydrograph_id: str,
    prediction: torch.Tensor,
    ref_members_wd,
    **artifact_kwargs,
):
    """Write a NEON nested prediction artifact through the common HDF5 schema.

    ``prediction`` must have shape ``[B, M, K, T, Nv, C]``.  The common artifact
    schema is per hydrograph, so this adapter requires ``B=1`` and depth-only
    ``C=1`` and flattens ``M*K`` forecast members while preserving the nested
    epistemic and aleatory member IDs.
    """

    if prediction.ndim != 6:
        raise ValueError(
            "prediction must have shape [B, M, K, T, Nv, C], "
            f"got {tuple(prediction.shape)}."
        )
    if prediction.shape[0] != 1:
        raise ValueError("save_nested_forecast_artifact expects one hydrograph at a time (B=1).")
    if prediction.shape[-1] != 1:
        raise ValueError("save_nested_forecast_artifact currently supports depth-only C=1 artifacts.")

    flattened = flatten_for_legacy_metrics(prediction)
    pred_members_wd = flattened.prediction[0, :, :, :, 0].detach().cpu().numpy()
    ref = _as_numpy_array(ref_members_wd)
    if ref.ndim == 4 and ref.shape[-1] == 1:
        ref = ref[..., 0]
    if ref.ndim != 3:
        raise ValueError(
            "ref_members_wd must have shape [R, T, Nv] or [R, T, Nv, 1], "
            f"got {tuple(ref.shape)}."
        )

    from neuralop.flood.eval.scientific_calibration import save_forecast_artifact

    kwargs = dict(artifact_kwargs)
    kwargs.setdefault("member_epistemic_id", flattened.member_epistemic_id.detach().cpu().numpy().tolist())
    kwargs.setdefault("member_aleatory_id", flattened.member_aleatory_id.detach().cpu().numpy().tolist())
    return save_forecast_artifact(
        path,
        hydrograph_id=hydrograph_id,
        pred_members_wd=pred_members_wd,
        ref_members_wd=ref,
        **kwargs,
    )

