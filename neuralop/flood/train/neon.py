"""Training helpers for NEON-aligned Stage-2 FGNO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from neuralop.flood.neon import (
    NEONEpistemicCorrection,
    NEONStage2LossOutput,
    NEONStage2LossWeights,
    compute_stage2_loss,
    freeze_stage1_model,
    sample_epistemic_indices,
)


@dataclass
class FrozenFGNOFeatureBatch:
    """Frozen Stage-1 outputs for one Stage-2 optimization batch."""

    base_prediction: torch.Tensor
    features: torch.Tensor
    aleatory_latents: torch.Tensor


def assert_optimizer_excludes_stage1(
    *,
    stage1_model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Fail early if a Stage-2 optimizer accidentally owns Stage-1 parameters."""

    stage1_ids = {id(param) for param in stage1_model.parameters()}
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group.get("params", [])):
            if id(param) in stage1_ids:
                raise ValueError(
                    "Stage-2 optimizer includes frozen Stage-1 parameter "
                    f"(group={group_idx}, param={param_idx})."
                )


def _latent_for_member(aleatory_latents: torch.Tensor, member_idx: int, batch_size: int) -> torch.Tensor:
    latent = aleatory_latents[int(member_idx)]
    if latent.ndim == 1:
        latent = latent.unsqueeze(0).expand(batch_size, -1)
    elif latent.ndim == 2 and latent.shape[0] == 1 and batch_size > 1:
        latent = latent.expand(batch_size, -1)
    elif latent.ndim != 2 or latent.shape[0] != batch_size:
        raise ValueError(
            "aleatory_latents must have shape [K, d_a] or [K, B, d_a], "
            f"got {tuple(aleatory_latents.shape)} for batch_size={batch_size}."
        )
    return latent


def _ensure_single_rollout_time_dim(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize one-step FGNO tensors to the Stage-2 rollout contract.

    GINO one-step outputs are usually ``[B, Nv, C]`` before stacking over
    aleatory members. After stacking they become ``[B, K, Nv, C]``. Stage 2
    consistently consumes rollout tensors ``[B, K, T, Nv, C]``; for one-step
    feature extraction the implicit rollout length is ``T=1``.
    """

    if tensor.ndim == 4:
        return tensor.unsqueeze(2)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(
        f"{name} must have shape [B, K, Nv, C] or [B, K, T, Nv, C], "
        f"got {tuple(tensor.shape)}."
    )


def collect_frozen_fgno_features(
    *,
    stage1_model: nn.Module,
    model_kwargs: dict[str, Any],
    aleatory_latents: torch.Tensor,
    feature_source: str = "decoder_pre_projection",
) -> FrozenFGNOFeatureBatch:
    """Run frozen FGNO over ``K`` aleatory latents and collect decoder features.

    ``model_kwargs`` should contain the normal GINO forward inputs except
    ``ada_in``, ``return_features``, and ``feature_source``.
    """

    freeze_stage1_model(stage1_model)
    x = model_kwargs.get("x")
    if x is None:
        raise ValueError("model_kwargs must contain x so batch size can be inferred.")
    batch_size = int(x.shape[0])
    if aleatory_latents.ndim not in {2, 3}:
        raise ValueError(
            "aleatory_latents must have shape [K, d_a] or [K, B, d_a], "
            f"got {tuple(aleatory_latents.shape)}."
        )
    base_members = []
    feature_members = []
    with torch.no_grad():
        for member_idx in range(int(aleatory_latents.shape[0])):
            latent = _latent_for_member(aleatory_latents, member_idx, batch_size)
            out = stage1_model(
                **model_kwargs,
                ada_in=latent,
                return_features=True,
                feature_source=feature_source,
            )
            if not isinstance(out, dict) or "prediction" not in out or "features" not in out:
                raise TypeError(
                    "Stage-1 model must return a feature dictionary when return_features=True."
                )
            features = out["features"].get(feature_source)
            if features is None and str(feature_source).strip().lower() == "all":
                # GINO returns a dictionary of all requested features; Stage-2 v1
                # consumes the decoder-pre-projection tensor from that bundle.
                features = out["features"].get("decoder_pre_projection")
            if features is None:
                raise KeyError(f"Stage-1 output did not include feature_source={feature_source!r}.")
            base_members.append(out["prediction"].detach())
            feature_members.append(features.detach())
    base_prediction = _ensure_single_rollout_time_dim(
        torch.stack(base_members, dim=1),
        name="base_prediction",
    )
    features = _ensure_single_rollout_time_dim(
        torch.stack(feature_members, dim=1),
        name="features",
    )
    return FrozenFGNOFeatureBatch(
        base_prediction=base_prediction,
        features=features,
        aleatory_latents=aleatory_latents.detach(),
    )


def neon_stage2_training_step(
    *,
    module: NEONEpistemicCorrection,
    optimizer: torch.optim.Optimizer,
    base_prediction: torch.Tensor,
    features: torch.Tensor,
    reference: torch.Tensor,
    z_e: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    zero_threshold: float | torch.Tensor = 0.0,
    loss_weights: NEONStage2LossWeights | None = None,
    grad_clip_norm: float | None = None,
) -> NEONStage2LossOutput:
    """Perform one Stage-2 optimization step from frozen FGNO tensors."""

    module.train()
    if z_e is None:
        z_e = sample_epistemic_indices(
            4,
            module.epistemic_dim,
            device=features.device,
            dtype=features.dtype,
        )
    optimizer.zero_grad(set_to_none=True)
    out = module(base_prediction, features, z_e)
    losses = compute_stage2_loss(
        prediction=out.prediction,
        reference=reference,
        correction=out.correction,
        module=module,
        weights=weights,
        edge_index=edge_index,
        edge_weights=edge_weights,
        zero_threshold=zero_threshold,
        loss_weights=loss_weights,
    )
    losses.total.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(module.trainable_branch.parameters(), float(grad_clip_norm))
    optimizer.step()
    return losses


def neon_stage2_eval_forward(
    *,
    module: NEONEpistemicCorrection,
    base_prediction: torch.Tensor,
    features: torch.Tensor,
    z_e: torch.Tensor,
) -> torch.Tensor:
    """Inference helper returning nested corrected predictions only."""

    module.eval()
    with torch.no_grad():
        return module(base_prediction, features, z_e).prediction


def build_neon_stage2_optimizer(
    module: NEONEpistemicCorrection,
    *,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
) -> torch.optim.Optimizer:
    """Build the default optimizer for Stage-2 trainable EpiNet parameters."""

    return torch.optim.AdamW(
        [param for param in module.parameters() if param.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
