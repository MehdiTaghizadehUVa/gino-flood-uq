"""NEON-aligned Stage-2 epistemic utilities for frozen FGNO models.

This module intentionally keeps Stage-2 logic behind a small offline research
interface.  It does not change Stage-1 FGNO training behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class NEONCorrectionOutput:
    """Nested Stage-2 prediction bundle.

    Shapes follow ``[B, M, K, T, Nv, C]`` where ``M`` indexes epistemic
    particles and ``K`` indexes existing FGNO aleatory latent samples.
    """

    prediction: torch.Tensor
    correction: torch.Tensor
    trainable_correction: torch.Tensor
    prior_correction: torch.Tensor


@dataclass
class NestedVarianceComponents:
    """Nested aleatory/epistemic variance estimates on the prediction grid."""

    aleatory: torch.Tensor
    epistemic: torch.Tensor
    total: torch.Tensor


@dataclass
class NEONStage2LossWeights:
    """Weights for the Stage-2 objective terms."""

    rpf: float = 1.0e-4
    smooth: float = 1.0e-3
    time: float = 1.0
    pos: float = 1.0e-2
    mag: float = 1.0e-4


@dataclass
class NEONStage2LossOutput:
    """Stage-2 loss terms for logging and optimization."""

    total: torch.Tensor
    fit: torch.Tensor
    rpf: torch.Tensor
    graph: torch.Tensor
    time: torch.Tensor
    pos: torch.Tensor
    mag: torch.Tensor


def freeze_stage1_model(model: nn.Module) -> nn.Module:
    """Freeze a pretrained Stage-1 model for NEON Stage-2 training."""

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def sample_epistemic_indices(
    num_particles: int,
    epistemic_dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample standard-normal NEON/EpiNet epistemic indices."""

    if int(num_particles) < 1:
        raise ValueError("num_particles must be >= 1.")
    if int(epistemic_dim) < 1:
        raise ValueError("epistemic_dim must be >= 1.")
    target_device = torch.device(device)
    # A torch.Generator is bound to a device; torch.randn forbids a generator
    # whose device differs from the requested device. When they differ (e.g. a
    # CPU generator driving CUDA sampling), draw on the generator's device then
    # move -- preserving the reproducible sample across the CPU/GPU boundary.
    if generator is not None and generator.device.type != target_device.type:
        sampled = torch.randn(
            int(num_particles), int(epistemic_dim), dtype=dtype, generator=generator
        )
        return sampled.to(target_device)
    return torch.randn(
        int(num_particles),
        int(epistemic_dim),
        device=target_device,
        dtype=dtype,
        generator=generator,
    )


class _LeadTimeEncoder(nn.Module):
    """Maps a rollout lead-time index to a per-timestep modulation bias.

    Deterministic in the lead time, so it adds temporally-coherent lead-
    dependent structure to the correction rather than iid node noise. The same
    ``z_e`` can therefore induce corrections that vary smoothly with lead time.
    """

    def __init__(self, lead_time_dim: int, hidden_channels: int) -> None:
        super().__init__()
        self.lead_time_dim = int(lead_time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.lead_time_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    @staticmethod
    def _sinusoidal(t_norm: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        device, dtype = t_norm.device, t_norm.dtype
        if half >= 1:
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=device, dtype=dtype)
                / max(half - 1, 1)
            )
            args = t_norm.view(-1, 1) * freqs.view(1, -1) * (2.0 * math.pi)
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        else:
            emb = t_norm.view(-1, 1)
        if emb.shape[1] < dim:
            pad = torch.zeros(emb.shape[0], dim - emb.shape[1], device=device, dtype=dtype)
            emb = torch.cat([emb, pad], dim=1)
        return emb[:, :dim]

    def forward(self, n_time: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        n_time = int(n_time)
        if n_time > 1:
            t_norm = torch.linspace(0.0, 1.0, n_time, device=device, dtype=dtype)
        else:
            t_norm = torch.zeros(1, device=device, dtype=dtype)
        emb = self._sinusoidal(t_norm, self.lead_time_dim)  # [T, lead_time_dim]
        return self.mlp(emb)  # [T, hidden]


class _CorrectionBranch(nn.Module):
    """Pointwise feature-conditioned correction branch.

    The branch is shared over time and mesh nodes.  A rollout-global epistemic
    index modulates coherent feature fields rather than adding iid node noise.
    An optional lead-time encoder adds temporally-coherent, lead-dependent
    structure so the same ``z_e`` can induce lead-varying corrections.
    """

    def __init__(
        self,
        *,
        feature_channels: int,
        out_channels: int,
        epistemic_dim: int,
        hidden_channels: int,
        n_hidden_layers: int = 2,
        lead_time_dim: int = 0,
    ) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be >= 1.")
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1.")
        if epistemic_dim < 1:
            raise ValueError("epistemic_dim must be >= 1.")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1.")

        layers: list[nn.Module] = [nn.Linear(feature_channels, hidden_channels), nn.SiLU()]
        for _ in range(max(0, int(n_hidden_layers) - 1)):
            layers.extend([nn.Linear(hidden_channels, hidden_channels), nn.SiLU()])
        self.feature_encoder = nn.Sequential(*layers)
        self.epistemic_encoder = nn.Sequential(
            nn.Linear(epistemic_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2 * hidden_channels),
        )
        self.output_head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.lead_time_encoder = (
            _LeadTimeEncoder(int(lead_time_dim), hidden_channels)
            if int(lead_time_dim) > 0
            else None
        )

    def forward(self, features: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "features must have shape [B, K, T, Nv, C_phi], "
                f"got {tuple(features.shape)}."
            )
        if z_e.ndim != 2:
            raise ValueError(f"z_e must have shape [M, d_e], got {tuple(z_e.shape)}.")
        feature_hidden = self.feature_encoder(features)
        gamma_beta = self.epistemic_encoder(z_e)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        view_shape = (1, z_e.shape[0], 1, 1, 1, gamma.shape[-1])
        gamma = gamma.view(view_shape)
        beta = beta.view(view_shape)
        modulated = feature_hidden.unsqueeze(1) * (1.0 + gamma) + beta
        if self.lead_time_encoder is not None:
            n_time = int(features.shape[2])
            lead = self.lead_time_encoder(n_time, feature_hidden.device, feature_hidden.dtype)
            modulated = modulated + lead.view(1, 1, 1, n_time, 1, lead.shape[-1])
        return self.output_head(modulated)


class NEONEpistemicCorrection(nn.Module):
    """Randomized-prior EpiNet correction head for frozen FGNO features."""

    def __init__(
        self,
        *,
        feature_channels: int,
        out_channels: int,
        epistemic_dim: int,
        hidden_channels: int = 64,
        alpha: float = 0.1,
        n_hidden_layers: int = 2,
        detach_features: bool = True,
        lead_time_dim: int = 0,
        za_dependent: bool = True,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.out_channels = int(out_channels)
        self.epistemic_dim = int(epistemic_dim)
        self.hidden_channels = int(hidden_channels)
        self.n_hidden_layers = int(n_hidden_layers)
        self.alpha = float(alpha)
        self.detach_features = bool(detach_features)
        self.lead_time_dim = int(lead_time_dim)
        self.za_dependent = bool(za_dependent)
        self.trainable_branch = _CorrectionBranch(
            feature_channels=self.feature_channels,
            out_channels=self.out_channels,
            epistemic_dim=self.epistemic_dim,
            hidden_channels=self.hidden_channels,
            n_hidden_layers=n_hidden_layers,
            lead_time_dim=self.lead_time_dim,
        )
        self.prior_branch = _CorrectionBranch(
            feature_channels=self.feature_channels,
            out_channels=self.out_channels,
            epistemic_dim=self.epistemic_dim,
            hidden_channels=self.hidden_channels,
            n_hidden_layers=n_hidden_layers,
            lead_time_dim=self.lead_time_dim,
        )
        for param in self.prior_branch.parameters():
            param.requires_grad_(False)

    def set_prior_scale(self, alpha: float) -> None:
        """Set the randomized-prior scale ``alpha`` (e.g. from auto-calibration)."""
        self.alpha = float(alpha)

    def forward(
        self,
        base_prediction: torch.Tensor,
        features: torch.Tensor,
        z_e: torch.Tensor,
    ) -> NEONCorrectionOutput:
        if base_prediction.ndim != 5:
            raise ValueError(
                "base_prediction must have shape [B, K, T, Nv, C], "
                f"got {tuple(base_prediction.shape)}."
            )
        if features.shape[:4] != base_prediction.shape[:4]:
            raise ValueError(
                "features and base_prediction must agree on [B, K, T, Nv]: "
                f"{tuple(features.shape[:4])} != {tuple(base_prediction.shape[:4])}."
            )
        if features.shape[-1] != self.feature_channels:
            raise ValueError(
                f"features last dimension must be {self.feature_channels}, "
                f"got {features.shape[-1]}."
            )
        if base_prediction.shape[-1] != self.out_channels:
            raise ValueError(
                f"base_prediction last dimension must be {self.out_channels}, "
                f"got {base_prediction.shape[-1]}."
            )
        if z_e.shape[-1] != self.epistemic_dim:
            raise ValueError(
                f"z_e last dimension must be {self.epistemic_dim}, got {z_e.shape[-1]}."
            )

        branch_features = features.detach() if self.detach_features else features
        base = base_prediction.detach() if self.detach_features else base_prediction
        if not self.za_dependent:
            # za-independent ablation: average the frozen features over the
            # aleatory (K) axis so the correction depends only on the conditional
            # mean feature and is shared across aleatory members.
            branch_features = branch_features.mean(dim=1, keepdim=True).expand_as(branch_features)
        trainable = self.trainable_branch(branch_features, z_e)
        with torch.no_grad():
            prior = self.prior_branch(branch_features, z_e)
        correction = trainable + self.alpha * prior
        prediction = base.unsqueeze(1) + correction
        return NEONCorrectionOutput(
            prediction=prediction,
            correction=correction,
            trainable_correction=trainable,
            prior_correction=prior,
        )

    def regularized_parameters(self):
        """Parameters that define the trainable correction modulation/output."""

        yield from self.trainable_branch.epistemic_encoder.parameters()
        yield from self.trainable_branch.output_head.parameters()
        if self.trainable_branch.lead_time_encoder is not None:
            yield from self.trainable_branch.lead_time_encoder.parameters()


def _validate_nested_prediction(prediction: torch.Tensor) -> tuple[int, int, int, int, int, int]:
    if prediction.ndim != 6:
        raise ValueError(
            "nested predictions must have shape [B, M, K, T, Nv, C], "
            f"got {tuple(prediction.shape)}."
        )
    return tuple(int(v) for v in prediction.shape)  # type: ignore[return-value]


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(device=values.device, dtype=values.dtype)
    # Left-pad (prepend) singleton dims so a [T, Nv, C] weight broadcasts against
    # [B, T, Nv, C] or [B, M, K, T, Nv, C] values per standard right-aligned
    # broadcasting -- matching the [T, Nv, C] contract used by the metric APIs
    # and _normalize_score_weights.
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(0)
    try:
        torch.broadcast_shapes(values.shape, weights.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"weights shape {tuple(weights.shape)} is not broadcastable to {tuple(values.shape)}."
        ) from exc
    weighted = values * weights
    denom = torch.broadcast_to(weights, values.shape).sum().clamp_min(1.0e-12)
    return weighted.sum() / denom


def _normalize_score_weights(
    weights: torch.Tensor,
    *,
    batch_size: int,
    time_steps: int,
    num_nodes: int,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = weights.to(device=device, dtype=dtype)
    if weights.shape == (time_steps, num_nodes, channels):
        weights = weights.unsqueeze(0).expand(batch_size, -1, -1, -1)
    elif weights.shape == (batch_size, time_steps, num_nodes):
        weights = weights.unsqueeze(-1).expand(-1, -1, -1, channels)
    elif weights.shape == (batch_size, time_steps, num_nodes, channels):
        pass
    else:
        raise ValueError(
            "weights must have shape [T, Nv, C], [B, T, Nv], or [B, T, Nv, C]; "
            f"got {tuple(weights.shape)}."
        )
    denom = weights.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0e-12)
    return weights / denom


def per_epistemic_fair_crps(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    include_reference_term: bool = False,
    reduction: str = "mean",
    chunk_size: int | None = 65536,
) -> torch.Tensor:
    """Compute ensemble-vs-reference fair CRPS separately for each ``z_e``.

    Parameters
    ----------
    prediction
        Tensor with shape ``[B, M, K, T, Nv, C]``.
    reference
        Tensor with shape ``[B, R, T, Nv, C]``.
    weights
        Optional mesh/time/channel weights.  When supplied, weights are
        normalized per batch item across ``[T, Nv, C]``.
    include_reference_term
        If ``True``, subtract the HEC-RAS reference self-distance term for a
        full distribution-to-distribution diagnostic.  This term is constant
        with respect to Stage-2 parameters for a fixed batch and is usually not
        needed for gradients.
    reduction
        ``"none"`` returns ``[B, M]``.  ``"mean"`` and ``"sum"`` reduce it.
    chunk_size
        Number of flattened ``[T, Nv, C]`` locations processed at once.  The
        default avoids materializing coastal-scale ``[B, M, K, R, T, Nv, C]``
        tensors.  Pass ``None`` only for tiny debugging tensors.
    """

    B, M, K, T, Nv, C = _validate_nested_prediction(prediction)
    if reference.shape[:1] != (B,) or reference.shape[2:] != (T, Nv, C):
        raise ValueError(
            "reference must have shape [B, R, T, Nv, C] matching prediction, "
            f"got prediction={tuple(prediction.shape)} reference={tuple(reference.shape)}."
        )
    R = int(reference.shape[1])
    if K < 2:
        raise ValueError("per_epistemic_fair_crps requires K >= 2 aleatory samples.")
    if R < 1:
        raise ValueError("reference ensemble must have at least one member.")

    flat_pred = prediction.reshape(B, M, K, T * Nv * C)
    flat_ref = reference.reshape(B, R, T * Nv * C)
    n_locations = int(flat_pred.shape[-1])
    if chunk_size is None:
        chunk_size = n_locations
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1 or None.")

    flat_weights = None
    if weights is not None:
        norm_weights = _normalize_score_weights(
            weights,
            batch_size=B,
            time_steps=T,
            num_nodes=Nv,
            channels=C,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        flat_weights = norm_weights.reshape(B, n_locations)

    per_particle = prediction.new_zeros((B, M))
    unweighted_count = 0

    for start in range(0, n_locations, chunk_size):
        stop = min(start + chunk_size, n_locations)
        pred_chunk = flat_pred[..., start:stop]
        ref_chunk = flat_ref[..., start:stop]
        q = int(stop - start)

        # E[|X-Y|] without materializing [B, M, K, R, q]. Looping over K keeps
        # peak memory bounded by [B, M, R, q], which is viable for coastal grids.
        term1 = prediction.new_zeros((B, M, q))
        for k_idx in range(K):
            diff = (pred_chunk[:, :, k_idx, :].unsqueeze(2) - ref_chunk.unsqueeze(1)).abs()
            term1 = term1 + diff.mean(dim=2)
        term1 = term1 / float(K)

        # Fair forecast self-distance: 1/(2K(K-1)) sum_{k != k'} |x_k-x_k'|.
        term2 = prediction.new_zeros((B, M, q))
        for k_idx in range(K):
            diff = (pred_chunk[:, :, k_idx, :].unsqueeze(2) - pred_chunk).abs()
            term2 = term2 + diff.sum(dim=2)
        term2 = term2 / float(2 * K * (K - 1))
        score = term1 - term2

        if include_reference_term:
            if R < 2:
                raise ValueError("include_reference_term=True requires R >= 2 reference members.")
            ref_term = prediction.new_zeros((B, q))
            for r_idx in range(R):
                diff = (ref_chunk[:, r_idx, :].unsqueeze(1) - ref_chunk).abs()
                ref_term = ref_term + diff.sum(dim=1)
            ref_term = ref_term / float(2 * R * (R - 1))
            score = score - ref_term.unsqueeze(1)

        if flat_weights is None:
            per_particle = per_particle + score.sum(dim=-1)
            unweighted_count += q
        else:
            w = flat_weights[:, start:stop]
            per_particle = per_particle + (score * w.unsqueeze(1)).sum(dim=-1)

    if flat_weights is None:
        per_particle = per_particle / max(float(unweighted_count), 1.0)

    reduction = str(reduction).strip().lower()
    if reduction == "none":
        return per_particle
    if reduction == "mean":
        return per_particle.mean()
    if reduction == "sum":
        return per_particle.sum()
    raise ValueError("reduction must be one of {'none', 'mean', 'sum'}.")

def pooled_fair_crps(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
    chunk_size: int | None = 65536,
) -> torch.Tensor:
    """Negative-control fair CRPS that pools ``(z_a, z_e)`` into one ensemble.

    Flattens ``[B, M, K, T, Nv, C]`` to a single ``M*K``-member ensemble and
    scores it once against the reference. This intentionally does NOT preserve
    the nested per-epistemic interpretation: pooling inflates the fair-CRPS
    self-distance term and can mask epistemic miscalibration. Provided only as
    the plan's negative control against :func:`per_epistemic_fair_crps`.
    """
    flat = flatten_nested_predictions(prediction).unsqueeze(1)  # [B, 1, M*K, T, Nv, C]
    return per_epistemic_fair_crps(
        flat, reference, weights=weights, reduction=reduction, chunk_size=chunk_size
    )


def stage2_fit_score(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    objective: str = "per_epistemic_fcrps",
    chunk_size: int | None = 65536,
) -> torch.Tensor:
    """Stage-2 fit objective selected by the NEON config.

    The main objective scores each fixed epistemic particle separately, so
    ``z_e`` indexes operators and ``z_a`` indexes aleatory samples within each
    operator. The pooled and L2 objectives are explicit ablations/controls and
    should not be used as the primary NEON-aligned training objective.
    """

    objective = str(objective).strip().lower()
    if objective in {"per_epistemic_fcrps", "per_epistemic_crps", "fcrps"}:
        return per_epistemic_fair_crps(
            prediction,
            reference,
            weights=weights,
            reduction="mean",
            chunk_size=chunk_size,
        )
    if objective == "pooled_fcrps":
        return pooled_fair_crps(
            prediction,
            reference,
            weights=weights,
            reduction="mean",
            chunk_size=chunk_size,
        )
    if objective in {"l2_mean", "mean_l2"}:
        flat = flatten_nested_predictions(prediction)
        pred_mean = flat.mean(dim=1)
        ref_mean = reference.mean(dim=1)
        return _weighted_mean((pred_mean - ref_mean).pow(2), weights)
    raise ValueError(
        "objective must be one of {'per_epistemic_fcrps', 'pooled_fcrps', 'l2_mean'}, "
        f"got {objective!r}."
    )


def rpf_l2_penalty(module: NEONEpistemicCorrection) -> torch.Tensor:
    """L2 penalty on selected trainable randomized-prior-function weights."""

    total = None
    for param in module.regularized_parameters():
        value = param.pow(2).mean()
        total = value if total is None else total + value
    if total is None:
        device = next(module.parameters()).device
        return torch.tensor(0.0, device=device)
    return total


def correction_magnitude_penalty(
    correction: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return _weighted_mean(correction.pow(2), weights)


def positivity_penalty(
    prediction: torch.Tensor,
    *,
    zero_threshold: float | torch.Tensor = 0.0,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    threshold = torch.as_tensor(zero_threshold, dtype=prediction.dtype, device=prediction.device)
    penalty = torch.relu(threshold - prediction).pow(2)
    return _weighted_mean(penalty, weights)


def graph_smoothness_penalty(
    correction: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_nested_prediction(correction)
    if edge_index.ndim != 2 or edge_index.shape[-1] != 2:
        raise ValueError(f"edge_index must have shape [E, 2], got {tuple(edge_index.shape)}.")
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=correction.device, dtype=correction.dtype)
    edge_index = edge_index.to(device=correction.device, dtype=torch.long)
    src = edge_index[:, 0]
    dst = edge_index[:, 1]
    diff = correction[..., src, :] - correction[..., dst, :]
    sq = diff.pow(2)
    if edge_weights is not None:
        ew = edge_weights.to(device=correction.device, dtype=correction.dtype)
        ew = ew.view(1, 1, 1, 1, -1, 1)
        denom = torch.broadcast_to(ew, sq.shape).sum().clamp_min(1.0e-12)
        return (sq * ew).sum() / denom
    return sq.mean()


def temporal_smoothness_penalty(
    correction: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    _validate_nested_prediction(correction)
    if correction.shape[3] < 2:
        return torch.tensor(0.0, device=correction.device, dtype=correction.dtype)
    diff = correction[:, :, :, 1:] - correction[:, :, :, :-1]
    return _weighted_mean(diff.pow(2), weights)


def nested_variance_components(prediction: torch.Tensor) -> NestedVarianceComponents:
    """Estimate nested aleatory and epistemic variance components.

    The reported total is the sum of the nested components. This is the
    variance-attribution quantity used for uncertainty decomposition rather
    than the flat unbiased variance over all ``M*K`` samples.
    """

    _, M, K, _, _, _ = _validate_nested_prediction(prediction)
    mean_k = prediction.mean(dim=2)
    if K > 1:
        aleatory = prediction.var(dim=2, unbiased=True).mean(dim=1)
    else:
        aleatory = torch.zeros_like(mean_k[:, 0])
    if M > 1:
        epistemic = mean_k.var(dim=1, unbiased=True)
    else:
        epistemic = torch.zeros_like(mean_k[:, 0])
    total = aleatory + epistemic
    return NestedVarianceComponents(aleatory=aleatory, epistemic=epistemic, total=total)


def anova_corrected_epistemic_variance(prediction: torch.Tensor) -> torch.Tensor:
    """ANOVA-style correction for finite-``K`` contamination of epistemic variance."""

    _, M, K, _, _, _ = _validate_nested_prediction(prediction)
    if M < 2 or K < 2:
        return torch.zeros_like(prediction[:, 0, 0])
    mean_k = prediction.mean(dim=2)
    between = mean_k.var(dim=1, unbiased=True)
    within_ms = prediction.var(dim=2, unbiased=True).mean(dim=1)
    return torch.clamp(between - within_ms / float(K), min=0.0)


def base_rmse_from_reference(
    base_prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> float:
    """RMSE of the base ensemble mean vs the reference ensemble mean.

    ``base_prediction`` is ``[B, K, T, Nv, C]`` and ``reference`` is
    ``[B, R, T, Nv, C]``. Used as the ``RMSE_base`` target for prior-scale
    auto-calibration.
    """
    if base_prediction.ndim != 5 or reference.ndim != 5:
        raise ValueError(
            "base_prediction and reference must both be 5-D "
            f"([B,K,T,Nv,C] / [B,R,T,Nv,C]); got {tuple(base_prediction.shape)} "
            f"and {tuple(reference.shape)}."
        )
    pred_mean = base_prediction.mean(dim=1)
    ref_mean = reference.mean(dim=1)
    mse = _weighted_mean((pred_mean - ref_mean).pow(2), weights)
    return float(mse.clamp_min(0.0).sqrt().item())


def calibrate_prior_scale(
    *,
    module: NEONEpistemicCorrection,
    features: torch.Tensor,
    z_e: torch.Tensor,
    base_rmse: float,
    target_fraction: float = 0.10,
    eps: float = 1.0e-8,
) -> float:
    """Return ``alpha`` s.t. ``Std_ze[alpha * E^P(phi, z_e)] ~= f * RMSE_base``.

    Measures the epistemic (z_e) standard deviation of the *unit-scale* prior
    correction ``E^P`` on a calibration batch and picks ``alpha`` so the scaled
    prior spread matches ``target_fraction * base_rmse``. Because std is linear
    in scale, this is exact for the same aggregation used here. Robust to a
    degenerate (near-zero) prior spread via ``eps``.
    """
    with torch.no_grad():
        prior = module.prior_branch(features, z_e)  # [B, M, K, T, Nv, C]
        if int(prior.shape[1]) > 1:
            spread = float(prior.std(dim=1, unbiased=True).mean().item())
        else:
            spread = 0.0
    return float(target_fraction) * float(base_rmse) / max(spread, eps)


def compute_stage2_loss(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    correction: torch.Tensor,
    module: NEONEpistemicCorrection,
    weights: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    zero_threshold: float | torch.Tensor = 0.0,
    loss_weights: NEONStage2LossWeights | None = None,
    objective: str = "per_epistemic_fcrps",
) -> NEONStage2LossOutput:
    """Compute the NEON Stage-2 objective from nested corrected predictions."""

    loss_weights = loss_weights or NEONStage2LossWeights()
    fit = stage2_fit_score(prediction, reference, weights=weights, objective=objective)
    rpf = rpf_l2_penalty(module)
    if edge_index is None:
        graph = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    else:
        graph = graph_smoothness_penalty(correction, edge_index, edge_weights=edge_weights)
    time = temporal_smoothness_penalty(correction)
    pos = positivity_penalty(prediction, zero_threshold=zero_threshold, weights=weights)
    mag = correction_magnitude_penalty(correction, weights=weights)
    smooth = graph + float(loss_weights.time) * time
    total = (
        fit
        + float(loss_weights.rpf) * rpf
        + float(loss_weights.smooth) * smooth
        + float(loss_weights.pos) * pos
        + float(loss_weights.mag) * mag
    )
    return NEONStage2LossOutput(
        total=total,
        fit=fit,
        rpf=rpf,
        graph=graph,
        time=time,
        pos=pos,
        mag=mag,
    )


def nested_member_metadata(num_epistemic: int, num_aleatory: int) -> dict[str, torch.Tensor]:
    """Return flattened member metadata for artifact writers."""

    if num_epistemic < 1 or num_aleatory < 1:
        raise ValueError("num_epistemic and num_aleatory must both be >= 1.")
    epi_ids = torch.arange(num_epistemic).repeat_interleave(num_aleatory)
    ale_ids = torch.arange(num_aleatory).repeat(num_epistemic)
    member_ids = torch.arange(num_epistemic * num_aleatory)
    return {
        "member_id": member_ids,
        "member_epistemic_id": epi_ids,
        "member_aleatory_id": ale_ids,
    }


def flatten_nested_predictions(prediction: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, M, K, T, Nv, C]`` to ``[B, M*K, T, Nv, C]``."""

    B, M, K, T, Nv, C = _validate_nested_prediction(prediction)
    return prediction.reshape(B, M * K, T, Nv, C)


def save_neon_stage2_checkpoint(
    path: str | Path,
    module: NEONEpistemicCorrection,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save trainable and fixed-prior Stage-2 parameters plus metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": module.state_dict(),
        "metadata": metadata or {},
        "architecture": {
            "feature_channels": module.feature_channels,
            "out_channels": module.out_channels,
            "epistemic_dim": module.epistemic_dim,
            "hidden_channels": module.hidden_channels,
            "n_hidden_layers": module.n_hidden_layers,
            "alpha": module.alpha,
            "detach_features": module.detach_features,
            "lead_time_dim": module.lead_time_dim,
            "za_dependent": module.za_dependent,
        },
    }
    torch.save(payload, path)


def load_neon_stage2_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[NEONEpistemicCorrection, dict[str, Any]]:
    """Load a Stage-2 EpiNet checkpoint and return ``(module, metadata)``."""

    payload = torch.load(Path(path), map_location=map_location)
    arch = dict(payload["architecture"])
    module = NEONEpistemicCorrection(**arch)
    module.load_state_dict(payload["state_dict"])
    for param in module.prior_branch.parameters():
        param.requires_grad_(False)
    return module, dict(payload.get("metadata") or {})


def fair_crps_members(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    reduction: str = "mean",
    chunk_size: int | None = 65536,
) -> torch.Tensor:
    """Exact fair CRPS of one N-member ensemble vs an R-member reference.

    Mathematically identical to per_epistemic_fair_crps(prediction.unsqueeze(1),
    ...) with the default no-reference-term setting, but O((N + R) log N) per
    location instead of O(N^2 + N R), using order statistics:

      sum_k |x_k - y|        = (2 c - N) y - 2 P_c + S,   c = #{x_k < y},
      sum_{i<j} (x_(j)-x_(i)) = sum_i (2 i - 1 - N) x_(i),

    where P is the prefix sum of the sorted members and S their total. This is
    required at evaluation budgets: the flattened marginal ensemble at
    M_eval=32 x K_eval=50 has N=1600 members, for which the pairwise
    self-distance term would be ~1e12 elementwise ops on the coastal mesh.
    """

    if prediction.ndim != 5:
        raise ValueError(
            f"prediction must be [B, N, T, Nv, C]; got {tuple(prediction.shape)}."
        )
    B, N, T, Nv, C = (int(v) for v in prediction.shape)
    if reference.ndim != 5 or int(reference.shape[0]) != B or reference.shape[2:] != prediction.shape[2:]:
        raise ValueError(
            "reference must be [B, R, T, Nv, C] matching prediction; got "
            f"prediction={tuple(prediction.shape)} reference={tuple(reference.shape)}."
        )
    R = int(reference.shape[1])
    if N < 2:
        raise ValueError("fair CRPS requires N >= 2 ensemble members.")

    n_loc = T * Nv * C
    flat_pred = prediction.reshape(B, N, n_loc)
    flat_ref = reference.reshape(B, R, n_loc)
    if chunk_size is None:
        chunk_size = n_loc
    chunk_size = max(1, int(chunk_size))

    flat_weights = None
    if weights is not None:
        flat_weights = _normalize_score_weights(
            weights,
            batch_size=B,
            time_steps=T,
            num_nodes=Nv,
            channels=C,
            device=prediction.device,
            dtype=prediction.dtype,
        ).reshape(B, n_loc)

    dtype = prediction.dtype
    coef = torch.arange(1, N + 1, device=prediction.device, dtype=dtype) * 2.0 - float(N + 1)
    out = prediction.new_zeros((B,))
    for start in range(0, n_loc, chunk_size):
        stop = min(start + chunk_size, n_loc)
        x = flat_pred[..., start:stop].transpose(1, 2).contiguous()  # [B, q, N]
        y = flat_ref[..., start:stop].transpose(1, 2).contiguous()   # [B, q, R]
        xs, _ = torch.sort(x, dim=-1)
        prefix = torch.zeros(
            xs.shape[0], xs.shape[1], N + 1, device=xs.device, dtype=dtype
        )
        prefix[..., 1:] = xs.cumsum(dim=-1)
        total = prefix[..., -1]                                       # [B, q]
        c = torch.searchsorted(xs, y)                                 # [B, q, R]
        pc = prefix.gather(-1, c)
        cross = ((2.0 * c.to(dtype) - float(N)) * y - 2.0 * pc + total.unsqueeze(-1)).sum(dim=-1)
        cross = cross / float(N * R)                                  # E|X - Y| term
        self_term = (xs * coef).sum(dim=-1) / float(N * (N - 1))      # fair self term
        crps = cross - self_term                                      # [B, q]
        if flat_weights is not None:
            out = out + (crps * flat_weights[:, start:stop]).sum(dim=-1)
        else:
            out = out + crps.sum(dim=-1)
    if flat_weights is None:
        out = out / float(n_loc)

    if reduction == "none":
        return out
    if reduction == "sum":
        return out.sum()
    if reduction == "mean":
        return out.mean()
    raise ValueError(f"reduction must be one of none|sum|mean, got {reduction!r}.")
