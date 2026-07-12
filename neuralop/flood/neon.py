"""NEON-aligned Stage-2 epistemic utilities for frozen FGNO models.

This module intentionally keeps Stage-2 logic behind a small offline research
interface.  It does not change Stage-1 FGNO training behavior.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    """Weights for opt-in Stage-2 ablation penalties.

    Native EpiNet/NEON training uses the data-fit objective with the
    randomized prior architecture. Flood-specific smoothness, positivity, and
    magnitude penalties are retained only as explicit ablation knobs.
    """

    rpf: float = 0.0
    smooth: float = 0.0
    time: float = 0.0
    pos: float = 0.0
    mag: float = 0.0


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
    diagnostics: dict[str, float] | None = None


def _activation(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError("branch_activation must be one of {'gelu', 'silu', 'relu'}.")


def _pointwise_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_hidden_layers: int,
    activation: str,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = int(input_dim)
    for _ in range(max(0, int(n_hidden_layers))):
        layers.extend([nn.Linear(current, int(hidden_dim)), _activation(activation)])
        current = int(hidden_dim)
    layers.append(nn.Linear(current, int(output_dim)))
    return nn.Sequential(*layers)


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


def _append_projected_lead_features(features: torch.Tensor, lead_time_dim: int) -> torch.Tensor:
    """Concatenate deterministic lead-time features to ``[B,K,T,Nv,C]`` tensors."""

    lead_time_dim = int(lead_time_dim)
    if lead_time_dim <= 0:
        return features
    n_time = int(features.shape[2])
    if n_time > 1:
        t_norm = torch.linspace(0.0, 1.0, n_time, device=features.device, dtype=features.dtype)
    else:
        t_norm = torch.zeros(1, device=features.device, dtype=features.dtype)
    lead = _LeadTimeEncoder._sinusoidal(t_norm, lead_time_dim)
    lead = lead.view(1, 1, n_time, 1, lead_time_dim).expand(
        features.shape[0], features.shape[1], -1, features.shape[3], -1
    )
    return torch.cat([features, lead], dim=-1)


class _ProjectedTrainableBranch(nn.Module):
    """NEON/ENN-style projected trainable EpiNet branch.

    The MLP emits ``out_channels * d_e`` coefficients at each mesh-time point;
    the correction is their dot product with the epistemic index. This mirrors
    DeepMind ENN ``ProjectedMLP`` and PIL NEON ``EpiTrain`` more closely than a
    FiLM/AdaIN modulator.
    """

    def __init__(
        self,
        *,
        feature_channels: int,
        out_channels: int,
        epistemic_dim: int,
        hidden_channels: int,
        n_hidden_layers: int = 2,
        activation: str = "gelu",
        concat_index: bool = True,
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
        self.feature_channels = int(feature_channels)
        self.out_channels = int(out_channels)
        self.epistemic_dim = int(epistemic_dim)
        self.concat_index = bool(concat_index)
        self.lead_time_dim = int(lead_time_dim)
        input_dim = self.feature_channels + self.lead_time_dim
        if self.concat_index:
            input_dim += self.epistemic_dim
        self.mlp = _pointwise_mlp(
            input_dim=input_dim,
            hidden_dim=int(hidden_channels),
            output_dim=self.out_channels * self.epistemic_dim,
            n_hidden_layers=int(n_hidden_layers),
            activation=activation,
        )

    def forward(self, features: torch.Tensor, z_e: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "features must have shape [B, K, T, Nv, C_phi], "
                f"got {tuple(features.shape)}."
            )
        if z_e.ndim != 2:
            raise ValueError(f"z_e must have shape [M, d_e], got {tuple(z_e.shape)}.")
        if z_e.shape[-1] != self.epistemic_dim:
            raise ValueError(
                f"z_e last dimension must be {self.epistemic_dim}, got {z_e.shape[-1]}."
            )
        B, K, T, Nv, _ = (int(v) for v in features.shape)
        M = int(z_e.shape[0])
        branch_features = _append_projected_lead_features(features, self.lead_time_dim)
        expanded = branch_features.unsqueeze(1).expand(B, M, K, T, Nv, -1)
        if self.concat_index:
            z_features = z_e.to(device=features.device, dtype=features.dtype).view(
                1, M, 1, 1, 1, self.epistemic_dim
            )
            z_features = z_features.expand(B, M, K, T, Nv, -1)
            expanded = torch.cat([expanded, z_features], dim=-1)
        flat = expanded.reshape(-1, expanded.shape[-1])
        coeff = self.mlp(flat).reshape(B, M, K, T, Nv, self.out_channels, self.epistemic_dim)
        z_dot = z_e.to(device=features.device, dtype=features.dtype).view(
            1, M, 1, 1, 1, 1, self.epistemic_dim
        )
        return (coeff * z_dot).sum(dim=-1)


class _ProjectedPriorBranch(nn.Module):
    """Fixed randomized-prior branch with one small MLP basis per z dimension."""

    def __init__(
        self,
        *,
        feature_channels: int,
        out_channels: int,
        epistemic_dim: int,
        hidden_channels: int,
        n_hidden_layers: int = 2,
        activation: str = "gelu",
        lead_time_dim: int = 0,
        extra_feature_channels: int = 0,
    ) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1.")
        self.feature_channels = int(feature_channels)
        self.out_channels = int(out_channels)
        self.epistemic_dim = int(epistemic_dim)
        self.lead_time_dim = int(lead_time_dim)
        self.extra_feature_channels = int(extra_feature_channels)
        if self.extra_feature_channels < 0:
            raise ValueError("extra_feature_channels must be >= 0.")
        input_dim = self.feature_channels + self.lead_time_dim + self.extra_feature_channels
        self.basis = nn.ModuleList(
            [
                _pointwise_mlp(
                    input_dim=input_dim,
                    hidden_dim=int(hidden_channels),
                    output_dim=self.out_channels,
                    n_hidden_layers=int(n_hidden_layers),
                    activation=activation,
                )
                for _ in range(self.epistemic_dim)
            ]
        )

    def forward(
        self,
        features: torch.Tensor,
        z_e: torch.Tensor,
        *,
        extra_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 5:
            raise ValueError(
                "features must have shape [B, K, T, Nv, C_phi], "
                f"got {tuple(features.shape)}."
            )
        if z_e.ndim != 2 or z_e.shape[-1] != self.epistemic_dim:
            raise ValueError(
                f"z_e must have shape [M, {self.epistemic_dim}], got {tuple(z_e.shape)}."
            )
        B, K, T, Nv, _ = (int(v) for v in features.shape)
        branch_features = _append_projected_lead_features(features, self.lead_time_dim)
        if self.extra_feature_channels:
            if extra_features is None:
                raise ValueError(
                    "extra_features are required because extra_feature_channels="
                    f"{self.extra_feature_channels}."
                )
            if extra_features.shape != (B, K, T, Nv, self.extra_feature_channels):
                raise ValueError(
                    "extra_features must have shape "
                    f"{(B, K, T, Nv, self.extra_feature_channels)}, got "
                    f"{tuple(extra_features.shape)}."
                )
            branch_features = torch.cat(
                [branch_features, extra_features.to(device=features.device, dtype=features.dtype)],
                dim=-1,
            )
        elif extra_features is not None:
            raise ValueError(
                "extra_features were supplied but this prior branch was built without them."
            )
        flat = branch_features.reshape(-1, branch_features.shape[-1])
        basis = torch.stack(
            [mlp(flat).reshape(B, K, T, Nv, self.out_channels) for mlp in self.basis],
            dim=-1,
        )
        z_dot = z_e.to(device=features.device, dtype=features.dtype).view(
            1, int(z_e.shape[0]), 1, 1, 1, 1, self.epistemic_dim
        )
        return (basis.unsqueeze(1) * z_dot).sum(dim=-1)


class NEONEpistemicCorrection(nn.Module):
    """Randomized-prior EpiNet correction head for frozen FGNO features."""

    def __init__(
        self,
        *,
        feature_channels: int,
        out_channels: int,
        epistemic_dim: int,
        hidden_channels: int | None = None,
        train_hidden_channels: int | None = None,
        prior_hidden_channels: int | None = None,
        alpha: float = 0.1,
        n_hidden_layers: int = 2,
        branch_layers: int | None = None,
        branch_activation: str = "gelu",
        branch_type: str = "projected",
        concat_index: bool = True,
        detach_features: bool = True,
        lead_time_dim: int = 0,
        za_dependent: bool = True,
        prior_rff_dim: int = 0,
        prior_rff_lengthscale: float = 0.25,
        prior_rff_include_lead: bool = True,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.out_channels = int(out_channels)
        self.epistemic_dim = int(epistemic_dim)
        # ``hidden_channels`` is the legacy constructor/checkpoint key. When it
        # is supplied, it defines the train branch only; projected priors stay
        # small unless explicitly overridden.
        if train_hidden_channels is None:
            train_hidden_channels = 16 if hidden_channels is None else int(hidden_channels)
        if prior_hidden_channels is None:
            prior_hidden_channels = 16 if hidden_channels is None else int(hidden_channels)
        if branch_layers is not None:
            n_hidden_layers = int(branch_layers)
        self.train_hidden_channels = int(train_hidden_channels)
        self.prior_hidden_channels = int(prior_hidden_channels)
        self.hidden_channels = self.train_hidden_channels
        self.n_hidden_layers = int(n_hidden_layers)
        self.branch_activation = str(branch_activation).strip().lower()
        self.branch_type = str(branch_type).strip().lower()
        if self.branch_type not in {"projected", "film"}:
            raise ValueError("branch_type must be one of {'projected', 'film'}.")
        self.concat_index = bool(concat_index)
        self.alpha = float(alpha)
        self.detach_features = bool(detach_features)
        self.lead_time_dim = int(lead_time_dim)
        self.za_dependent = bool(za_dependent)
        self.prior_rff_dim = int(prior_rff_dim)
        self.prior_rff_lengthscale = float(prior_rff_lengthscale)
        self.prior_rff_include_lead = bool(prior_rff_include_lead)
        if self.prior_rff_dim < 0:
            raise ValueError("prior_rff_dim must be >= 0.")
        if self.prior_rff_dim % 2 != 0:
            raise ValueError("prior_rff_dim must be even.")
        if self.prior_rff_lengthscale <= 0.0:
            raise ValueError("prior_rff_lengthscale must be > 0.")
        if self.prior_rff_dim > 0 and self.branch_type != "projected":
            raise ValueError("prior_rff_dim > 0 is only supported for branch_type='projected'.")
        if self.prior_rff_dim > 0:
            rff_input_dim = 3 if self.prior_rff_include_lead else 2
            freqs = torch.randn(rff_input_dim, self.prior_rff_dim // 2) / self.prior_rff_lengthscale
            self.register_buffer("prior_rff_freqs", freqs)
        else:
            self.register_buffer("prior_rff_freqs", torch.empty(0))
        if self.branch_type == "film":
            self.trainable_branch = _CorrectionBranch(
                feature_channels=self.feature_channels,
                out_channels=self.out_channels,
                epistemic_dim=self.epistemic_dim,
                hidden_channels=self.train_hidden_channels,
                n_hidden_layers=n_hidden_layers,
                lead_time_dim=self.lead_time_dim,
            )
            self.prior_branch = _CorrectionBranch(
                feature_channels=self.feature_channels,
                out_channels=self.out_channels,
                epistemic_dim=self.epistemic_dim,
                hidden_channels=self.prior_hidden_channels,
                n_hidden_layers=n_hidden_layers,
                lead_time_dim=self.lead_time_dim,
            )
        else:
            self.trainable_branch = _ProjectedTrainableBranch(
                feature_channels=self.feature_channels,
                out_channels=self.out_channels,
                epistemic_dim=self.epistemic_dim,
                hidden_channels=self.train_hidden_channels,
                n_hidden_layers=n_hidden_layers,
                activation=self.branch_activation,
                concat_index=self.concat_index,
                lead_time_dim=self.lead_time_dim,
            )
            self.prior_branch = _ProjectedPriorBranch(
                feature_channels=self.feature_channels,
                out_channels=self.out_channels,
                epistemic_dim=self.epistemic_dim,
                hidden_channels=self.prior_hidden_channels,
                n_hidden_layers=n_hidden_layers,
                activation=self.branch_activation,
                lead_time_dim=self.lead_time_dim,
                extra_feature_channels=self.prior_rff_dim,
            )
        for param in self.prior_branch.parameters():
            param.requires_grad_(False)
        self.prior_rff_freqs.requires_grad_(False)

    def set_prior_scale(self, alpha: float) -> None:
        """Set the randomized-prior scale ``alpha`` (e.g. from auto-calibration)."""
        self.alpha = float(alpha)

    def _prepare_branch_features(self, features: torch.Tensor) -> torch.Tensor:
        branch_features = features.detach() if self.detach_features else features
        if not self.za_dependent:
            # za-independent ablation: average the frozen features over the
            # aleatory (K) axis so the correction depends only on the conditional
            # mean feature and is shared across aleatory members.
            branch_features = branch_features.mean(dim=1, keepdim=True).expand_as(branch_features)
        return branch_features

    def _prior_extra_features(
        self,
        features: torch.Tensor,
        node_coords: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self.prior_rff_dim == 0:
            return None
        if node_coords is None:
            raise ValueError("node_coords are required when prior_rff_dim > 0.")
        if features.ndim != 5:
            raise ValueError(f"features must be [B, K, T, Nv, C], got {tuple(features.shape)}.")
        B, K, T, Nv, _ = (int(v) for v in features.shape)
        coords = node_coords.to(device=features.device, dtype=features.dtype)
        if coords.ndim == 2:
            if coords.shape != (Nv, 2):
                raise ValueError(f"node_coords must be [Nv, 2] = {(Nv, 2)}, got {tuple(coords.shape)}.")
            coords = coords.unsqueeze(0).expand(B, -1, -1)
        elif coords.ndim == 3:
            if coords.shape[1:] != (Nv, 2):
                raise ValueError(
                    f"node_coords must have trailing shape [Nv, 2] = {(Nv, 2)}, "
                    f"got {tuple(coords.shape)}."
                )
            if coords.shape[0] == 1 and B != 1:
                coords = coords.expand(B, -1, -1)
            elif coords.shape[0] != B:
                raise ValueError(f"node_coords batch must be 1 or {B}, got {coords.shape[0]}.")
        else:
            raise ValueError(f"node_coords must be [Nv, 2] or [B, Nv, 2], got {tuple(coords.shape)}.")

        # The RFF lengthscale is in normalized-coordinate units; normalize per
        # family so projected meshes with large absolute coordinates remain
        # numerically comparable.
        c_min = coords.amin(dim=1, keepdim=True)
        c_span = (coords.amax(dim=1, keepdim=True) - c_min).clamp_min(1.0e-6)
        coords = (coords - c_min) / c_span
        coords_bt = coords[:, None, :, :].expand(B, T, Nv, 2)
        if self.prior_rff_include_lead:
            if T > 1:
                lead = torch.linspace(0.0, 1.0, T, device=features.device, dtype=features.dtype)
            else:
                lead = torch.zeros(1, device=features.device, dtype=features.dtype)
            lead = lead.view(1, T, 1, 1).expand(B, T, Nv, 1)
            rff_input = torch.cat([coords_bt, lead], dim=-1)
        else:
            rff_input = coords_bt
        freqs = self.prior_rff_freqs.to(device=features.device, dtype=features.dtype)
        phase = 2.0 * math.pi * torch.einsum("btnd,df->btnf", rff_input, freqs)
        scale = math.sqrt(2.0 / float(self.prior_rff_dim))
        psi = scale * torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return psi[:, None].expand(B, K, T, Nv, self.prior_rff_dim)

    def compute_prior(
        self,
        features: torch.Tensor,
        z_e: torch.Tensor,
        *,
        node_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the unit-scale fixed prior correction for frozen features."""

        branch_features = self._prepare_branch_features(features)
        extra = self._prior_extra_features(branch_features, node_coords)
        if self.branch_type == "projected":
            return self.prior_branch(branch_features, z_e, extra_features=extra)
        if extra is not None:
            raise ValueError("prior RFF extra features are only supported for projected branches.")
        return self.prior_branch(branch_features, z_e)

    def forward(
        self,
        base_prediction: torch.Tensor,
        features: torch.Tensor,
        z_e: torch.Tensor,
        *,
        node_coords: torch.Tensor | None = None,
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

        branch_features = self._prepare_branch_features(features)
        base = base_prediction.detach() if self.detach_features else base_prediction
        trainable = self.trainable_branch(branch_features, z_e)
        with torch.no_grad():
            extra = self._prior_extra_features(branch_features, node_coords)
            if self.branch_type == "projected":
                prior = self.prior_branch(branch_features, z_e, extra_features=extra)
            else:
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

        if self.branch_type == "film":
            yield from self.trainable_branch.epistemic_encoder.parameters()
            yield from self.trainable_branch.output_head.parameters()
            if self.trainable_branch.lead_time_encoder is not None:
                yield from self.trainable_branch.lead_time_encoder.parameters()
        else:
            yield from self.trainable_branch.mlp.parameters()


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
    sample_weights: torch.Tensor | None = None,
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
    sample_weights
        Optional family-by-epistemic weights with shape ``[B, M]``. These are
        applied after per-location CRPS aggregation; mesh/time weights keep
        their separate meaning. The mean reduction uses ``mean(score * weight)``
        so pre-normalized bootstrap weights keep their intended SGD weighting.
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

    if sample_weights is not None:
        sample_weights = sample_weights.to(device=prediction.device, dtype=prediction.dtype)
        if sample_weights.shape != (B, M):
            raise ValueError(
                f"sample_weights must have shape [B, M] = {(B, M)}, "
                f"got {tuple(sample_weights.shape)}."
            )
        per_particle = per_particle * sample_weights

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
    sample_weights: torch.Tensor | None = None,
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
            sample_weights=sample_weights,
            reduction="mean",
            chunk_size=chunk_size,
        )
    if objective == "pooled_fcrps":
        if sample_weights is not None:
            raise ValueError("sample_weights are only supported for per_epistemic_fcrps.")
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
    node_coords: torch.Tensor | None = None,
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
        prior = module.compute_prior(features, z_e, node_coords=node_coords)  # [B, M, K, T, Nv, C]
        if int(prior.shape[1]) > 1:
            spread = float(prior.std(dim=1, unbiased=True).mean().item())
        else:
            spread = 0.0
    return float(target_fraction) * float(base_rmse) / max(spread, eps)


def _stable_family_seed(seed: int, family_id: str) -> int:
    payload = f"{int(seed)}::{family_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def epistemic_bootstrap_weights(
    family_ids: Sequence[str],
    z_e: torch.Tensor,
    *,
    seed: int = 0,
    distribution: str = "tempered_exponential",
    temperature: float = 0.5,
    normalize: str = "per_epistemic_batch",
    min_weight: float = 0.05,
    max_weight: float = 5.0,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Return ENN-style index-dependent bootstrap weights ``[B, M]``.

    Uses the Gaussian-index exponential bootstrap from DeepMind ENN, with an
    optional tempering step that keeps the expected weight near one while
    reducing extreme per-family weights for the flood setting.
    """

    if z_e.ndim != 2:
        raise ValueError(f"z_e must have shape [M, d_e], got {tuple(z_e.shape)}.")
    distribution = str(distribution).strip().lower()
    normalize = str(normalize).strip().lower()
    family_ids = [str(fid) for fid in family_ids]
    if len(family_ids) < 1:
        raise ValueError("family_ids must be non-empty.")
    M, d_e = (int(v) for v in z_e.shape)
    if d_e < 1:
        raise ValueError("z_e must have nonzero epistemic dimension.")
    if distribution in {"none", "disabled"}:
        return torch.ones(len(family_ids), M, device=z_e.device, dtype=z_e.dtype)
    if distribution not in {"tempered_exponential", "exponential", "bernoulli"}:
        raise ValueError(
            "distribution must be one of {'tempered_exponential', 'exponential', "
            f"'bernoulli', 'none'}}, got {distribution!r}."
        )
    rows = []
    scale = 1.0 / math.sqrt(float(d_e))
    z_cpu = z_e.detach().to(device="cpu", dtype=torch.float32)
    for family_id in family_ids:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_family_seed(seed, family_id))
        b = torch.randn(d_e, generator=generator, dtype=torch.float32) * scale
        if distribution == "bernoulli":
            raw = 1.0 + torch.sign(z_cpu @ b)
        else:
            c = torch.randn(d_e, generator=generator, dtype=torch.float32) * scale
            raw = 0.5 * ((z_cpu @ b).pow(2) + (z_cpu @ c).pow(2))
            if distribution == "tempered_exponential":
                tau = float(temperature)
                raw = (1.0 - tau) + tau * raw
        rows.append(raw)
    weights = torch.stack(rows, dim=0).to(device=z_e.device, dtype=z_e.dtype)
    weights = weights.clamp(min=float(min_weight), max=float(max_weight))
    if normalize == "per_epistemic_batch":
        weights = weights / weights.mean(dim=0, keepdim=True).clamp_min(float(eps))
    elif normalize in {"none", "disabled"}:
        pass
    else:
        raise ValueError(
            "normalize must be one of {'per_epistemic_batch', 'none'}, "
            f"got {normalize!r}."
        )
    return weights


def cancellation_diagnostics(
    *,
    trainable_correction: torch.Tensor,
    prior_correction: torch.Tensor,
    alpha: float,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Compute train/prior cancellation diagnostics for logging."""

    with torch.no_grad():
        train = trainable_correction.detach().float()
        prior_scaled = float(alpha) * prior_correction.detach().float()
        total = train + prior_scaled
        train_norm = torch.linalg.vector_norm(train)
        prior_norm = torch.linalg.vector_norm(prior_scaled)
        total_norm = torch.linalg.vector_norm(total)
        train_rms = train.pow(2).mean().sqrt()
        prior_rms = prior_scaled.pow(2).mean().sqrt()
        total_rms = total.pow(2).mean().sqrt()
        cosine = (train * prior_scaled).sum() / (train_norm * prior_norm + eps)
        cancellation = 1.0 - total_norm / (train_norm + prior_norm + eps)
    return {
        "trainable_rms": float(train_rms.item()),
        "scaled_prior_rms": float(prior_rms.item()),
        "total_correction_rms": float(total_rms.item()),
        "train_prior_cosine": float(cosine.item()),
        "cancellation_fraction": float(cancellation.item()),
    }


def epistemic_variance_diagnostics(
    *,
    mbar_total: torch.Tensor,
    mbar_prior_scaled: torch.Tensor,
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """Epistemic (``z_e``) variance of the aleatory-averaged corrections.

    Computes ``Var_{z_e}[ E_{z_a}[ correction ] ]`` as the variance over the
    assembled epistemic (``M``) axis of corrections already averaged over the
    aleatory (``K``) axis. Both inputs are ``[B, M, T, Nv, C]`` and MUST carry
    the full set of ``M`` epistemic particles.

    This exists because the epistemic variance is inherently a cross-``M``
    quantity: computing it inside a per-``z_e`` chunk (``M == 1``) is undefined
    and silently returns zero. That is exactly the failure that produced
    all-zero epistemic-variance logs under ``epistemic_chunk_size=1``.
    """

    with torch.no_grad():
        total = mbar_total.detach().float()
        prior = mbar_prior_scaled.detach().float()
        if total.shape[1] > 1:
            total_var = total.var(dim=1, unbiased=True).mean()
            prior_var = prior.var(dim=1, unbiased=True).mean()
            retention = total_var / (prior_var + eps)
        else:
            zero = total.new_zeros(())
            total_var = zero
            prior_var = zero
            retention = zero
    return {
        "prior_retention_ratio": float(retention.item()),
        "total_epistemic_variance": float(total_var.item()),
        "prior_epistemic_variance": float(prior_var.item()),
    }


def prior_psi_floor_diagnostic(
    *,
    module: NEONEpistemicCorrection,
    features: torch.Tensor,
    z_e: torch.Tensor,
    node_coords: torch.Tensor | None = None,
) -> dict[str, float]:
    """Estimate the RFF-prior variance floor retained by asymmetric inputs.

    This is an observational diagnostic, not a training term. It measures
    ``Var_ze[alpha E^P(phi, z_e)]`` after averaging over aleatory samples; with
    ``prior_rff_dim=0`` it reports zero for legacy runs.
    """

    if int(getattr(module, "prior_rff_dim", 0)) <= 0:
        return {"prior_floor_var": 0.0}
    with torch.no_grad():
        prior = float(module.alpha) * module.compute_prior(
            features.detach(),
            z_e,
            node_coords=node_coords,
        ).detach().float()
        mbar_prior = prior.mean(dim=2)
        if int(mbar_prior.shape[1]) > 1:
            floor = mbar_prior.var(dim=1, unbiased=True).mean()
        else:
            floor = mbar_prior.new_zeros(())
    return {"prior_floor_var": float(floor.item())}


def compute_stage2_loss(
    *,
    prediction: torch.Tensor,
    reference: torch.Tensor,
    correction: torch.Tensor,
    module: NEONEpistemicCorrection,
    trainable_correction: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    zero_threshold: float | torch.Tensor = 0.0,
    loss_weights: NEONStage2LossWeights | None = None,
    objective: str = "per_epistemic_fcrps",
) -> NEONStage2LossOutput:
    """Compute the NEON Stage-2 objective from nested corrected predictions."""

    loss_weights = loss_weights or NEONStage2LossWeights()
    fit = stage2_fit_score(
        prediction,
        reference,
        weights=weights,
        sample_weights=sample_weights,
        objective=objective,
    )
    zero = torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)
    if float(loss_weights.rpf) != 0.0:
        rpf = rpf_l2_penalty(module)
    else:
        rpf = zero
    if float(loss_weights.smooth) != 0.0 and edge_index is not None:
        graph = graph_smoothness_penalty(correction, edge_index, edge_weights=edge_weights)
    else:
        graph = zero
    if float(loss_weights.smooth) != 0.0 and float(loss_weights.time) != 0.0:
        time = temporal_smoothness_penalty(correction)
    else:
        time = zero
    if float(loss_weights.pos) != 0.0:
        pos = positivity_penalty(prediction, zero_threshold=zero_threshold, weights=weights)
    else:
        pos = zero
    if float(loss_weights.mag) != 0.0:
        mag_source = correction if trainable_correction is None else trainable_correction
        mag = correction_magnitude_penalty(mag_source, weights=weights)
    else:
        mag = zero
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
        diagnostics=None,
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
            "train_hidden_channels": module.train_hidden_channels,
            "prior_hidden_channels": module.prior_hidden_channels,
            "n_hidden_layers": module.n_hidden_layers,
            "branch_layers": module.n_hidden_layers,
            "branch_activation": module.branch_activation,
            "branch_type": module.branch_type,
            "concat_index": module.concat_index,
            "alpha": module.alpha,
            "detach_features": module.detach_features,
            "lead_time_dim": module.lead_time_dim,
            "za_dependent": module.za_dependent,
            "prior_rff_dim": module.prior_rff_dim,
            "prior_rff_lengthscale": module.prior_rff_lengthscale,
            "prior_rff_include_lead": module.prior_rff_include_lead,
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
    if "branch_type" not in arch:
        # Backward compatibility for early Stage-2 checkpoints: those stored
        # only the old symmetric FiLM/AdaIN branch architecture.
        arch["branch_type"] = "film"
        if "train_hidden_channels" not in arch:
            arch["train_hidden_channels"] = arch.get("hidden_channels", 64)
        if "prior_hidden_channels" not in arch:
            arch["prior_hidden_channels"] = arch.get("hidden_channels", 64)
    arch.setdefault("prior_rff_dim", 0)
    arch.setdefault("prior_rff_lengthscale", 0.25)
    arch.setdefault("prior_rff_include_lead", True)
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
