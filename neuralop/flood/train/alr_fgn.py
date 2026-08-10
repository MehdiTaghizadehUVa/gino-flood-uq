"""Training utilities for anchored low-rank FGNO ensembles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from neuralop.flood.processing.wv_impl import (
    _build_x_from_dynamic_boundary,
    get_flood_crps_weights,
)
from neuralop.flood.losses import FloodMaskedCRPSLoss, masked_rmse
from neuralop.flood.train.fgn import FGNTrainer
from neuralop.flood.utils.runtime_core import parse_family_id_from_run_id
from neuralop.losses.probabilistic_losses import CRPSLoss
from neuralop.flood.data.reference_dispersion import ReferenceDispersionTable
from neuralop.flood.train.dispersion_pinning import (
    DEFAULT_WET_THRESHOLDS,
    dispersion_pinning_penalty,
)


@dataclass(frozen=True)
class NestedParticleBatch:
    """Flattened ``[particle, aleatory, batch]`` inputs and identifiers."""

    values: torch.Tensor
    latents: torch.Tensor
    particle_ids: torch.Tensor
    aleatory_ids: torch.Tensor


@dataclass(frozen=True)
class ParticleCRPSResult:
    """Mean Stage-1 objective and separately scored particle risks."""

    mean: torch.Tensor
    per_particle: torch.Tensor


@dataclass(frozen=True)
class ResidualDecompositionComponents:
    """Signed model and reference residuals around their respective means."""

    model_residuals: torch.Tensor
    reference_residual: torch.Tensor


@dataclass(frozen=True)
class ParticleMeanLossResult:
    """Bootstrap-weighted conditional-mean MSE for each particle."""

    mean: torch.Tensor
    per_particle: torch.Tensor


@dataclass(frozen=True)
class ResidualCenteringMonitor:
    """Fixed-bank discrepancy between stochastic and operational mean paths."""

    discrepancy_rms: torch.Tensor
    mc_se_rms: torch.Tensor
    exceeds_two_se_fraction: torch.Tensor


def stable_gradient_cosine(
    first_gradients,
    second_gradients,
    *,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Cosine similarity with float64 reductions to avoid float32 norm overflow."""
    if len(first_gradients) != len(second_gradients):
        raise ValueError("Gradient collections must have the same length.")
    dot = torch.zeros((), device=reference.device, dtype=torch.float64)
    first_norm_sq = torch.zeros_like(dot)
    second_norm_sq = torch.zeros_like(dot)
    for first, second in zip(first_gradients, second_gradients):
        if first is not None:
            first64 = first.detach().to(dtype=torch.float64)
            first_norm_sq = first_norm_sq + first64.square().sum()
        if second is not None:
            second64 = second.detach().to(dtype=torch.float64)
            second_norm_sq = second_norm_sq + second64.square().sum()
        if first is not None and second is not None:
            dot = dot + (first64 * second64).sum()
    denominator = (first_norm_sq * second_norm_sq).sqrt()
    if denominator.item() == 0.0:
        return reference.new_zeros(())
    return (dot / denominator).clamp(-1.0, 1.0).to(dtype=reference.dtype)


def residual_decomposition_components(
    stochastic_predictions: torch.Tensor,
    mean_predictions: torch.Tensor,
    target: torch.Tensor,
    reference_mean: torch.Tensor,
) -> ResidualDecompositionComponents:
    """Build the two residual laws without clamping either residual field.

    Shapes are ``[M,K,B,N,C]``, ``[M,B,N,C]``, and ``[B,N,C]`` for the
    final two arguments.  The same depth shift applied to predictions and
    references cancels exactly, which is the required location/dispersion
    separation for the pilot objective.
    """
    if stochastic_predictions.ndim != 5:
        raise ValueError("stochastic_predictions must have shape [M, K, B, N, C].")
    expected_mean_shape = (
        stochastic_predictions.shape[0],
        *stochastic_predictions.shape[2:],
    )
    if mean_predictions.shape != expected_mean_shape:
        raise ValueError(
            f"mean_predictions must have shape {expected_mean_shape}, "
            f"got {tuple(mean_predictions.shape)}."
        )
    expected_target_shape = tuple(stochastic_predictions.shape[2:])
    if target.shape != expected_target_shape or reference_mean.shape != expected_target_shape:
        raise ValueError(
            f"target and reference_mean must have shape {expected_target_shape}."
        )
    return ResidualDecompositionComponents(
        model_residuals=stochastic_predictions - mean_predictions.unsqueeze(1),
        reference_residual=target - reference_mean,
    )


def particle_bootstrap_mean_mse(
    predictions: torch.Tensor,
    reference_mean: torch.Tensor,
    *,
    family_weights: torch.Tensor,
    structural_dry_mask: torch.Tensor | None = None,
) -> ParticleMeanLossResult:
    """Compute family-bootstrap-weighted physical-space mean-field MSE."""
    if predictions.ndim != 4:
        raise ValueError("predictions must have shape [M, B, N, C].")
    particles, batch_size, n_cells, channels = predictions.shape
    if reference_mean.shape != (batch_size, n_cells, channels):
        raise ValueError("reference_mean must have shape [B, N, C].")
    if family_weights.shape != (particles, batch_size):
        raise ValueError(
            f"family_weights must have shape {(particles, batch_size)}, "
            f"got {tuple(family_weights.shape)}."
        )
    active = torch.ones(
        batch_size, n_cells, channels,
        device=predictions.device,
        dtype=predictions.dtype,
    )
    if structural_dry_mask is not None:
        dry = torch.as_tensor(
            structural_dry_mask, device=predictions.device, dtype=torch.bool
        )
        while dry.ndim > 2 and dry.shape[-1] == 1:
            dry = dry.squeeze(-1)
        if dry.ndim == 1:
            dry = dry.unsqueeze(0).expand(batch_size, n_cells)
        if dry.shape != (batch_size, n_cells):
            raise ValueError(
                f"structural_dry_mask must have shape {(batch_size, n_cells)}, "
                f"got {tuple(dry.shape)}."
            )
        active = active * (~dry).unsqueeze(-1).to(dtype=active.dtype)

    squared_error = (predictions - reference_mean.unsqueeze(0)).square()
    weights = family_weights.to(
        device=predictions.device, dtype=predictions.dtype
    ).view(particles, batch_size, 1, 1) * active.unsqueeze(0)
    denominator = weights.sum(dim=(1, 2, 3)).clamp_min(
        torch.finfo(predictions.dtype).eps
    )
    per_particle = (squared_error * weights).sum(dim=(1, 2, 3)) / denominator
    return ParticleMeanLossResult(
        mean=per_particle.mean(),
        per_particle=per_particle,
    )


def residual_centering_monitor(
    stochastic_predictions: torch.Tensor,
    mean_predictions: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor | None = None,
) -> ResidualCenteringMonitor:
    """Compare a fixed Monte Carlo bank mean with the zero-latent mean path."""
    if stochastic_predictions.ndim != 5 or stochastic_predictions.shape[1] < 2:
        raise ValueError(
            "stochastic_predictions must have shape [M, K>=2, B, N, C]."
        )
    expected_mean_shape = (
        stochastic_predictions.shape[0],
        *stochastic_predictions.shape[2:],
    )
    if mean_predictions.shape != expected_mean_shape:
        raise ValueError(
            f"mean_predictions must have shape {expected_mean_shape}."
        )
    particles, samples, batch_size, n_cells, channels = stochastic_predictions.shape
    discrepancy = (
        stochastic_predictions.mean(dim=1) - mean_predictions
    ).abs()
    mc_se = stochastic_predictions.std(dim=1, unbiased=True) / samples**0.5
    active = torch.ones_like(discrepancy, dtype=torch.bool)
    if structural_dry_mask is not None:
        dry = torch.as_tensor(
            structural_dry_mask, device=discrepancy.device, dtype=torch.bool
        )
        while dry.ndim > 2 and dry.shape[-1] == 1:
            dry = dry.squeeze(-1)
        if dry.ndim == 1:
            dry = dry.unsqueeze(0).expand(batch_size, n_cells)
        if dry.shape != (batch_size, n_cells):
            raise ValueError("structural_dry_mask shape does not match predictions.")
        active = (~dry).view(1, batch_size, n_cells, 1).expand(
            particles, batch_size, n_cells, channels
        )
    discrepancy_values = discrepancy[active]
    mc_se_values = mc_se[active]
    if discrepancy_values.numel() == 0:
        zero = discrepancy.new_zeros(())
        return ResidualCenteringMonitor(zero, zero, zero)
    return ResidualCenteringMonitor(
        discrepancy_rms=discrepancy_values.square().mean().sqrt(),
        mc_se_rms=mc_se_values.square().mean().sqrt(),
        exceeds_two_se_fraction=(
            discrepancy_values > 2.0 * mc_se_values
        ).to(dtype=discrepancy.dtype).mean(),
    )


class PhysicalRMSE:
    """Wettable-domain RMSE for postprocessed physical predictions."""

    reduction = "mean"
    expects_samples = False

    def __call__(self, pred, target, *, structural_dry_mask=None, **kwargs):
        del kwargs
        return masked_rmse(
            pred,
            target,
            structural_dry_mask=structural_dry_mask,
        )


@dataclass(frozen=True)
class ALRFamilySplit:
    train_indices: list[int]
    validation_indices: list[int]
    train_family_ids: list[str]
    validation_family_ids: list[str]


def split_alr_family_indices(
    dataset,
    *,
    validation_family_count: int,
    seed: int,
    train_family_limit: int | None = None,
    max_windows_per_family: int | None = None,
) -> ALRFamilySplit:
    """Create deterministic member-safe family splits from a flood dataset."""

    sample_index = getattr(dataset, "sample_index", None)
    if sample_index is None:
        raise TypeError("ALR family splitting requires dataset.sample_index.")
    family_by_index = [
        parse_family_id_from_run_id(str(run_id)) for run_id, _ in sample_index
    ]
    families = sorted(set(family_by_index))
    val_count = int(validation_family_count)
    if val_count <= 0 or val_count >= len(families):
        raise ValueError(
            "validation_family_count must leave at least one fit and one validation family."
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    order = torch.randperm(len(families), generator=generator).tolist()
    shuffled = [families[index] for index in order]
    validation_families = shuffled[:val_count]
    fit_pool = shuffled[val_count:]
    if train_family_limit is not None:
        limit = int(train_family_limit)
        if limit <= 0 or limit > len(fit_pool):
            raise ValueError(
                f"train_family_limit must be in [1, {len(fit_pool)}], got {limit}."
            )
        fit_pool = fit_pool[:limit]
    train_set = set(fit_pool)
    validation_set = set(validation_families)

    window_limit = None
    if max_windows_per_family is not None:
        window_limit = int(max_windows_per_family)
        if window_limit <= 0:
            raise ValueError("max_windows_per_family must be positive when set.")

    def selected_indices(selected_families):
        counts = {family: 0 for family in selected_families}
        selected = []
        for index, family in enumerate(family_by_index):
            if family not in selected_families:
                continue
            if window_limit is not None and counts[family] >= window_limit:
                continue
            selected.append(index)
            counts[family] += 1
        return selected

    return ALRFamilySplit(
        train_indices=selected_indices(train_set),
        validation_indices=selected_indices(validation_set),
        train_family_ids=list(fit_pool),
        validation_family_ids=list(validation_families),
    )


def make_nested_particle_batch(
    values: torch.Tensor,
    *,
    latent_bank: torch.Tensor,
    num_particles: int,
) -> NestedParticleBatch:
    """Expand one batch over particles while sharing aleatory draws across them.

    ``latent_bank`` has shape ``[K, B, d_a]``. The result is flattened in
    particle-major, aleatory-major, batch-major order for one vectorized model
    call.
    """

    if values.ndim < 1:
        raise ValueError("values must have a batch dimension.")
    if latent_bank.ndim != 3:
        raise ValueError("latent_bank must have shape [K, B, latent_dim].")
    particles = int(num_particles)
    if particles <= 0:
        raise ValueError("num_particles must be positive.")
    aleatory, batch_size, latent_dim = latent_bank.shape
    if values.shape[0] != batch_size:
        raise ValueError(
            f"values batch size {values.shape[0]} does not match latent bank {batch_size}."
        )

    expanded_values = values.unsqueeze(0).unsqueeze(0).expand(
        particles, aleatory, *values.shape
    )
    expanded_latents = latent_bank.unsqueeze(0).expand(
        particles, aleatory, batch_size, latent_dim
    )
    particle_ids = torch.arange(particles, device=values.device).view(
        particles, 1, 1
    ).expand(particles, aleatory, batch_size)
    aleatory_ids = torch.arange(aleatory, device=values.device).view(
        1, aleatory, 1
    ).expand(particles, aleatory, batch_size)
    flat_batch = particles * aleatory * batch_size
    return NestedParticleBatch(
        values=expanded_values.reshape(flat_batch, *values.shape[1:]),
        latents=expanded_latents.reshape(flat_batch, latent_dim),
        particle_ids=particle_ids.reshape(flat_batch),
        aleatory_ids=aleatory_ids.reshape(flat_batch),
    )


def particle_bootstrap_crps(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    family_weights: torch.Tensor,
    loss_fn,
    spatial_weights: torch.Tensor | None = None,
    structural_dry_mask: torch.Tensor | None = None,
) -> ParticleCRPSResult:
    """Score aleatory members separately within each epistemic particle.

    Parameters use nested predictions shaped ``[M, K, B, N, C]`` and persistent
    family weights shaped ``[M, B]``. The particle dimension is never pooled
    into the fair-CRPS ensemble dimension.
    """

    if predictions.ndim != 5:
        raise ValueError("predictions must have shape [M, K, B, N, C].")
    particles, _, batch_size, n_cells, n_channels = predictions.shape
    if target.shape != (batch_size, n_cells, n_channels):
        raise ValueError(
            "target shape must match the [B, N, C] axes of nested predictions."
        )
    if family_weights.shape != (particles, batch_size):
        raise ValueError(
            f"family_weights must have shape {(particles, batch_size)}, "
            f"got {tuple(family_weights.shape)}."
        )
    if spatial_weights is not None and spatial_weights.shape != target.shape:
        raise ValueError("spatial_weights must have the same shape as target.")

    risks = []
    for particle in range(particles):
        weights = family_weights[particle].to(
            device=target.device, dtype=target.dtype
        ).view(batch_size, 1, 1).expand_as(target)
        if spatial_weights is not None:
            weights = weights * spatial_weights.to(
                device=target.device, dtype=target.dtype
            )
        risks.append(
            loss_fn(
                predictions[particle],
                target,
                spatial_weights=weights,
                structural_dry_mask=structural_dry_mask,
            )
        )
    per_particle = torch.stack(risks)
    return ParticleCRPSResult(mean=per_particle.mean(), per_particle=per_particle)


def clamp_nested_feedback(
    predictions: torch.Tensor,
    *,
    structural_dry_mask: torch.Tensor | None,
    target_normalizer,
    water_depth_index: int = 0,
) -> torch.Tensor:
    """Apply physical nonnegativity and structural-dry constraints before AR feedback."""

    if predictions.ndim != 5:
        raise ValueError("predictions must have shape [M, K, B, N, C].")
    if target_normalizer is None:
        raise ValueError("target_normalizer is required for normalized ALR feedback.")
    particles, aleatory, batch_size, n_cells, channels = predictions.shape
    wd_index = int(water_depth_index)
    if wd_index < 0 or wd_index >= channels:
        raise ValueError(f"water_depth_index must be in [0, {channels - 1}].")

    flat = predictions.reshape(particles * aleatory * batch_size, n_cells, channels)
    physical = target_normalizer.inverse_transform(flat).clone()
    physical[..., wd_index].clamp_(min=0.0)
    if structural_dry_mask is not None:
        dry = torch.as_tensor(
            structural_dry_mask, dtype=torch.bool, device=physical.device
        )
        while dry.ndim > 2 and dry.shape[-1] == 1:
            dry = dry.squeeze(-1)
        if dry.ndim == 1:
            dry = dry.unsqueeze(0).expand(batch_size, n_cells)
        if dry.shape != (batch_size, n_cells):
            raise ValueError(
                f"structural_dry_mask must have shape {(batch_size, n_cells)}, "
                f"got {tuple(dry.shape)}."
            )
        dry = dry.unsqueeze(0).unsqueeze(0).expand(
            particles, aleatory, batch_size, n_cells
        ).reshape(particles * aleatory * batch_size, n_cells, 1)
        physical.masked_fill_(dry, 0.0)
    normalized = target_normalizer.transform(physical)
    return normalized.reshape_as(predictions)


def update_nested_history(
    histories: torch.Tensor,
    next_states: torch.Tensor,
) -> torch.Tensor:
    """Advance every ``(particle, aleatory)`` history with its own prediction."""

    if histories.ndim != 6:
        raise ValueError("histories must have shape [M, K, B, H, N, C].")
    if next_states.shape != (
        histories.shape[0],
        histories.shape[1],
        histories.shape[2],
        histories.shape[4],
        histories.shape[5],
    ):
        raise ValueError("next_states must match histories except for the history axis.")
    return torch.cat(
        [histories[..., 1:, :, :], next_states.unsqueeze(3)], dim=3
    )


class DirichletFamilyBootstrap(nn.Module):
    """Persistent particle-by-family Bayesian-bootstrap weights.

    Dirichlet(1, ..., 1) draws are generated from normalized exponential
    samples and scaled to have mean one over families. Family ordering and the
    seed are stored in ``extra_state``; the complete weight matrix is a buffer.
    """

    def __init__(
        self,
        *,
        family_ids: Sequence[str],
        num_particles: int,
        seed: int,
    ) -> None:
        super().__init__()
        ids = [str(value) for value in family_ids]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("family_ids must be a non-empty sequence of unique values.")
        if int(num_particles) <= 0:
            raise ValueError("num_particles must be positive.")

        self.family_ids = ids
        self.num_particles = int(num_particles)
        self.seed = int(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        uniforms = torch.rand(
            self.num_particles,
            len(ids),
            generator=generator,
            dtype=torch.float64,
        ).clamp_min(torch.finfo(torch.float64).tiny)
        exponential = -uniforms.log()
        weights = exponential / exponential.mean(dim=1, keepdim=True)
        self.register_buffer("weights", weights.to(dtype=torch.float32))
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._family_to_index = {
            family_id: index for index, family_id in enumerate(self.family_ids)
        }

    def weights_for(self, family_ids: Sequence[str]) -> torch.Tensor:
        try:
            indices = [self._family_to_index[str(value)] for value in family_ids]
        except KeyError as exc:
            raise KeyError(f"Unknown ALR-FGNO family ID: {exc.args[0]!r}.") from exc
        index = torch.tensor(indices, dtype=torch.long, device=self.weights.device)
        return self.weights.index_select(1, index)

    def get_extra_state(self):
        return {
            "family_ids": list(self.family_ids),
            "num_particles": int(self.num_particles),
            "seed": int(self.seed),
        }

    def set_extra_state(self, state) -> None:
        family_ids = [str(value) for value in state["family_ids"]]
        if len(family_ids) != self.weights.shape[1]:
            raise ValueError(
                "Checkpoint family ordering does not match the bootstrap weight matrix."
            )
        if int(state["num_particles"]) != self.weights.shape[0]:
            raise ValueError(
                "Checkpoint particle count does not match the bootstrap weight matrix."
            )
        self.family_ids = family_ids
        self.num_particles = int(state["num_particles"])
        self.seed = int(state["seed"])
        self._rebuild_index()


def _batch_family_ids(sample: dict) -> list[str]:
    values = sample.get("family_id")
    if values is not None:
        if isinstance(values, str):
            return [values]
        return [str(value) for value in values]
    run_ids = sample.get("run_id")
    if run_ids is None:
        raise KeyError("ALR-FGNO batches require family_id or run_id metadata.")
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    return [parse_family_id_from_run_id(str(run_id)) for run_id in run_ids]


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


class AnchoredLowRankFGNTrainer(FGNTrainer):
    """Stage-1 FGNO trainer with persistent low-rank epistemic particles."""

    def __init__(
        self,
        *,
        num_particles: int,
        family_bootstrap: DirichletFamilyBootstrap,
        anchor_penalty_weight: float = 1.0e-3,
        adapter_warmup_epochs: int = 5,
        eval_aleatory_samples: int | None = None,
        eval_member_chunk_size: int | None = None,
        rmse_noninferiority_margin: float = 0.001,
        target_normalizer=None,
        water_depth_index: int = 0,
        reference_dispersion: ReferenceDispersionTable | None = None,
        dispersion_penalty_weight: float = 0.0,
        dispersion_wet_thresholds: tuple[float, float] = DEFAULT_WET_THRESHOLDS,
        residual_decomposition_enabled: bool = False,
        mean_loss_weight: float = 1.0,
        residual_crps_weight: float = 1.0,
        residual_monitor_samples: int = 32,
        residual_monitor_seed: int = 20260809,
        residual_gradient_probe_batches: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_particles = int(num_particles)
        if self.num_particles <= 1:
            raise ValueError("ALR-FGNO requires at least two epistemic particles.")
        if family_bootstrap.num_particles != self.num_particles:
            raise ValueError("Bootstrap particle count must match the ALR model.")
        model = _unwrap_model(self.model)
        if not bool(getattr(model, "anchored_low_rank_enabled", False)):
            raise ValueError("AnchoredLowRankFGNTrainer requires an ALR-enabled model.")
        if int(getattr(model, "anchored_low_rank_num_particles", -1)) != self.num_particles:
            raise ValueError("Trainer and model particle counts do not match.")
        self.family_bootstrap = family_bootstrap
        self.anchor_penalty_weight = float(anchor_penalty_weight)
        self.adapter_warmup_epochs = max(0, int(adapter_warmup_epochs))
        self.eval_aleatory_samples = max(
            2,
            int(eval_aleatory_samples or self.crps_n_samples),
        )
        total_eval_members = self.num_particles * self.eval_aleatory_samples
        if eval_member_chunk_size is None:
            self.eval_member_chunk_size = total_eval_members
        else:
            if int(eval_member_chunk_size) <= 0:
                raise ValueError("eval_member_chunk_size must be positive.")
            self.eval_member_chunk_size = min(
                int(eval_member_chunk_size), total_eval_members
            )
        self.target_normalizer = target_normalizer
        self.water_depth_index = int(water_depth_index)
        self.reference_dispersion = reference_dispersion
        self.residual_decomposition_enabled = bool(residual_decomposition_enabled)
        self.mean_loss_weight = float(mean_loss_weight)
        self.residual_crps_weight = float(residual_crps_weight)
        self.residual_monitor_samples = int(residual_monitor_samples)
        self.residual_monitor_seed = int(residual_monitor_seed)
        if self.residual_monitor_samples < 2:
            raise ValueError("residual_monitor_samples must be at least two.")
        monitor_generator = torch.Generator(device="cpu")
        monitor_generator.manual_seed(self.residual_monitor_seed)
        self._residual_monitor_latents_cpu = torch.randn(
            self.residual_monitor_samples,
            int(self.fgn_noise_dim),
            generator=monitor_generator,
            dtype=torch.float32,
        )
        self._residual_monitor_active = False
        self.residual_gradient_probe_batches = max(
            0, int(residual_gradient_probe_batches)
        )
        self._residual_gradient_probes_done = 0
        self._residual_epoch_metric_sums = {}
        self._residual_epoch_metric_counts = {}
        self.dispersion_penalty_weight = float(dispersion_penalty_weight)
        self.dispersion_wet_thresholds = (
            float(dispersion_wet_thresholds[0]),
            float(dispersion_wet_thresholds[1]),
        )
        if self.dispersion_penalty_weight < 0.0:
            raise ValueError("dispersion_penalty_weight must be nonnegative.")
        if self.mean_loss_weight < 0.0 or self.residual_crps_weight < 0.0:
            raise ValueError("Residual-decomposition loss weights must be nonnegative.")
        if self.residual_decomposition_enabled:
            if self.reference_dispersion is None:
                raise ValueError(
                    "Residual decomposition requires a reference-dispersion table."
                )
            if self.reference_dispersion.reference_mean is None:
                raise ValueError(
                    "Residual decomposition requires reference_mean in the artifact."
                )
            if self.reference_dispersion.reference_mean_variance is None:
                raise ValueError(
                    "Residual decomposition requires reference_mean_variance in the artifact."
                )
            if self.target_normalizer is None:
                raise ValueError(
                    "Residual decomposition is scored in physical space and requires "
                    "the target normalizer."
                )
            if self.dispersion_penalty_weight > 0.0:
                raise ValueError(
                    "Residual decomposition and dispersion pinning cannot be enabled together."
                )
        if self.dispersion_penalty_weight > 0.0:
            if self.reference_dispersion is None:
                raise ValueError(
                    "dispersion_penalty_weight > 0 requires a reference_dispersion table."
                )
            if self.target_normalizer is None:
                raise ValueError(
                    "dispersion pinning compares physical depths and requires a "
                    "target_normalizer to invert the training normalization."
                )
        self.rmse_noninferiority_margin = float(rmse_noninferiority_margin)
        if self.rmse_noninferiority_margin < 0:
            raise ValueError("RMSE non-inferiority margin must be nonnegative.")
        self.base_validation_rmse = float("nan")

    def _seed_for_eval_batch(self, *, epoch, log_prefix, batch_idx):
        # Checkpoint selection must compare every epoch against the frozen base
        # under the same aleatory draws.
        return super()._seed_for_eval_batch(
            epoch=0,
            log_prefix=log_prefix,
            batch_idx=batch_idx,
        )

    def configure_selection_contract(self, *, base_rmse, margin=None):
        value = float(base_rmse)
        if not torch.isfinite(torch.tensor(value)):
            raise ValueError("Frozen-base validation RMSE must be finite.")
        self.base_validation_rmse = value
        if margin is not None:
            margin = float(margin)
            if margin < 0:
                raise ValueError("RMSE non-inferiority margin must be nonnegative.")
            self.rmse_noninferiority_margin = margin
        model = _unwrap_model(self.model)
        stored = getattr(model, "anchored_low_rank_base_validation_rmse", None)
        if torch.is_tensor(stored):
            stored.fill_(value)

    def _selection_base_rmse(self):
        model = _unwrap_model(self.model)
        stored = getattr(model, "anchored_low_rank_base_validation_rmse", None)
        if torch.is_tensor(stored) and bool(torch.isfinite(stored).item()):
            return float(stored.item())
        return float(self.base_validation_rmse)

    def measure_frozen_base_rmse(self, validation_loader):
        """Measure the exact warm-started backbone with all adapters bypassed."""

        self.model = self.model.to(self.device)
        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        model = _unwrap_model(self.model)
        was_training = bool(model.training)
        was_active = bool(getattr(model, "anchored_low_rank_active", True))
        model.set_anchored_low_rank_active(False)
        try:
            metrics = self.evaluate(
                {"rmse": PhysicalRMSE()},
                validation_loader,
                log_prefix="test",
                epoch=0,
            )
        finally:
            model.set_anchored_low_rank_active(was_active)
            model.train(was_training)
            if self.data_processor is not None:
                self.data_processor.train(was_training)
        value = float(metrics["test_rmse"])
        self.configure_selection_contract(base_rmse=value)
        return value

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        all_metrics = {}
        self._residual_monitor_active = bool(
            self.residual_decomposition_enabled
            and (int(epoch) == 0 or int(epoch) + 1 == int(self.n_epochs))
        )
        base_rmse = self._selection_base_rmse()
        if not torch.isfinite(torch.tensor(base_rmse)):
            raise RuntimeError(
                "ALR checkpoint selection requires a frozen-base validation RMSE."
            )
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                log_prefix=loader_name,
                epoch=epoch,
            )
            crps_key = f"{loader_name}_crps"
            rmse_key = f"{loader_name}_rmse"
            if crps_key not in loader_metrics or rmse_key not in loader_metrics:
                raise KeyError("ALR selection requires both CRPS and physical RMSE metrics.")
            raw_crps = float(loader_metrics[crps_key])
            rmse_delta = float(loader_metrics[rmse_key]) - base_rmse
            gate_passed = rmse_delta <= self.rmse_noninferiority_margin
            loader_metrics[f"{loader_name}_crps_unconstrained"] = raw_crps
            loader_metrics[f"{loader_name}_rmse_delta_from_base"] = rmse_delta
            loader_metrics[f"{loader_name}_rmse_gate_passed"] = float(gate_passed)
            if not gate_passed:
                loader_metrics[crps_key] = float("inf")
            all_metrics.update(loader_metrics)
        for name, total in self._residual_epoch_metric_sums.items():
            count = self._residual_epoch_metric_counts[name]
            all_metrics[f"train_{name}"] = total / max(1, count)
        if self.verbose:
            self.log_eval(epoch=epoch, eval_metrics=all_metrics)
        return all_metrics

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
        self._residual_epoch_metric_sums = {}
        self._residual_epoch_metric_counts = {}
        _unwrap_model(self.model).set_anchored_low_rank_training_phase(
            adapters_only=int(epoch) < self.adapter_warmup_epochs
        )

    def _prepare_sample(self, sample):
        if self.data_processor is not None:
            return self.data_processor.preprocess(sample)
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }

    def _sample_common_latents(self, *, count, batch_size, dtype):
        return torch.randn(
            int(count),
            int(batch_size),
            int(self.fgn_noise_dim),
            device=self.device,
            dtype=dtype,
        )

    def _forward_nested_x(self, sample, *, aleatory_samples, latent_bank=None):
        batch_size = int(sample["x"].shape[0])
        if latent_bank is None:
            latent_bank = self._sample_common_latents(
                count=aleatory_samples,
                batch_size=batch_size,
                dtype=sample["x"].dtype,
            )
        if latent_bank.shape != (
            int(aleatory_samples), batch_size, int(self.fgn_noise_dim)
        ):
            raise ValueError(
                "latent_bank must match [aleatory_samples, batch_size, fgn_noise_dim]."
            )
        nested = make_nested_particle_batch(
            sample["x"],
            latent_bank=latent_bank,
            num_particles=self.num_particles,
        )
        total_members = self.num_particles * int(aleatory_samples)
        values = nested.values.reshape(
            total_members, batch_size, *nested.values.shape[1:]
        )
        latents = nested.latents.reshape(
            total_members, batch_size, *nested.latents.shape[1:]
        )
        particle_ids = nested.particle_ids.reshape(total_members, batch_size)
        outputs = []
        chunk_size = min(self.eval_member_chunk_size, total_members)
        for start in range(0, total_members, chunk_size):
            stop = min(start + chunk_size, total_members)
            flat_count = (stop - start) * batch_size
            kwargs = {
                "x": values[start:stop].reshape(flat_count, *values.shape[2:]),
                "input_geom": sample["input_geom"],
                "latent_queries": sample["latent_queries"],
                "output_queries": sample["output_queries"],
                "ada_in": latents[start:stop].reshape(
                    flat_count, *latents.shape[2:]
                ),
                "particle_ids": particle_ids[start:stop].reshape(flat_count),
            }
            if self.mixed_precision:
                with torch.autocast(device_type=self.autocast_device_type):
                    out = self.model(**kwargs)
            else:
                out = self.model(**kwargs)
            outputs.append(
                out.reshape(stop - start, batch_size, *out.shape[1:])
            )
        output = torch.cat(outputs, dim=0)
        return output.reshape(
            self.num_particles,
            int(aleatory_samples),
            batch_size,
            *output.shape[2:],
        )

    def _forward_nested_ar_step(self, sample, *, histories, boundary, latent_bank):
        particles, aleatory, batch_size = histories.shape[:3]
        static = sample["static"]
        nested_static = static.unsqueeze(0).unsqueeze(0).expand(
            particles, aleatory, *static.shape
        ).reshape(particles * aleatory * batch_size, *static.shape[1:])
        nested_boundary = boundary.unsqueeze(0).unsqueeze(0).expand(
            particles, aleatory, *boundary.shape
        ).reshape(particles * aleatory * batch_size, *boundary.shape[1:])
        nested_histories = histories.reshape(
            particles * aleatory * batch_size, *histories.shape[3:]
        )
        x = _build_x_from_dynamic_boundary(
            nested_static,
            nested_boundary,
            nested_histories,
        )
        identity = make_nested_particle_batch(
            static,
            latent_bank=latent_bank,
            num_particles=particles,
        )
        kwargs = {
            "x": x,
            "input_geom": sample["input_geom"],
            "latent_queries": sample["latent_queries"],
            "output_queries": sample["output_queries"],
            "ada_in": identity.latents,
            "particle_ids": identity.particle_ids,
        }
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**kwargs)
        else:
            out = self.model(**kwargs)
        return out.reshape(particles, aleatory, batch_size, *out.shape[1:])

    def _spatial_weights(self, sample, target):
        if not self.use_flood_crps_spatial_weights or "static" not in sample:
            return None
        if target.shape[-1] < 3:
            return None
        return get_flood_crps_weights(
            sample["static"],
            target,
            wet_threshold=self.flood_crps_wet_threshold,
            wet_smooth_scale=self.flood_crps_wet_smooth_scale,
            dry_weight_alpha=self.flood_crps_dry_weight_alpha,
            static_normalizer=self.static_normalizer,
        )

    def _particle_loss(self, predictions, target, sample, training_loss):
        family_ids = _batch_family_ids(sample)
        family_weights = self.family_bootstrap.weights_for(family_ids)
        return particle_bootstrap_crps(
            predictions,
            target,
            family_weights=family_weights,
            loss_fn=training_loss,
            spatial_weights=self._spatial_weights(sample, target),
            structural_dry_mask=sample.get("structural_dry_mask"),
        )

    def _zero_latent_bank(self, *, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(
            1,
            int(batch_size),
            int(self.fgn_noise_dim),
            device=self.device,
            dtype=dtype,
        )

    def _residual_monitor_latent_bank(
        self, *, batch_size: int, dtype: torch.dtype
    ) -> torch.Tensor:
        base = self._residual_monitor_latents_cpu.to(
            device=self.device, dtype=dtype
        )
        return base.unsqueeze(1).expand(
            self.residual_monitor_samples,
            int(batch_size),
            int(self.fgn_noise_dim),
        )

    def _reference_mean_fields(self, sample, target, *, step: int = 0):
        time_index = sample.get("time_index")
        if time_index is None:
            raise KeyError("Residual decomposition requires time_index metadata.")
        family_ids = _batch_family_ids(sample)
        mean = self.reference_dispersion.lookup_reference_mean(
            family_ids, time_index, step=step
        )
        mean_variance = self.reference_dispersion.lookup_reference_mean_variance(
            family_ids, time_index, step=step
        )
        if mean is None or mean_variance is None:
            raise RuntimeError(
                "Residual-decomposition artifact is missing mean or mean variance fields."
            )
        if target.shape[-1] != 1:
            raise ValueError("The coastal residual pilot currently supports depth-only targets.")
        return (
            mean.to(device=target.device, dtype=target.dtype).unsqueeze(-1),
            mean_variance.to(device=target.device, dtype=target.dtype).unsqueeze(-1),
        )

    @staticmethod
    def _weighted_field_mean(field, family_weights, structural_dry_mask=None):
        batch_size, n_cells, channels = field.shape
        particles = family_weights.shape[0]
        active = torch.ones_like(field)
        if structural_dry_mask is not None:
            dry = torch.as_tensor(
                structural_dry_mask, device=field.device, dtype=torch.bool
            )
            while dry.ndim > 2 and dry.shape[-1] == 1:
                dry = dry.squeeze(-1)
            if dry.ndim == 1:
                dry = dry.unsqueeze(0).expand(batch_size, n_cells)
            if dry.shape != (batch_size, n_cells):
                raise ValueError("structural_dry_mask shape does not match field.")
            active = active * (~dry).unsqueeze(-1).to(dtype=field.dtype)
        weights = family_weights.to(device=field.device, dtype=field.dtype).view(
            particles, batch_size, 1, 1
        ) * active.unsqueeze(0)
        denominator = weights.sum(dim=(1, 2, 3)).clamp_min(
            torch.finfo(field.dtype).eps
        )
        per_particle = (field.unsqueeze(0) * weights).sum(dim=(1, 2, 3)) / denominator
        return per_particle.mean()

    def _residual_decomposition_loss(
        self,
        predictions,
        mean_predictions,
        target,
        sample,
        training_loss,
        *,
        step: int = 0,
    ):
        prediction_physical = self._to_physical(predictions)
        mean_physical = self._to_physical(mean_predictions)
        target_physical = self._to_physical(target)
        reference_mean, reference_mean_variance = self._reference_mean_fields(
            sample, target_physical, step=step
        )
        components = residual_decomposition_components(
            prediction_physical,
            mean_physical,
            target_physical,
            reference_mean,
        )
        family_weights = self.family_bootstrap.weights_for(_batch_family_ids(sample))
        mean_result = particle_bootstrap_mean_mse(
            mean_physical,
            reference_mean,
            family_weights=family_weights,
            structural_dry_mask=sample.get("structural_dry_mask"),
        )
        residual_result = particle_bootstrap_crps(
            components.model_residuals,
            components.reference_residual,
            family_weights=family_weights,
            loss_fn=training_loss,
            structural_dry_mask=sample.get("structural_dry_mask"),
        )
        reference_mean_variance_average = self._weighted_field_mean(
            reference_mean_variance,
            family_weights,
            sample.get("structural_dry_mask"),
        )
        data_loss = (
            self.mean_loss_weight * mean_result.mean
            + self.residual_crps_weight * residual_result.mean
        )
        return data_loss, mean_result, residual_result, reference_mean_variance_average

    def _anchor_penalty(self):
        return _unwrap_model(self.model).anchored_low_rank_offset_penalty()

    def _mean_residual_gradient_cosine(self, mean_loss, residual_loss):
        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        mean_gradients = torch.autograd.grad(
            mean_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        residual_gradients = torch.autograd.grad(
            residual_loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        return stable_gradient_cosine(
            mean_gradients,
            residual_gradients,
            reference=mean_loss,
        )

    def _record_residual_training_metrics(self, metrics):
        for name in (
            "alr_mean_mse_m2",
            "alr_mean_rmse_m",
            "alr_residual_crps_m",
            "alr_reference_mean_se_rms_m",
            "alr_mean_residual_gradient_cosine",
        ):
            if name not in metrics:
                continue
            value = float(torch.as_tensor(metrics[name]).detach().cpu())
            self._residual_epoch_metric_sums[name] = (
                self._residual_epoch_metric_sums.get(name, 0.0) + value
            )
            self._residual_epoch_metric_counts[name] = (
                self._residual_epoch_metric_counts.get(name, 0) + 1
            )

    def _to_physical(self, tensor: torch.Tensor) -> torch.Tensor:
        """Invert the training normalization; dispersion targets are in metres."""
        if tensor.ndim == 5:
            p, k, b, n, c = tensor.shape
            flat = tensor.reshape(p * k * b, n, c)
            return self.target_normalizer.inverse_transform(flat).reshape(p, k, b, n, c)
        return self.target_normalizer.inverse_transform(tensor)

    def _dispersion_penalty(self, predictions, target, sample, *, step: int = 0):
        """Pin the within-particle channel to the HEC-RAS reference dispersion.

        Returns ``None`` when disabled, so the loss graph is untouched and a
        weight of zero reproduces the unpinned objective exactly.
        """
        if self.dispersion_penalty_weight <= 0.0 or self.reference_dispersion is None:
            return None
        time_index = sample.get("time_index")
        if time_index is None:
            raise KeyError(
                "dispersion pinning requires time_index metadata on the batch."
            )
        families = _batch_family_ids(sample)
        reference = self.reference_dispersion.lookup(families, time_index, step=step)
        # Stratify on the reference-ensemble mean, matching the offline
        # calibration.  Without this the penalty is tuned against one
        # stratification and optimised against another.
        stratify_by = self.reference_dispersion.lookup_reference_mean(
            families, time_index, step=step
        )
        if stratify_by is not None:
            stratify_by = stratify_by.to(device=target.device, dtype=target.dtype)
        return dispersion_pinning_penalty(
            self._to_physical(predictions),
            self._to_physical(target),
            reference,
            structural_dry_mask=sample.get("structural_dry_mask"),
            wet_thresholds=self.dispersion_wet_thresholds,
            channel=self.water_depth_index,
            stratify_by=stratify_by,
        )

    @staticmethod
    def _dispersion_metrics(result, prefix: str = "alr_dispersion") -> dict:
        metrics = {f"{prefix}_penalty": result.penalty.detach()}
        for name, value in result.per_stratum_mean_deviation_m.items():
            metrics[f"{prefix}_bias_{name}_m"] = value
        return metrics

    @staticmethod
    def _particle_correlation_mean(predictions):
        particle_mean = predictions.mean(dim=1).reshape(predictions.shape[0], -1)
        centered = particle_mean - particle_mean.mean(dim=1, keepdim=True)
        normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1.0e-12)
        correlation = normalized @ normalized.transpose(0, 1)
        mask = ~torch.eye(
            correlation.shape[0], dtype=torch.bool, device=correlation.device
        )
        return correlation[mask].mean()

    @staticmethod
    def _particle_crps_loss(loss_fn, n_samples):
        if isinstance(loss_fn, CRPSLoss):
            return CRPSLoss(
                n_samples=int(n_samples),
                channel_weights=loss_fn.channel_weights,
                reduction=loss_fn.reduction,
            )
        if isinstance(loss_fn, FloodMaskedCRPSLoss) and isinstance(
            loss_fn.base_loss, CRPSLoss
        ):
            return FloodMaskedCRPSLoss(
                policy=loss_fn.policy,
                base_loss=CRPSLoss(
                    n_samples=int(n_samples),
                    channel_weights=loss_fn.base_loss.channel_weights,
                    reduction=loss_fn.base_loss.reduction,
                ),
            )
        return None

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        del idx
        predictions = self._forward_nested_x(
            sample,
            aleatory_samples=self.crps_n_samples,
        )
        if self.residual_decomposition_enabled:
            zero_latent = self._zero_latent_bank(
                batch_size=int(sample["x"].shape[0]), dtype=sample["x"].dtype
            )
            mean_predictions = self._forward_nested_x(
                sample, aleatory_samples=1, latent_bank=zero_latent
            ).squeeze(1)
            data_loss, mean_result, result, mean_variance = (
                self._residual_decomposition_loss(
                    predictions,
                    mean_predictions,
                    sample["y"],
                    sample,
                    training_loss,
                )
            )
        else:
            result = self._particle_loss(predictions, sample["y"], sample, training_loss)
            data_loss = result.mean
        penalty = self._anchor_penalty()
        loss = data_loss + self.anchor_penalty_weight * penalty
        dispersion = self._dispersion_penalty(predictions, sample["y"], sample)
        if dispersion is not None:
            loss = loss + self.dispersion_penalty_weight * dispersion.penalty
        metrics = {
            f"alr_particle_crps_{particle}": value.detach()
            for particle, value in enumerate(result.per_particle)
        }
        metrics["alr_anchor_penalty"] = penalty.detach()
        metrics["alr_anchor_displacement_norm"] = penalty.detach().sqrt()
        if self.residual_decomposition_enabled:
            metrics["alr_mean_mse_m2"] = mean_result.mean.detach()
            metrics["alr_mean_rmse_m"] = mean_result.mean.detach().sqrt()
            metrics["alr_residual_crps_m"] = result.mean.detach()
            metrics["alr_reference_mean_se_rms_m"] = mean_variance.detach().sqrt()
            if (
                self._residual_gradient_probes_done
                < self.residual_gradient_probe_batches
            ):
                metrics["alr_mean_residual_gradient_cosine"] = (
                    self._mean_residual_gradient_cosine(
                        self.mean_loss_weight * mean_result.mean,
                        self.residual_crps_weight * result.mean,
                    )
                )
                self._residual_gradient_probes_done += 1
        if dispersion is not None:
            metrics.update(self._dispersion_metrics(dispersion))
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                metrics["rel_l2"] = self.rel_l2_loss_fn(
                    predictions.mean(dim=(0, 1)),
                    sample["y"],
                    structural_dry_mask=sample.get("structural_dry_mask"),
                )
        self._record_residual_training_metrics(metrics)
        return loss, metrics

    def _train_one_batch_ar(self, idx, sample, training_loss):
        del idx
        target_sequence = sample["target_sequence"]
        boundary_sequence = sample["boundary_sequence"]
        if target_sequence.dim() == 5:
            target_sequence = target_sequence.squeeze(1)
        if boundary_sequence.dim() == 5:
            boundary_sequence = boundary_sequence.squeeze(1)
        n_ar_steps = self._effective_ar_steps(int(target_sequence.shape[1]))
        batch_size = int(sample["dynamic"].shape[0])
        aleatory = int(self.crps_n_samples)
        histories = sample["dynamic"].unsqueeze(0).unsqueeze(0).expand(
            self.num_particles,
            aleatory,
            *sample["dynamic"].shape,
        ).clone()
        mean_histories = None
        zero_latent_bank = None
        if self.residual_decomposition_enabled:
            mean_histories = sample["dynamic"].unsqueeze(0).unsqueeze(0).expand(
                self.num_particles,
                1,
                *sample["dynamic"].shape,
            ).clone()
            zero_latent_bank = self._zero_latent_bank(
                batch_size=batch_size, dtype=sample["dynamic"].dtype
            )
        boundary = sample["boundary"].clone()
        latent_bank = self._sample_common_latents(
            count=aleatory,
            batch_size=batch_size,
            dtype=sample["dynamic"].dtype,
        )

        gradient_mode = self.ar_gradient_mode
        if gradient_mode == "adaptive":
            gradient_mode = "truncated" if self._force_truncated_next_batch else "full"
        detach_every = self.ar_truncation_steps if gradient_mode == "truncated" else 0
        total_loss = None
        window_loss = None
        total_loss_scalar = 0.0
        last_result = None
        last_mean_result = None
        last_mean_variance = None
        last_dispersion = None
        last_rel_l2 = None
        penalty = None

        for step in range(n_ar_steps):
            target = target_sequence[:, step]
            predictions = self._forward_nested_ar_step(
                sample,
                histories=histories,
                boundary=boundary,
                latent_bank=latent_bank,
            )
            if self.residual_decomposition_enabled:
                mean_predictions = self._forward_nested_ar_step(
                    sample,
                    histories=mean_histories,
                    boundary=boundary,
                    latent_bank=zero_latent_bank,
                ).squeeze(1)
                data_loss, mean_result, result, mean_variance = (
                    self._residual_decomposition_loss(
                        predictions,
                        mean_predictions,
                        target,
                        sample,
                        training_loss,
                        step=step,
                    )
                )
                last_mean_result = mean_result
                last_mean_variance = mean_variance
            else:
                result = self._particle_loss(predictions, target, sample, training_loss)
                data_loss = result.mean
            penalty = self._anchor_penalty()
            step_loss = data_loss + self.anchor_penalty_weight * penalty
            dispersion = self._dispersion_penalty(
                predictions, target, sample, step=step
            )
            if dispersion is not None:
                step_loss = step_loss + self.dispersion_penalty_weight * dispersion.penalty
                last_dispersion = dispersion
            last_result = result
            total_loss_scalar += float(step_loss.detach())
            if gradient_mode == "truncated":
                window_loss = step_loss if window_loss is None else window_loss + step_loss
            else:
                total_loss = step_loss if total_loss is None else total_loss + step_loss

            if self.rel_l2_loss_fn is not None:
                with torch.no_grad():
                    last_rel_l2 = self.rel_l2_loss_fn(
                        predictions.mean(dim=(0, 1)),
                        target,
                        structural_dry_mask=sample.get("structural_dry_mask"),
                    )

            feedback = clamp_nested_feedback(
                predictions,
                structural_dry_mask=sample.get("structural_dry_mask"),
                target_normalizer=self.target_normalizer,
                water_depth_index=self.water_depth_index,
            )
            histories = update_nested_history(histories, feedback)
            if self.residual_decomposition_enabled:
                mean_feedback = clamp_nested_feedback(
                    mean_predictions.unsqueeze(1),
                    structural_dry_mask=sample.get("structural_dry_mask"),
                    target_normalizer=self.target_normalizer,
                    water_depth_index=self.water_depth_index,
                )
                mean_histories = update_nested_history(mean_histories, mean_feedback)

            next_boundary = boundary_sequence[:, step : step + 1]
            boundary = torch.cat([boundary[:, 1:], next_boundary], dim=1)

            truncation_boundary = detach_every > 0 and (step + 1) % detach_every == 0
            final_step = step + 1 == n_ar_steps
            if gradient_mode == "truncated" and (truncation_boundary or final_step):
                self._backward_ar_window(window_loss, n_ar_steps=n_ar_steps)
                window_loss = None
                histories = histories.detach()
                if mean_histories is not None:
                    mean_histories = mean_histories.detach()

        if gradient_mode == "truncated":
            loss = torch.tensor(
                total_loss_scalar / max(1, n_ar_steps),
                device=self.device,
                dtype=sample["dynamic"].dtype,
            )
        else:
            loss = total_loss / max(1, n_ar_steps)
        metrics = {
            f"alr_particle_crps_{particle}": value.detach()
            for particle, value in enumerate(last_result.per_particle)
        }
        metrics["alr_anchor_penalty"] = penalty.detach()
        metrics["alr_anchor_displacement_norm"] = penalty.detach().sqrt()
        if self.residual_decomposition_enabled:
            metrics["alr_mean_mse_m2"] = last_mean_result.mean.detach()
            metrics["alr_mean_rmse_m"] = last_mean_result.mean.detach().sqrt()
            metrics["alr_residual_crps_m"] = last_result.mean.detach()
            metrics["alr_reference_mean_se_rms_m"] = (
                last_mean_variance.detach().sqrt()
            )
        if last_dispersion is not None:
            metrics.update(self._dispersion_metrics(last_dispersion))
        if last_rel_l2 is not None:
            metrics["rel_l2"] = last_rel_l2
        if gradient_mode == "truncated":
            metrics["_backward_done"] = True
        self._record_residual_training_metrics(metrics)
        return loss, metrics

    def train_one_batch(self, idx, sample, training_loss):
        if not getattr(self, "_skip_internal_zero_grad", False):
            self.optimizer.zero_grad(set_to_none=True)
        sample = self._prepare_sample(sample)
        use_ar = (
            self.ar_rollout_steps > 1
            and self.epoch >= self.ar_finetune_start_epoch
            and sample.get("target_sequence") is not None
        )
        if use_ar:
            target_sequence = sample["target_sequence"]
            if target_sequence.dim() == 5:
                target_sequence = target_sequence.squeeze(1)
            increment = int(sample["y"].shape[0]) * self._effective_ar_steps(
                int(target_sequence.shape[1])
            )
            self.n_samples = int(getattr(self, "n_samples", 0)) + increment
            return self._train_one_batch_ar(idx, sample, training_loss)
        self.n_samples = int(getattr(self, "n_samples", 0)) + int(sample["y"].shape[0])
        return self._train_one_batch_single_step(idx, sample, training_loss)

    def eval_one_batch(self, sample, eval_losses, return_output=False):
        sample = self._prepare_sample(sample)
        batch_size = int(sample["y"].shape[0])
        self.n_samples = int(getattr(self, "n_samples", 0)) + batch_size
        predictions = self._forward_nested_x(
            sample,
            aleatory_samples=self.eval_aleatory_samples,
        )
        target = sample["y"]
        monitor_losses = {}
        if self.residual_decomposition_enabled and self._residual_monitor_active:
            zero_latent = self._zero_latent_bank(
                batch_size=batch_size, dtype=sample["x"].dtype
            )
            mean_predictions = self._forward_nested_x(
                sample, aleatory_samples=1, latent_bank=zero_latent
            ).squeeze(1)
            monitor_latents = self._residual_monitor_latent_bank(
                batch_size=batch_size, dtype=sample["x"].dtype
            )
            monitor_predictions = self._forward_nested_x(
                sample,
                aleatory_samples=self.residual_monitor_samples,
                latent_bank=monitor_latents,
            )
            mean_physical = self._to_physical(mean_predictions)
            monitor_physical = self._to_physical(monitor_predictions)
            target_physical = self._to_physical(target)
            reference_mean, reference_mean_variance = self._reference_mean_fields(
                sample, target_physical
            )
            unit_family_weights = torch.ones(
                self.num_particles,
                batch_size,
                device=target_physical.device,
                dtype=target_physical.dtype,
            )
            mean_result = particle_bootstrap_mean_mse(
                mean_physical,
                reference_mean,
                family_weights=unit_family_weights,
                structural_dry_mask=sample.get("structural_dry_mask"),
            )
            components = residual_decomposition_components(
                monitor_physical,
                mean_physical,
                target_physical,
                reference_mean,
            )
            crps_loss = self._particle_crps_loss(
                eval_losses.get("crps"), self.residual_monitor_samples
            )
            if crps_loss is None:
                raise TypeError(
                    "Residual monitoring requires a CRPS loss in eval_losses."
                )
            residual_result = particle_bootstrap_crps(
                components.model_residuals,
                components.reference_residual,
                family_weights=unit_family_weights,
                loss_fn=crps_loss,
                structural_dry_mask=sample.get("structural_dry_mask"),
            )
            centering = residual_centering_monitor(
                monitor_physical,
                mean_physical,
                structural_dry_mask=sample.get("structural_dry_mask"),
            )
            mean_variance_average = self._weighted_field_mean(
                reference_mean_variance,
                unit_family_weights,
                sample.get("structural_dry_mask"),
            )
            monitor_losses = {
                "alr_monitor_mean_rmse_m": mean_result.mean.sqrt(),
                "alr_monitor_residual_crps_m": residual_result.mean,
                "alr_monitor_centering_discrepancy_rms_m": centering.discrepancy_rms,
                "alr_monitor_centering_mc_se_rms_m": centering.mc_se_rms,
                "alr_monitor_centering_exceeds_2se_fraction": (
                    centering.exceeds_two_se_fraction
                ),
                "alr_monitor_reference_mean_se_rms_m": (
                    mean_variance_average.sqrt()
                ),
            }
        if self.data_processor is not None:
            flat = predictions.reshape(
                self.num_particles * self.eval_aleatory_samples * batch_size,
                *predictions.shape[3:],
            )
            flat, processed = self.data_processor.postprocess(
                flat,
                {
                    "y": target,
                    "structural_dry_mask": sample.get("structural_dry_mask"),
                },
            )
            predictions = flat.reshape(
                self.num_particles,
                self.eval_aleatory_samples,
                batch_size,
                *flat.shape[1:],
            )
            target = processed["y"]
        mixture = predictions.reshape(
            self.num_particles * self.eval_aleatory_samples,
            batch_size,
            *predictions.shape[3:],
        )
        mean_prediction = mixture.mean(dim=0)
        losses = {}
        for name, loss_fn in eval_losses.items():
            expects_samples = (
                getattr(loss_fn, "expects_samples", False)
                or isinstance(loss_fn, CRPSLoss)
                or name == "crps"
            )
            losses[name] = loss_fn(
                mixture if expects_samples else mean_prediction,
                target,
                structural_dry_mask=sample.get("structural_dry_mask"),
            )
            if expects_samples:
                particle_loss_fn = self._particle_crps_loss(
                    loss_fn, self.eval_aleatory_samples
                )
                if particle_loss_fn is None:
                    continue
                for particle in range(self.num_particles):
                    losses[f"alr_particle_{name}_{particle}"] = particle_loss_fn(
                        predictions[particle],
                        target,
                        structural_dry_mask=sample.get("structural_dry_mask"),
                    )
        losses["alr_particle_correlation_mean"] = self._particle_correlation_mean(
            predictions
        )
        losses.update(monitor_losses)
        penalty = self._anchor_penalty()
        losses["alr_anchor_displacement_norm"] = penalty.sqrt()
        return losses, mean_prediction if return_output else None
