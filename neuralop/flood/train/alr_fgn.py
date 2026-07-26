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
from neuralop.flood.train.fgn import FGNTrainer
from neuralop.flood.utils.runtime_core import parse_family_id_from_run_id
from neuralop.losses.probabilistic_losses import CRPSLoss


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
    return ALRFamilySplit(
        train_indices=[
            index for index, family in enumerate(family_by_index) if family in train_set
        ],
        validation_indices=[
            index
            for index, family in enumerate(family_by_index)
            if family in validation_set
        ],
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
        target_normalizer=None,
        water_depth_index: int = 0,
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
        self.target_normalizer = target_normalizer
        self.water_depth_index = int(water_depth_index)

    def on_epoch_start(self, epoch):
        super().on_epoch_start(epoch)
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

    def _forward_nested_x(self, sample, *, aleatory_samples):
        batch_size = int(sample["x"].shape[0])
        latent_bank = self._sample_common_latents(
            count=aleatory_samples,
            batch_size=batch_size,
            dtype=sample["x"].dtype,
        )
        nested = make_nested_particle_batch(
            sample["x"],
            latent_bank=latent_bank,
            num_particles=self.num_particles,
        )
        kwargs = {
            "x": nested.values,
            "input_geom": sample["input_geom"],
            "latent_queries": sample["latent_queries"],
            "output_queries": sample["output_queries"],
            "ada_in": nested.latents,
            "particle_ids": nested.particle_ids,
        }
        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**kwargs)
        else:
            out = self.model(**kwargs)
        return out.reshape(
            self.num_particles,
            int(aleatory_samples),
            batch_size,
            *out.shape[1:],
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

    def _anchor_penalty(self):
        return _unwrap_model(self.model).anchored_low_rank_offset_penalty()

    def _train_one_batch_single_step(self, idx, sample, training_loss):
        del idx
        predictions = self._forward_nested_x(
            sample,
            aleatory_samples=self.crps_n_samples,
        )
        result = self._particle_loss(predictions, sample["y"], sample, training_loss)
        penalty = self._anchor_penalty()
        loss = result.mean + self.anchor_penalty_weight * penalty
        metrics = {
            f"alr_particle_crps_{particle}": value.detach()
            for particle, value in enumerate(result.per_particle)
        }
        metrics["alr_anchor_penalty"] = penalty.detach()
        if self.rel_l2_loss_fn is not None:
            with torch.no_grad():
                metrics["rel_l2"] = self.rel_l2_loss_fn(
                    predictions.mean(dim=(0, 1)),
                    sample["y"],
                    structural_dry_mask=sample.get("structural_dry_mask"),
                )
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
            result = self._particle_loss(predictions, target, sample, training_loss)
            penalty = self._anchor_penalty()
            step_loss = result.mean + self.anchor_penalty_weight * penalty
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

            next_boundary = boundary_sequence[:, step : step + 1]
            boundary = torch.cat([boundary[:, 1:], next_boundary], dim=1)

            truncation_boundary = detach_every > 0 and (step + 1) % detach_every == 0
            final_step = step + 1 == n_ar_steps
            if gradient_mode == "truncated" and (truncation_boundary or final_step):
                self._backward_ar_window(window_loss, n_ar_steps=n_ar_steps)
                window_loss = None
                histories = histories.detach()

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
        if last_rel_l2 is not None:
            metrics["rel_l2"] = last_rel_l2
        if gradient_mode == "truncated":
            metrics["_backward_done"] = True
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
        return losses, mean_prediction if return_output else None
