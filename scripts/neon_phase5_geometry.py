#!/usr/bin/env python3
"""Common-anchor gradient geometry and local-sensitivity diagnostics (D2/D1)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from neuralop.flood.eval.neon_direct import (
    BatchedDirectLastLayer,
    direct_weighted_scores,
    fit_batched_direct_particles,
)
from neuralop.flood.eval.neon_phase5 import (
    bootstrap_scale_interval,
    common_anchor_gradient_geometry,
    gradient_row_geometry,
    phase5_predeclared_protocol,
    raw_probit_weight_rank,
    verify_checksummed_artifact,
    write_checksummed_artifact,
)
from neuralop.flood.neon import (
    epistemic_bootstrap_weights,
    per_epistemic_fair_crps,
    probit_exponential_raw_weights,
)
from neon_phase5_direct import DirectProvider
from neon_phase5_runtime import load_phase5_context, write_provenance


LOG = logging.getLogger("neon_phase5_geometry")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in (
        "config",
        "bundle",
        "checkpoint",
        "history",
        "preflight",
        "phase5_preflight",
        "cache_dir",
        "protocol",
        "output_dir",
        "expected_head",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--anchor-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--validation-locations-per-family", type=int, default=256)
    parser.add_argument("--noise-batch-size", type=int, default=8)
    parser.add_argument("--indexed-gradient-draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _flat_grad(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in module.parameters()]
    )


def _sample_validation_jacobian(
    provider: DirectProvider,
    *,
    count: int,
    seed: int,
    physical_scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    family_rows = []
    if provider.context.val_families:
        provider.prefetch(0, validation=True)
    for index, _family in enumerate(provider.context.val_families):
        frozen = provider.load(index, validation=True)
        if index + 1 < len(provider.context.val_families):
            provider.prefetch(index + 1, validation=True)
        feature = frozen.features.double().mean(dim=0).reshape(-1, frozen.features.shape[-1])
        bias = torch.ones(feature.shape[0], 1, dtype=feature.dtype, device=feature.device)
        design = torch.cat([feature, bias], dim=1)
        if frozen.score_weights is None:
            probabilities = torch.ones(feature.shape[0], dtype=torch.float64)
        else:
            probabilities = frozen.score_weights.double().reshape(-1)
        probabilities = probabilities / probabilities.sum().clamp_min(1.0e-15)
        selected = torch.multinomial(
            probabilities.cpu(), int(count), replacement=True, generator=generator
        ).to(design.device)
        family_rows.append(design.index_select(0, selected).cpu() * float(physical_scale))
    return torch.cat(family_rows, dim=0), family_rows


def _local_scale(
    family_jacobians: list[torch.Tensor],
    displacement: torch.Tensor,
    *,
    seed: int,
) -> dict[str, float]:
    contributions = []
    delta = displacement.double().T
    for jacobian in family_jacobians:
        effects = jacobian.double() @ delta
        centered = effects - effects.mean(dim=1, keepdim=True)
        contributions.append(centered.square().mean(dim=0))
    return bootstrap_scale_interval(
        torch.stack(contributions, dim=0), replicates=2000, seed=seed
    )


def _indexed_prediction_state_gradients(
    provider: DirectProvider,
    z_e: torch.Tensor,
    particle_weights: torch.Tensor,
) -> torch.Tensor:
    """Gradients at B3 indexed predictions in a common direct-head coordinate system."""

    draws = int(z_e.shape[0])
    probe = provider.load(0)
    head = BatchedDirectLastLayer(
        draws,
        feature_channels=int(probe.features.shape[-1]),
        out_channels=1,
    ).to(provider.context.device)
    normalized = particle_weights.double()
    normalized = normalized / normalized.mean(dim=0, keepdim=True).clamp_min(1.0e-15)
    head.zero_grad(set_to_none=True)
    count = len(provider.context.train_families)
    if count:
        provider.prefetch(0)
    for index in range(count):
        family = provider.load(index)
        if index + 1 < count:
            provider.prefetch(index + 1)
        state = provider.stage2_prediction_state(index, z_e)
        correction = head(family.features)
        reference = family.reference.to(device=state.device, dtype=state.dtype)
        score_weights = (
            None
            if family.score_weights is None
            else family.score_weights.to(device=state.device, dtype=state.dtype)
        )
        scores = per_epistemic_fair_crps(
            (state + correction).unsqueeze(0),
            reference.unsqueeze(0),
            weights=score_weights,
            reduction="none",
        )[0]
        contribution = (
            scores
            * normalized[index].to(device=state.device, dtype=state.dtype)
        ).mean() / float(count)
        contribution.backward()
    # The scalar objective averages particles. Undo that harmless scale so
    # each row is the gradient of its own normalized empirical risk.
    rows = torch.cat(
        [head.weight.grad.reshape(draws, -1), head.bias.grad.reshape(draws, -1)],
        dim=1,
    )
    return (float(draws) * rows).detach().cpu().double()


def main() -> int:
    args = parse_args()
    protocol_sha = verify_checksummed_artifact(args.protocol)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "neon_phase5_geometry_plan_v1",
        "fit_families": 450,
        "validation_families": 50,
        "draws": int(args.draws),
        "gradient_anchor": "one_uniform_last_layer_optimum",
        "curvature_estimator": "empirical_family_gradient_gram_psd",
        "hessian_limitation": (
            "gradient-Gram curvature is a declared local preconditioner; negative curvature "
            "of the DC CRPS objective is not identifiable from this proxy"
        ),
        "validation_functional": "fixed weighted spatial sample in physical meters",
        "indexed_gradient_draws": int(args.indexed_gradient_draws),
        "noise_batch_size": int(args.noise_batch_size),
        "indexed_gradient_semantics": (
            "B3 prediction-state score gradients in the same direct last-layer coordinates; "
            "reported separately and never attributed solely to bootstrap weights"
        ),
    }
    write_checksummed_artifact(output / "PLAN.json", plan)
    if args.plan_only:
        return 0
    context = load_phase5_context(
        config_path=args.config,
        bundle_path=args.bundle,
        checkpoint_path=args.checkpoint,
        history_path=args.history,
        preflight_path=args.preflight,
        phase5_preflight_path=args.phase5_preflight,
        cache_dir=args.cache_dir,
        device=args.device,
        expected_head=args.expected_head,
    )
    write_provenance(
        output,
        head=context.git_head,
        checkpoint=context.checkpoint_path,
        cache_dir=context.cache_dir,
        protocol_sha256=protocol_sha,
        frozen_inputs=context.run_metadata["frozen_inputs"],
    )
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    z_e = torch.randn(
        int(args.draws), int(context.stage2.epistemic_dim), generator=generator
    ).to(context.device)
    provider = DirectProvider(context, z_e, prior_chunk=4)
    probe = provider.load(0)
    anchor = BatchedDirectLastLayer(
        1, feature_channels=int(probe.features.shape[-1]), out_channels=1
    ).to(context.device)
    anchor_fit = fit_batched_direct_particles(
        anchor,
        family_ids=[family.family_id for family in context.train_families],
        load_family=provider.load,
        family_particle_weights=torch.ones(450, 1),
        epochs=args.anchor_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        shuffle_seed=args.seed,
        objective_audit_interval=3,
        prefetch_family=provider.prefetch,
    )

    per_family = []
    per_family_scores = []
    train_count = len(context.train_families)
    if train_count != int(plan["fit_families"]):
        raise ValueError(
            f"D2 requires {plan['fit_families']} fit families, got {train_count}."
        )
    if len(context.val_families) != int(plan["validation_families"]):
        raise ValueError(
            "D2 requires "
            f"{plan['validation_families']} validation families, got "
            f"{len(context.val_families)}."
        )
    provider.prefetch(0)
    for index in range(train_count):
        family = provider.load(index)
        if index + 1 < train_count:
            provider.prefetch(index + 1)
        anchor.zero_grad(set_to_none=True)
        score = direct_weighted_scores(
            anchor, family, particle_weights=torch.ones(1, device=context.device)
        ).mean()
        per_family_scores.append(score.detach().cpu())
        score.backward()
        per_family.append(_flat_grad(anchor).cpu())
    gradients = torch.stack(per_family).double()
    family_ids = [family.family_id for family in context.train_families]
    bootstrap = dict(context.stage2_metadata.get("bootstrap", {}))
    distribution = str(bootstrap.get("distribution", "")).strip().lower()
    if distribution != "probit_exponential":
        raise ValueError(
            "D2 requires the B3 probit-exponential law, got "
            f"{distribution!r}."
        )
    weights = epistemic_bootstrap_weights(
        family_ids,
        z_e,
        seed=int(bootstrap.get("seed", 0)),
        distribution=distribution,
        temperature=float(bootstrap.get("temperature", 0.5)),
        normalize=str(bootstrap.get("normalize", "per_epistemic_batch")),
        min_weight=float(bootstrap.get("min_weight", 0.05)),
        max_weight=float(bootstrap.get("max_weight", 5.0)),
    ).double().cpu()
    raw_weights = probit_exponential_raw_weights(
        family_ids, z_e, seed=int(bootstrap.get("seed", 0))
    ).double().cpu()

    centered_gradients = gradients - gradients.mean(dim=0, keepdim=True)
    curvature = centered_gradients.T @ centered_gradients / float(gradients.shape[0])
    curvature = curvature + 1.0e-8 * torch.eye(curvature.shape[0], dtype=curvature.dtype)
    noise = []
    for _ in range(int(args.draws)):
        idx = torch.randint(
            gradients.shape[0],
            (int(args.noise_batch_size),),
            generator=generator,
        )
        noise.append(gradients.index_select(0, idx).mean(dim=0) - gradients.mean(dim=0))
    noise = torch.stack(noise)
    validation_jacobian, family_jacobians = _sample_validation_jacobian(
        provider,
        count=args.validation_locations_per_family,
        seed=args.seed,
        physical_scale=context.physical_scale_m,
    )
    geometry = common_anchor_gradient_geometry(
        gradients,
        weights,
        minibatch_noise_gradients=noise,
        hessian=curvature,
        validation_jacobian=validation_jacobian,
        damping=(1.0e-6, 1.0e-4, 1.0e-2),
    )
    local_scales = {
        str(damping): _local_scale(
            family_jacobians, displacement, seed=args.seed + int(1e6 * damping)
        )
        for damping, displacement in geometry.displacements.items()
    }
    indexed_count = min(int(args.indexed_gradient_draws), int(args.draws))
    if indexed_count < 2:
        raise ValueError("indexed-gradient diagnostic requires at least two draws.")
    indexed_gradients = _indexed_prediction_state_gradients(
        provider,
        z_e[:indexed_count],
        weights[:, :indexed_count],
    )
    indexed_summary = gradient_row_geometry(indexed_gradients)
    bootstrap_delta = geometry.delta_gradients[:indexed_count]
    matched_cosine = (
        indexed_gradients * bootstrap_delta
    ).sum(dim=1) / (
        indexed_gradients.norm(dim=1)
        * bootstrap_delta.norm(dim=1)
    ).clamp_min(1.0e-15)
    rank = raw_probit_weight_rank(raw_weights, epistemic_indices=z_e.cpu())
    ess = weights.sum(dim=0).square() / weights.square().sum(dim=0)
    report = {
        **plan,
        "anchor_fit": {
            "best_epoch": anchor_fit.best_epoch,
            "best_loss": anchor_fit.best_loss,
            "final_gradient_norm": anchor_fit.final_gradient_norm,
        },
        "gradient_geometry": {
            "effective_rank": geometry.gradient_effective_rank,
            "participation_ratio": geometry.gradient_participation_ratio,
            "pairwise_cosine_mean": geometry.pairwise_cosine_mean,
            "pairwise_absolute_cosine_mean": geometry.pairwise_absolute_cosine_mean,
            "signal_to_minibatch_noise": geometry.gradient_signal_to_noise,
            "hessian_proxy": geometry.hessian,
            "functional_displacement_rms_m": {
                str(key): value for key, value in geometry.functional_displacement_rms.items()
            },
            "local_sensitivity_scale_m": local_scales,
        },
        "indexed_prediction_state_gradient_geometry": {
            "draws": indexed_count,
            "numerical_rank": int(indexed_summary["numerical_rank"]),
            "effective_rank": float(indexed_summary["effective_rank"]),
            "participation_ratio": float(indexed_summary["participation_ratio"]),
            "pairwise_cosine_mean": float(indexed_summary["pairwise_cosine_mean"]),
            "pairwise_absolute_cosine_mean": float(
                indexed_summary["pairwise_absolute_cosine_mean"]
            ),
            "row_norm_mean": float(indexed_summary["row_norm_mean"]),
            "matched_cosine_with_bootstrap_only_delta_mean": float(
                matched_cosine.mean()
            ),
            "matched_cosine_with_bootstrap_only_delta_abs_mean": float(
                matched_cosine.abs().mean()
            ),
            "attribution_warning": (
                "contains prior and indexed prediction-state effects; do not label as "
                "bootstrap differentiation"
            ),
        },
        "weight_geometry": {
            "raw_inverse_probit_rank": int(rank["numerical_rank"]),
            "raw_inverse_probit_participation_ratio": float(rank["participation_ratio"]),
            "raw_inverse_probit_effective_rank": float(rank["effective_rank"]),
            "raw_inverse_probit_tolerance": float(rank["tolerance"]),
            "off_diagonal_family_logit_correlation_mean": float(
                rank["off_diagonal_family_logit_correlation_mean"]
            ),
            "off_diagonal_family_logit_abs_correlation_mean": float(
                rank["off_diagonal_family_logit_abs_correlation_mean"]
            ),
            "expected_random_direction_abs_correlation": float(
                rank["expected_random_direction_abs_correlation"]
            ),
            "jacobian_numerical_rank": int(rank["jacobian_numerical_rank"]),
            "jacobian_participation_ratio": float(
                rank["jacobian_participation_ratio"]
            ),
            "jacobian_effective_rank": float(rank["jacobian_effective_rank"]),
            "jacobian_singular_values": [
                float(value) for value in rank["jacobian_singular_values"]
            ],
            "jacobian_relative_reconstruction_error": float(
                rank["jacobian_relative_reconstruction_error"]
            ),
            "ess_min": float(ess.min()),
            "ess_mean": float(ess.mean()),
            "ess_max": float(ess.max()),
        },
    }
    write_checksummed_artifact(output / "RESULT.json", report)
    torch.save(
        {
            "anchor_state_dict": anchor.state_dict(),
            "z_e": z_e.cpu(),
            "family_ids": family_ids,
            "per_family_gradients": gradients,
            "per_family_anchor_scores": torch.stack(per_family_scores).double(),
            "noise_batch_size": int(args.noise_batch_size),
            "weights": weights,
            "raw_weights": raw_weights,
            "raw_inverse_probit_logits": rank["inverse_logits"],
            "raw_logit_jacobian": rank["jacobian"],
            "minibatch_noise_gradients": noise,
            "curvature_proxy": curvature,
            "validation_jacobian_physical": validation_jacobian,
            "displacements": geometry.displacements,
            "indexed_prediction_state_gradients": indexed_gradients,
            "indexed_prediction_state_singular_values": indexed_summary[
                "singular_values"
            ],
            "indexed_to_bootstrap_delta_cosines": matched_cosine,
        },
        output / "GEOMETRY.pt",
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    LOG.info("Phase-5 D2/D1 complete: %s", report)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
