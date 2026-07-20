#!/usr/bin/env python3
"""Law-matched direct-particle re-optimization for NEON Phase 5 D1-prime."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuralop.flood.eval.impact_metrics import compute_nested_flood_impact_crps_metrics
from neuralop.flood.eval.neon_direct import (
    BatchedDirectFullHead,
    BatchedDirectLastLayer,
    DirectFamilyBatch,
    audit_direct_objective,
    direct_particle_mean_scale_interval,
    fit_batched_direct_particles,
    matched_particle_displacement_interval,
    prior_linear_representability_error,
    subset_direct_module,
)
from neuralop.flood.eval.neon_phase5 import (
    phase5_predeclared_protocol,
    verify_checksummed_artifact,
    write_checksummed_artifact,
)
from neuralop.flood.neon import (
    PersistentDirichletParticleControl,
    base_rmse_from_reference,
    crossed_fair_crps_members,
    epistemic_bootstrap_weights,
    fixed_support_fair_crps_members,
    project_bootstrap_epistemic_indices,
    sample_epistemic_indices,
)
from neon_phase5_runtime import (
    Phase5Context,
    collect_cached_family,
    inverse_predictions,
    inverse_reference,
    load_phase5_context,
    write_provenance,
)


LOG = logging.getLogger("neon_phase5_direct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--phase5-preflight", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--mode", choices=("data", "rpf"), required=True)
    parser.add_argument("--law", choices=("probit", "dirichlet"), default="probit")
    parser.add_argument("--head", choices=("last", "full"), default="last")
    parser.add_argument("--draws", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--anchor-epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--objective-audit-interval", type=int, default=3)
    parser.add_argument("--prior-chunk", type=int, default=4)
    parser.add_argument("--multistart-spot-draws", type=int)
    parser.add_argument("--multistart-perturbation", type=float)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _module(args: argparse.Namespace, particles: int, feature_channels: int) -> torch.nn.Module:
    if args.head == "last":
        return BatchedDirectLastLayer(
            particles, feature_channels=feature_channels, out_channels=1
        )
    return BatchedDirectFullHead(
        particles,
        feature_channels=feature_channels,
        hidden_channels=16,
        out_channels=1,
    )


class DirectProvider:
    def __init__(self, context: Phase5Context, z_e: torch.Tensor, prior_chunk: int) -> None:
        self.context = context
        self.z_e = z_e
        self.prior_chunk = max(1, int(prior_chunk))
        self._key: tuple[bool, int] | None = None
        self._batch = None
        self._family_batch = None

    def prefetch(self, index: int, *, validation: bool = False) -> bool:
        """Schedule one cached family payload without changing provider state."""

        prefetch = getattr(self.context.collector, "prefetch", None)
        if not callable(prefetch):
            return False
        families = self.context.val_families if validation else self.context.train_families
        return bool(
            prefetch(
                families[int(index)],
                num_aleatory=8,
                latent_bank_id=0,
            )
        )

    def load(self, index: int, *, validation: bool = False) -> DirectFamilyBatch:
        families = self.context.val_families if validation else self.context.train_families
        family = families[int(index)]
        batch = collect_cached_family(self.context, family, k=8, bank=0)
        with torch.no_grad():
            dummy = self.z_e[:1].to(batch.features)
            output = self.context.stage2(
                batch.base_prediction,
                batch.features,
                dummy,
                node_coords=family.geometry,
                canonical_mean_features=batch.canonical_mean_features,
            )
            fixed_base = (
                batch.base_prediction + output.deterministic_correction[:, 0]
            )[0]
        self._key = (bool(validation), int(index))
        self._batch = batch
        self._family_batch = DirectFamilyBatch(
            family_id=family.family_id,
            base_prediction=fixed_base,
            features=batch.features[0],
            reference=family.reference,
            score_weights=family.weights,
        )
        return self._family_batch

    def prior(self, index: int, *, validation: bool = False) -> torch.Tensor:
        if self._key != (bool(validation), int(index)) or self._batch is None:
            self.load(index, validation=validation)
        family = (
            self.context.val_families[int(index)]
            if validation
            else self.context.train_families[int(index)]
        )
        chunks = []
        with torch.no_grad():
            for start in range(0, self.z_e.shape[0], self.prior_chunk):
                raw = self.context.stage2.compute_prior(
                    self._batch.features,
                    self.z_e[start : start + self.prior_chunk].to(self._batch.features),
                    node_coords=family.geometry,
                )
                chunks.append(float(self.context.stage2.alpha) * raw[0])
        return torch.cat(chunks, dim=0)

    def stage2_prediction_state(
        self,
        index: int,
        z_e: torch.Tensor,
        *,
        validation: bool = False,
        chunk: int | None = None,
    ) -> torch.Tensor:
        """Return B3 indexed predictions, isolated from direct-head gradients."""

        if self._key != (bool(validation), int(index)) or self._batch is None:
            self.load(index, validation=validation)
        families = self.context.val_families if validation else self.context.train_families
        family = families[int(index)]
        size = max(1, int(chunk or self.prior_chunk))
        predictions = []
        with torch.no_grad():
            for start in range(0, int(z_e.shape[0]), size):
                output = self.context.stage2(
                    self._batch.base_prediction,
                    self._batch.features,
                    z_e[start : start + size].to(self._batch.features),
                    node_coords=family.geometry,
                    canonical_mean_features=self._batch.canonical_mean_features,
                )
                predictions.append(output.prediction[0])
        return torch.cat(predictions, dim=0)


def _support_and_weights(
    context: Phase5Context,
    *,
    law: str,
    draws: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    ids = [family.family_id for family in context.train_families]
    if law == "dirichlet":
        raw = context.stage2_metadata.get("dirichlet_particle_control")
        if not raw:
            raise ValueError("Dirichlet direct optimization requires a law-matched P1a checkpoint.")
        control = PersistentDirichletParticleControl.from_metadata(raw)
        if int(draws) != control.num_particles:
            raise ValueError("Dirichlet direct support must use every persistent particle.")
        indices = torch.arange(control.num_particles)
        z_e = control.eval_epistemic_indices(device=context.device, dtype=torch.float32)
        weights = control.family_particle_weights(ids, indices, dtype=torch.float64)
        return z_e, weights, "fixed_support"
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    model_dim = int(context.stage2.epistemic_dim)
    bootstrap_dim = int(
        context.stage2_metadata.get("bootstrap_index_dim") or model_dim
    )
    bootstrap_index = sample_epistemic_indices(
        int(draws),
        bootstrap_dim,
        device=context.device,
        generator=generator,
    )
    z_e = project_bootstrap_epistemic_indices(
        bootstrap_index,
        model_dim=model_dim,
        projection=context.stage2_metadata.get("bootstrap_model_projection"),
    )
    bootstrap = dict(context.stage2_metadata.get("bootstrap", {}))
    distribution = str(bootstrap.get("distribution", "")).strip().lower()
    if distribution != "probit_exponential":
        raise ValueError(
            "Phase-5 probit direct law requires checkpoint bootstrap distribution "
            f"'probit_exponential', got {distribution!r}."
        )
    weights = epistemic_bootstrap_weights(
        ids,
        bootstrap_index,
        seed=int(bootstrap.get("seed", 0)),
        distribution=distribution,
        temperature=float(bootstrap.get("temperature", 0.5)),
        normalize=str(bootstrap.get("normalize", "per_epistemic_batch")),
        min_weight=float(bootstrap.get("min_weight", 0.05)),
        max_weight=float(bootstrap.get("max_weight", 5.0)),
    ).double().cpu()
    return z_e, weights, "crossed"


def _copy_anchor(destination: torch.nn.Module, source: torch.nn.Module) -> None:
    if isinstance(destination, BatchedDirectLastLayer):
        destination.copy_particle_from(source, source_particle=0)
    else:
        destination.copy_from_shared_anchor(source, source_particle=0)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else math.nan


def _perturb_module(module: torch.nn.Module, *, scale: float, seed: int) -> None:
    """Apply one deterministic alternative initialization for a spot check."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.no_grad():
        for parameter in module.parameters():
            noise = torch.randn(
                parameter.shape, generator=generator, dtype=torch.float32
            ).to(device=parameter.device, dtype=parameter.dtype)
            parameter.add_(float(scale) * noise)


def _multistart_spot_check(
    *,
    context: Phase5Context,
    direct: torch.nn.Module,
    anchor_template: torch.nn.Module,
    z_e: torch.Tensor,
    bootstrap_weights: torch.Tensor,
    mode: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    objective_audit_interval: int,
    prior_chunk: int,
    spot_draws: int,
    perturbation: float,
    seed: int,
) -> dict[str, Any]:
    """Compare the primary optimum with a perturbed second start."""

    count = min(int(spot_draws), int(z_e.shape[0]))
    if count < 1:
        raise ValueError("multistart spot check requires at least one draw.")
    indices = list(range(count))
    provider = DirectProvider(context, z_e[:count], prior_chunk)
    family_ids = [family.family_id for family in context.train_families]
    weights = bootstrap_weights[:, :count]
    prior = provider.prior if mode == "rpf" else None
    primary = subset_direct_module(direct, indices)
    primary_objective, primary_gradient = audit_direct_objective(
        primary,
        family_ids=family_ids,
        load_family=provider.load,
        family_particle_weights=weights,
        prior_slice=prior,
        regularization=float(weight_decay),
        prefetch_family=provider.prefetch,
    )

    alternative = subset_direct_module(anchor_template, indices)
    _perturb_module(alternative, scale=perturbation, seed=seed)
    alternative_fit = fit_batched_direct_particles(
        alternative,
        family_ids=family_ids,
        load_family=provider.load,
        family_particle_weights=weights,
        prior_slice=prior,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        shuffle_seed=seed + 1,
        objective_audit_interval=objective_audit_interval,
        prefetch_family=provider.prefetch,
    )
    denominator = max(
        min(abs(primary_objective), abs(alternative_fit.final_objective)), 1.0e-12
    )
    relative_gap = abs(primary_objective - alternative_fit.final_objective) / denominator
    return {
        "draws": count,
        "primary_objective": primary_objective,
        "primary_gradient_norm": primary_gradient,
        "alternative_initial_objective": alternative_fit.initial_objective,
        "alternative_final_objective": alternative_fit.final_objective,
        "alternative_final_gradient_norm": alternative_fit.final_gradient_norm,
        "alternative_gradient_reduction_ratio": alternative_fit.gradient_reduction_ratio,
        "relative_objective_gap": relative_gap,
        "initialization_perturbation": float(perturbation),
        "seed": int(seed),
    }


def main() -> int:
    args = parse_args()
    protocol_sha = verify_checksummed_artifact(args.protocol)
    protocol = phase5_predeclared_protocol()
    default_draws = protocol["direct"]["last_layer_draws" if args.head == "last" else "full_head_draws"]
    draws = int(args.draws or default_draws)
    spot_draws = int(
        args.multistart_spot_draws
        or protocol["direct"][
            "multistart_last_layer_draws"
            if args.head == "last"
            else "multistart_full_head_draws"
        ]
    )
    perturbation = float(
        args.multistart_perturbation
        if args.multistart_perturbation is not None
        else protocol["direct"]["multistart_initialization_perturbation"]
    )
    plan = {
        "schema_version": "neon_phase5_direct_plan_v1",
        "mode": args.mode,
        "law": args.law,
        "head": args.head,
        "draws": draws,
        "fit_families": 450,
        "validation_families": 50,
        "prior_policy": "alpha_zero" if args.mode == "data" else "law_matched_fixed_prior",
        "representability_caveat": args.head == "last" and args.mode == "rpf",
        "physical_space_evaluation": True,
        "multistart_spot_draws": spot_draws,
        "multistart_initialization_perturbation": perturbation,
        "objective_audit_interval": int(args.objective_audit_interval),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_checksummed_artifact(output_dir / "PLAN.json", plan)
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
        output_dir,
        head=context.git_head,
        checkpoint=context.checkpoint_path,
        cache_dir=context.cache_dir,
        protocol_sha256=protocol_sha,
        frozen_inputs=context.run_metadata["frozen_inputs"],
    )
    z_e, bootstrap_weights, sampling_design = _support_and_weights(
        context, law=args.law, draws=draws, seed=args.seed
    )
    provider = DirectProvider(context, z_e, args.prior_chunk)
    probe = provider.load(0)
    feature_channels = int(probe.features.shape[-1])

    if args.mode == "data":
        anchor = _module(args, 1, feature_channels).to(context.device)
        anchor_fit = fit_batched_direct_particles(
            anchor,
            family_ids=[family.family_id for family in context.train_families],
            load_family=provider.load,
            family_particle_weights=torch.ones(450, 1),
            epochs=args.anchor_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            shuffle_seed=args.seed,
            objective_audit_interval=args.objective_audit_interval,
            prefetch_family=provider.prefetch,
        )
        direct = _module(args, draws, feature_channels).to(context.device)
        _copy_anchor(direct, anchor)
        prior_callback = None
        anchor_description = "one shared uniform-weight anchor"
    else:
        direct = _module(args, draws, feature_channels).to(context.device)
        anchor_fit = fit_batched_direct_particles(
            direct,
            family_ids=[family.family_id for family in context.train_families],
            load_family=provider.load,
            family_particle_weights=torch.ones(450, draws),
            prior_slice=provider.prior,
            epochs=args.anchor_epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            shuffle_seed=args.seed,
            objective_audit_interval=args.objective_audit_interval,
            prefetch_family=provider.prefetch,
        )
        prior_callback = provider.prior
        anchor_description = "per-draw uniform anchors with indexed prior retained"
    anchor_template = subset_direct_module(direct, range(draws))
    anchor_state = {key: value.detach().cpu() for key, value in direct.state_dict().items()}
    fit = fit_batched_direct_particles(
        direct,
        family_ids=[family.family_id for family in context.train_families],
        load_family=provider.load,
        family_particle_weights=bootstrap_weights,
        prior_slice=prior_callback,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        shuffle_seed=args.seed + 1,
        objective_audit_interval=args.objective_audit_interval,
        prefetch_family=provider.prefetch,
    )
    multistart = _multistart_spot_check(
        context=context,
        direct=direct,
        anchor_template=anchor_template,
        z_e=z_e,
        bootstrap_weights=bootstrap_weights,
        mode=args.mode,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        objective_audit_interval=args.objective_audit_interval,
        prior_chunk=args.prior_chunk,
        spot_draws=spot_draws,
        perturbation=perturbation,
        seed=args.seed + 100_003,
    )
    multistart["pass"] = bool(
        multistart["relative_objective_gap"]
        <= protocol["direct"]["multistart_relative_objective_tolerance"]
        and multistart["alternative_gradient_reduction_ratio"]
        <= protocol["direct"]["maximum_gradient_reduction_ratio"]
    )
    multistart["relative_objective_tolerance"] = protocol["direct"][
        "multistart_relative_objective_tolerance"
    ]
    torch.save(
        {
            "state_dict": direct.state_dict(),
            "anchor_state_dict": anchor_state,
            "z_e": z_e.detach().cpu(),
            "family_particle_weights": bootstrap_weights,
            "plan": plan,
            "anchor_fit": anchor_fit.__dict__,
            "fit": fit.__dict__,
            "multistart": multistart,
        },
        output_dir / "direct_particles.pt",
    )

    particle_means = []
    anchor_particle_means = []
    score_weights = []
    crps_rows: list[float] = []
    rmse_rows: list[float] = []
    impact_area: list[float] = []
    impact_peak: list[float] = []
    impact_arrival: list[float] = []
    representability: list[float] = []
    for index, family in enumerate(context.val_families):
        frozen = provider.load(index, validation=True)
        with torch.no_grad():
            correction = direct(frozen.features)
            nested = frozen.base_prediction.unsqueeze(0) + correction
            anchor_nested = (
                frozen.base_prediction.unsqueeze(0) + anchor_template(frozen.features)
            )
            prior = None
            if args.mode == "rpf":
                prior = provider.prior(index, validation=True)
                nested = nested + prior
                anchor_nested = anchor_nested + prior
                if args.head == "last":
                    projection = prior_linear_representability_error(
                        frozen.features,
                        prior,
                        weights=frozen.score_weights,
                    )
                    representability.extend(
                        float(value)
                        for value in projection[
                            "relative_squared_error_per_particle"
                        ]
                    )
            nested_physical = inverse_predictions(context, nested)
            anchor_nested_physical = inverse_predictions(context, anchor_nested)
            reference = frozen.reference.to(nested.device, nested.dtype)
            reference_physical = inverse_reference(context, reference)
            scorer = (
                fixed_support_fair_crps_members
                if sampling_design == "fixed_support"
                else crossed_fair_crps_members
            )
            crps = scorer(
                nested_physical.unsqueeze(0),
                reference_physical.unsqueeze(0),
                weights=frozen.score_weights,
            )
            flat = nested_physical.reshape(1, draws * nested.shape[1], *nested.shape[2:])
            rmse = base_rmse_from_reference(
                flat,
                reference_physical.unsqueeze(0),
                weights=frozen.score_weights,
            )
        crps_rows.append(float(crps))
        rmse_rows.append(float(rmse))
        means = nested_physical.mean(dim=1).detach().cpu().to(torch.float16)
        anchor_means = (
            anchor_nested_physical.mean(dim=1).detach().cpu().to(torch.float16)
        )
        particle_means.append(means)
        anchor_particle_means.append(anchor_means)
        score_weights.append(None if frozen.score_weights is None else frozen.score_weights.cpu())

        area = (
            torch.ones(nested.shape[-2])
            if frozen.score_weights is None
            else frozen.score_weights[0, :, 0].detach().cpu()
        )
        static_raw = torch.stack([torch.zeros_like(area), area], dim=1).numpy()
        wettable = area.numpy() > 0.0
        impact = compute_nested_flood_impact_crps_metrics(
            nested_physical[..., 0].detach().cpu().numpy(),
            reference_physical[..., 0].detach().cpu().numpy(),
            family.geometry[0].detach().cpu().numpy(),
            static_raw,
            wettable,
            {"inundation_threshold_m": 0.1},
            sampling_design=(
                "fixed_epistemic_support_common_random_numbers"
                if sampling_design == "fixed_support"
                else "crossed_common_random_numbers"
            ),
        )
        impact_area.append(float(np.mean(impact["crps_total_inundated_area_km2"])))
        impact_peak.append(float(np.mean(impact["crps_peak_inundated_area_km2"])))
        impact_arrival.append(float(impact["crps_arrival_time_step"]))
        torch.save(
            {
                "family_id": family.family_id,
                "particle_mean_physical_m": means,
                "uniform_anchor_particle_mean_physical_m": anchor_means,
                "weight_induced_particle_displacement_physical_m": means - anchor_means,
            },
            output_dir / f"val_{family.family_id}.pt",
        )

    scale = direct_particle_mean_scale_interval(
        particle_means,
        score_weights=score_weights,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    anchor_scale = direct_particle_mean_scale_interval(
        anchor_particle_means,
        score_weights=score_weights,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    weight_induced_displacement = matched_particle_displacement_interval(
        particle_means,
        anchor_particle_means,
        score_weights=score_weights,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 2,
    )
    report = {
        **plan,
        "dirichlet_particle_seed": (
            int(context.stage2_metadata["dirichlet_particle_control"]["seed"])
            if args.law == "dirichlet"
            else None
        ),
        "anchor": anchor_description,
        "anchor_initial_objective": anchor_fit.initial_objective,
        "anchor_final_objective": anchor_fit.final_objective,
        "anchor_initial_gradient_norm": anchor_fit.initial_gradient_norm,
        "anchor_final_gradient_norm": anchor_fit.final_gradient_norm,
        "anchor_gradient_reduction_ratio": anchor_fit.gradient_reduction_ratio,
        "direct_initial_objective": fit.initial_objective,
        "direct_final_objective": fit.final_objective,
        "direct_initial_gradient_norm": fit.initial_gradient_norm,
        "direct_final_gradient_norm": fit.final_gradient_norm,
        "direct_gradient_reduction_ratio": fit.gradient_reduction_ratio,
        "direct_best_epoch": fit.best_epoch,
        "multistart": multistart,
        "optimization_valid": bool(
            np.isfinite(anchor_fit.final_objective)
            and np.isfinite(fit.final_objective)
            and anchor_fit.final_objective <= anchor_fit.initial_objective + 1.0e-8
            and fit.final_objective <= fit.initial_objective + 1.0e-8
            and anchor_fit.gradient_reduction_ratio
            <= protocol["direct"]["maximum_gradient_reduction_ratio"]
            and fit.gradient_reduction_ratio
            <= protocol["direct"]["maximum_gradient_reduction_ratio"]
            and multistart["pass"]
        ),
        "direct_scale_m": scale,
        "uniform_anchor_scale_m": anchor_scale,
        "weight_induced_displacement_rms_m": weight_induced_displacement,
        "direct_scale_interpretation": (
            "total prior-inclusive spread after weighted re-optimization"
            if args.mode == "rpf"
            else "data-bootstrap spread after weighted re-optimization"
        ),
        "validation": {
            "fair_crps_m": _mean(crps_rows),
            "rmse_m": _mean(rmse_rows),
            "impact_total_area_crps_km2": _mean(impact_area),
            "impact_peak_area_crps_km2": _mean(impact_peak),
            "arrival_time_crps_steps": _mean(impact_arrival),
        },
        "prior_linear_representability_relative_error_mean": _mean(representability),
        "prior_linear_representability_particle_family_count": len(representability),
        "prior_linear_representability_aggregation": (
            "mean over every validation-family and epistemic-particle prior slice"
        ),
        "representability_interpretation": (
            "last-layer RPF retention has a construction floor; full-head sensitivity is required"
            if args.head == "last" and args.mode == "rpf"
            else "not_applicable"
        ),
    }
    write_checksummed_artifact(output_dir / "RESULT.json", report)
    (output_dir / "COMPLETE").write_text("complete\n", encoding="utf-8")
    LOG.info("direct Phase-5 result: %s", report)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
