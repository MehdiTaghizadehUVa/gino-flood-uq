#!/usr/bin/env python3
"""Evaluate Phase-5 pilots with their design-correct nested scores."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuralop.flood.eval.impact_metrics import compute_nested_flood_impact_crps_metrics
from neuralop.flood.eval.neon_direct import gram_scale_interval, particle_mean_grams
from neuralop.flood.eval.neon_phase5 import (
    build_stage2_variant_predictions,
    hermite_basis_covariance,
    paired_family_noninferiority,
    particle_risk_differentiation,
    phase5_predeclared_protocol,
    stratified_whitened_cancellation,
    verify_checksummed_artifact,
    write_rung_attribution_csv,
    write_checksummed_artifact,
)
from neuralop.flood.neon import (
    PersistentDirichletParticleControl,
    base_rmse_from_reference,
    cancellation_diagnostics,
    crossed_fair_crps_members,
    epistemic_bootstrap_weights,
    fixed_support_fair_crps_members,
    project_bootstrap_epistemic_indices,
    sample_epistemic_indices,
)
from neuralop.flood.eval.neon import (
    crossed_sampling_design,
    domain_average_variance_summary,
    fixed_support_sampling_design,
    neon_epistemic_error_correlation,
)
from neon_phase5_runtime import (
    collect_cached_family,
    inverse_predictions,
    inverse_reference,
    load_phase5_context,
    write_provenance,
)


LOG = logging.getLogger("neon_phase5_p1a_eval")


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
        "geometry_tensors",
        "output_dir",
        "expected_head",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    parser.add_argument("--epistemic-chunk", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--index-seed", type=int, default=271828)
    parser.add_argument("--epistemic-particles", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _nested_skill(
    nested: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    sampling_design: str,
) -> tuple[float, float]:
    scorer = (
        fixed_support_fair_crps_members
        if sampling_design == "fixed_epistemic_support_common_random_numbers"
        else crossed_fair_crps_members
    )
    crps = scorer(
        nested.unsqueeze(0), reference.unsqueeze(0), weights=weights
    )
    flat = nested.reshape(1, nested.shape[0] * nested.shape[1], *nested.shape[2:])
    rmse = base_rmse_from_reference(flat, reference.unsqueeze(0), weights=weights)
    return float(crps), float(rmse)


def _mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> int:
    args = parse_args()
    protocol_sha = verify_checksummed_artifact(args.protocol)
    protocol = phase5_predeclared_protocol()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "neon_phase5_p1a_eval_plan_v1",
        "sampling_design": "auto_from_checkpoint",
        "physical_space": True,
        "validation_families": 50,
        "primary_threshold_m": 0.1,
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
    raw_control = context.stage2_metadata.get("dirichlet_particle_control")
    control = (
        None
        if not raw_control
        else PersistentDirichletParticleControl.from_metadata(raw_control)
    )
    if control is not None:
        z_e = control.eval_epistemic_indices(device=context.device, dtype=torch.float32)
        bootstrap_index = z_e
        M = control.num_particles
        sampling_design = "fixed_epistemic_support_common_random_numbers"
        support_seed = int(control.seed)
    else:
        M = int(args.epistemic_particles)
        if M < 2:
            raise ValueError("continuous pilot evaluation requires at least two particles.")
        bootstrap_dim = int(
            context.stage2_metadata.get("bootstrap_index_dim")
            or context.stage2.epistemic_dim
        )
        index_generator = torch.Generator(device="cpu").manual_seed(int(args.index_seed))
        bootstrap_index = sample_epistemic_indices(
            M,
            bootstrap_dim,
            device=context.device,
            generator=index_generator,
        )
        z_e = project_bootstrap_epistemic_indices(
            bootstrap_index,
            model_dim=int(context.stage2.epistemic_dim),
            projection=context.stage2_metadata.get("bootstrap_model_projection"),
        )
        sampling_design = "crossed_common_random_numbers"
        support_seed = None
    module = context.stage2
    if module.epistemic_basis_module is not None:
        basis_values = module.epistemic_basis_module(z_e).double()
        basis_covariance = hermite_basis_covariance(
            module.epistemic_dim,
            module.epistemic_basis_module.quadratic_vectors,
            linear_terms=module.epistemic_basis_module.linear_terms,
        ).to(context.device)
    else:
        basis_values = z_e.double()
        centered_basis = basis_values - basis_values.mean(dim=0, keepdim=True)
        if control is None:
            basis_covariance = torch.eye(
                basis_values.shape[1], device=context.device, dtype=torch.float64
            )
        else:
            basis_covariance = centered_basis.T @ centered_basis / float(M)
    lead_bins = {key: tuple(value) for key, value in protocol["lead_strata"].items()}
    skill = {
        key: []
        for key in ("base", "deterministic", "prior", "trainable", "both")
    }
    impact = {key: [] for key in skill}
    grams = []
    cancellation = []
    stratified_cancellation: dict[str, list[dict[str, float]]] = {}
    cancellation_lead_slopes: list[float] = []
    per_family = []
    association_rows: list[dict[str, float]] = []
    for index, family in enumerate(context.val_families):
        frozen = collect_cached_family(context, family, k=8, bank=0)
        train_chunks = []
        prior_chunks = []
        deterministic = None
        with torch.no_grad():
            for start in range(0, M, args.epistemic_chunk):
                out = module(
                    frozen.base_prediction,
                    frozen.features,
                    z_e[start : start + args.epistemic_chunk],
                    node_coords=family.geometry,
                    canonical_mean_features=frozen.canonical_mean_features,
                )
                train_chunks.append(out.trainable_correction[0])
                prior_chunks.append(out.prior_correction[0])
                if deterministic is None:
                    deterministic = out.deterministic_correction[0, 0]
        assert deterministic is not None
        train = torch.cat(train_chunks, dim=0)
        prior_raw = torch.cat(prior_chunks, dim=0)
        prior = float(module.alpha) * prior_raw
        # Cancellation is a cross-particle property. Compute it once over the
        # complete persistent support; chunk-level values depend on the chosen
        # evaluation chunk and are not scientifically interpretable.
        cancellation.append(
            cancellation_diagnostics(
                trainable_correction=train,
                prior_correction=prior_raw,
                alpha=module.alpha,
            )
        )
        train_mean_m = train.mean(dim=1) * context.physical_scale_m
        prior_mean_m = prior.mean(dim=1) * context.physical_scale_m
        stratified = stratified_whitened_cancellation(
            train_mean_m,
            prior_mean_m,
            basis_values,
            basis_covariance,
            reference_depth=inverse_reference(
                context, family.reference.to(context.device)
            ).mean(dim=0),
            wettable_mask=(
                torch.ones(
                    family.reference.shape[2],
                    dtype=torch.bool,
                    device=context.device,
                )
                if family.structural_dry_mask is None
                else ~family.structural_dry_mask.to(context.device)
            ),
            lead_bins=lead_bins,
            score_weights=family.weights,
        )
        for stratum, values in stratified["strata"].items():
            stratified_cancellation.setdefault(stratum, []).append(values)
        cancellation_lead_slopes.append(
            float(stratified["cancellation_regression_slope_per_lead_step"])
        )
        variants = build_stage2_variant_predictions(
            frozen.base_prediction[0], deterministic, train, prior
        )
        base = variants["base"]
        reference = inverse_reference(context, family.reference.to(context.device))
        rows = {"family_id": family.family_id}
        area = (
            torch.ones(base.shape[-2])
            if family.weights is None
            else family.weights[0, :, 0].cpu()
        )
        static_raw = torch.stack([torch.zeros_like(area), area], dim=1).numpy()
        wettable = area.numpy() > 0.0
        for name, normalized in variants.items():
            physical = inverse_predictions(context, normalized)
            crps, rmse = _nested_skill(
                physical,
                reference,
                family.weights,
                sampling_design=sampling_design,
            )
            skill[name].append({"fair_crps_m": crps, "rmse_m": rmse})
            result = compute_nested_flood_impact_crps_metrics(
                physical[..., 0].cpu().numpy(),
                reference[..., 0].cpu().numpy(),
                family.geometry[0].cpu().numpy(),
                static_raw,
                wettable,
                {"inundation_threshold_m": 0.1},
                sampling_design=sampling_design,
            )
            impact[name].append(
                {
                    "area_crps_km2": float(np.mean(result["crps_total_inundated_area_km2"])),
                    "peak_crps_km2": float(np.mean(result["crps_peak_inundated_area_km2"])),
                    "arrival_crps_steps": float(result["crps_arrival_time_step"]),
                }
            )
            rows[f"{name}_fair_crps_m"] = crps
            rows[f"{name}_rmse_m"] = rmse
        design_factory = (
            fixed_support_sampling_design if control is not None else crossed_sampling_design
        )
        design = design_factory(
            frozen.aleatory_latents,
            m=M,
            bank_id=f"{family.family_id}:phase5-pilot-bank0:k8",
            state_update_mode="member_feedback_persistent_latent",
        )
        both_physical = inverse_predictions(context, variants["both"]).unsqueeze(0)
        variance_summary = domain_average_variance_summary(
            both_physical,
            weights=family.weights,
            sampling_design=design,
        )
        rows.update(
            {
                # Match neon_ood_ranking_analysis.py's per-family input schema.
                # These are physical-space values under the same crossed/fixed
                # design used for the pilot's depth and impact scores.
                "ensemble_mean_rmse": rows["both_rmse_m"],
                "variance_epistemic_anova_corrected_mean": float(
                    variance_summary["variance_epistemic_anova_corrected"][0]
                ),
            }
        )
        association_rows.append(
            neon_epistemic_error_correlation(
                both_physical,
                reference.unsqueeze(0),
                weights=family.weights,
                sampling_design=design,
            )
        )
        both_means = both_physical[0].mean(dim=1).cpu()
        grams.append(
            particle_mean_grams(
                [both_means],
                score_weights=[None if family.weights is None else family.weights.cpu()],
            )[0]
        )
        per_family.append(rows)
        LOG.info("P1a eval family %d/50 %s", index + 1, family.family_id)

    posterior_scale = gram_scale_interval(
        torch.stack(grams), replicates=args.bootstrap_replicates, seed=args.seed
    )
    crps_margin = float(protocol["noninferiority"]["crps_margin_m"])
    def noninferiority_for(name: str, *, seed: int) -> dict[str, Any]:
        return paired_family_noninferiority(
            variant_crps=[row["fair_crps_m"] for row in skill[name]],
            base_crps=[row["fair_crps_m"] for row in skill["base"]],
            variant_rmse=[row["rmse_m"] for row in skill[name]],
            base_rmse=[row["rmse_m"] for row in skill["base"]],
            variant_impact=[row["area_crps_km2"] for row in impact[name]],
            base_impact=[row["area_crps_km2"] for row in impact["base"]],
            crps_margin=crps_margin,
            rmse_margin=float(protocol["noninferiority"]["rmse_margin_m"]),
            impact_margin=float(
                protocol["noninferiority"]["impact_area_crps_margin_km2"]
            ),
            replicates=args.bootstrap_replicates,
            seed=seed,
        )

    combined_noninferiority = noninferiority_for("both", seed=args.seed)
    noncancellation_noninferiority = {
        "prior": noninferiority_for("prior", seed=args.seed + 101),
        "trainable": noninferiority_for("trainable", seed=args.seed + 202),
    }
    skill_without_cancellation = any(
        bool(result["pass"]) for result in noncancellation_noninferiority.values()
    )

    geometry = torch.load(args.geometry_tensors, map_location="cpu")
    gradients = geometry["per_family_gradients"].double()
    train_ids = [family.family_id for family in context.train_families]
    if control is not None:
        family_weights = control.family_particle_weights(
            train_ids, torch.arange(M), dtype=torch.float64
        )
    else:
        bootstrap = dict(context.stage2_metadata.get("bootstrap") or {})
        family_weights = epistemic_bootstrap_weights(
            train_ids,
            bootstrap_index.detach().cpu().double(),
            seed=int(bootstrap.get("seed", 0)),
            distribution=str(bootstrap.get("distribution", "probit_exponential")),
            temperature=float(bootstrap.get("temperature", 0.5)),
            normalize=str(bootstrap.get("normalize", "per_epistemic_batch")),
            min_weight=float(bootstrap.get("min_weight", 0.05)),
            max_weight=float(bootstrap.get("max_weight", 5.0)),
        )
    delta = (family_weights - 1.0).T @ gradients / float(gradients.shape[0])
    noise = geometry["minibatch_noise_gradients"].double()
    gradient_snr = float(
        delta.norm(dim=1).square().mean().sqrt()
        / noise.norm(dim=1).square().mean().sqrt()
    )
    if "per_family_anchor_scores" not in geometry:
        raise ValueError(
            "P1a risk differentiation requires D2 per_family_anchor_scores; "
            "regenerate the Phase-5 geometry artifact."
        )
    risk_differentiation = particle_risk_differentiation(
        geometry["per_family_anchor_scores"],
        family_weights,
        minibatch_size=int(geometry.get("noise_batch_size", 8)),
        replicates=args.bootstrap_replicates,
        seed=args.seed + 303,
    )
    # A claim of resolved particle differentiation requires both the score
    # gradient and scalar empirical risk to clear their matched noise floors.
    joint_snr = min(gradient_snr, float(risk_differentiation["signal_to_noise"]))
    curvature = geometry["curvature_proxy"].double()
    displacement = -torch.linalg.solve(
        curvature + 1.0e-4 * torch.eye(curvature.shape[0]), delta.T
    ).T
    jacobian = geometry["validation_jacobian_physical"].double()
    functional_displacement = float((jacobian @ displacement.T).square().mean().sqrt())
    cancel_mean = _mean_rows(
        [
            {
                "cancellation_fraction": float(row["cancellation_fraction"]),
                "train_prior_cosine": float(row["train_prior_cosine"]),
            }
            for row in cancellation
        ]
    )
    report = {
        **plan,
        "sampling_design": sampling_design,
        "epistemic_index_mode": (
            "dirichlet_particles" if control is not None else "continuous"
        ),
        "dirichlet_particle_seed": support_seed,
        "dirichlet_num_particles": (
            None if control is None else int(control.num_particles)
        ),
        "family_split_fingerprint": (
            None if control is None else str(control.split_fingerprint)
        ),
        "evaluation_index_seed": int(args.index_seed),
        "ladder_rung": str(context.run_metadata.get("ladder_rung", "")).upper(),
        "prior_seed": context.run_metadata.get("prior_seed"),
        "prior_scale_mode": context.stage2_metadata.get("prior_scale_mode"),
        "prior_scale_target_std_m": context.stage2_metadata.get(
            "prior_scale_target_std_m"
        ),
        "prior_scale_alpha_normalized": float(module.alpha),
        "training_validation_seed": context.run_metadata.get("val_seed"),
        "amplitude_tuned_on_evaluation_events": False,
        "amplitude_tuning_contract": (
            "training_fit_and_fixed_in_distribution_validation_only"
        ),
        "epistemic_particles": int(M),
        "posterior_scale_m": posterior_scale,
        "skill": {key: _mean_rows(value) for key, value in skill.items()},
        "impact": {key: _mean_rows(value) for key, value in impact.items()},
        "epistemic_error_association": _mean_rows(association_rows),
        "noninferiority": combined_noninferiority,
        "noncancellation_noninferiority": noncancellation_noninferiority,
        "differentiation": {
            "signal_to_noise": joint_snr,
            "joint_rule": "minimum_of_gradient_and_common_anchor_risk_snr",
            "gradient_signal_to_noise": gradient_snr,
            "risk_signal_to_noise": float(risk_differentiation["signal_to_noise"]),
            "risk": risk_differentiation,
            "functional_displacement_m": functional_displacement,
        },
        **cancel_mean,
        "stratified_cancellation": {
            "strata": {
                key: _mean_rows(rows)
                for key, rows in sorted(stratified_cancellation.items())
            },
            "cancellation_regression_slope_per_lead_step_mean": float(
                np.nanmean(cancellation_lead_slopes)
            ),
            "basis_mode": (
                "centered_hermite"
                if module.epistemic_basis_module is not None
                else "fixed_support_empirical"
                if control is not None
                else "identity_gaussian"
            ),
        },
        "skill_without_cancellation": bool(skill_without_cancellation),
        "per_family": per_family,
    }
    write_checksummed_artifact(output / "RESULT.json", report)
    write_rung_attribution_csv(output / "rung_attribution.csv", [report])
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
