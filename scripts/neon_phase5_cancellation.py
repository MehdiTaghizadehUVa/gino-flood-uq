#!/usr/bin/env python3
"""Whitened cancellation, observed scale, and skill attribution for Phase-5 D3."""

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
    hermite_basis_covariance,
    phase5_predeclared_protocol,
    stratified_whitened_cancellation,
    verify_checksummed_artifact,
    write_checksummed_artifact,
)
from neuralop.flood.neon import (
    base_rmse_from_reference,
    crossed_fair_crps_members,
    fair_crps_members,
)
from neon_phase5_runtime import (
    collect_cached_family,
    inverse_predictions,
    inverse_reference,
    load_phase5_context,
    write_provenance,
)


LOG = logging.getLogger("neon_phase5_cancellation")


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
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--epistemic-chunk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _average_nested(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for record in records for key in record})
    return {
        key: float(np.nanmean([record.get(key, np.nan) for record in records]))
        for key in keys
    }


def _skill(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[float, float]:
    crps = crossed_fair_crps_members(
        prediction.unsqueeze(0), reference.unsqueeze(0), weights=weights
    )
    flat = prediction.reshape(1, prediction.shape[0] * prediction.shape[1], *prediction.shape[2:])
    rmse = base_rmse_from_reference(flat, reference.unsqueeze(0), weights=weights)
    return float(crps), float(rmse)


def main() -> int:
    args = parse_args()
    protocol_sha = verify_checksummed_artifact(args.protocol)
    protocol = phase5_predeclared_protocol()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "neon_phase5_cancellation_plan_v1",
        "checkpoint_state": "B3 selected checkpoint",
        "validation_families": 50,
        "draws": int(args.draws),
        "sampling_design": "crossed_shared_aleatory_bank",
        "physical_space_skill_and_scale": True,
        "skill_variants": ["base", "base_plus_prior", "base_plus_trainable", "base_plus_both"],
        "lead_bins": protocol["lead_strata"],
        "wetness_strata_m": protocol["wetness_strata_m"],
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
    module = context.stage2
    if module.epistemic_basis_module is None:
        raise ValueError("D3 requires the B3 centered-Hermite checkpoint.")
    generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
    z_e = torch.randn(args.draws, module.epistemic_dim, generator=generator).to(context.device)
    basis_values = module.epistemic_basis_module(z_e).double()
    basis_covariance = hermite_basis_covariance(
        module.epistemic_dim,
        module.epistemic_basis_module.quadratic_vectors,
        linear_terms=module.epistemic_basis_module.linear_terms,
    ).to(context.device)
    lead_bins = {key: tuple(value) for key, value in protocol["lead_strata"].items()}

    strata_records: dict[str, list[dict[str, float]]] = {}
    lead_slopes = []
    total_grams = []
    prior_grams = []
    train_grams = []
    skill_records = {name: [] for name in plan["skill_variants"]}
    impact_records = {name: [] for name in plan["skill_variants"]}
    per_family = []
    for family_index, family in enumerate(context.val_families):
        frozen = collect_cached_family(context, family, k=8, bank=0)
        full_train = []
        full_prior = []
        deterministic = None
        with torch.no_grad():
            for start in range(0, args.draws, args.epistemic_chunk):
                out = module(
                    frozen.base_prediction,
                    frozen.features,
                    z_e[start : start + args.epistemic_chunk],
                    node_coords=family.geometry,
                    canonical_mean_features=frozen.canonical_mean_features,
                )
                full_train.append(out.trainable_correction[0])
                full_prior.append(float(module.alpha) * out.prior_correction[0])
                if deterministic is None:
                    deterministic = out.deterministic_correction[0, 0]
        assert deterministic is not None
        full_train_t = torch.cat(full_train, dim=0)
        full_prior_t = torch.cat(full_prior, dim=0)
        train_mean = full_train_t.mean(dim=1) * context.physical_scale_m
        prior_mean = full_prior_t.mean(dim=1) * context.physical_scale_m
        cancellation = stratified_whitened_cancellation(
            train_mean,
            prior_mean,
            basis_values,
            basis_covariance,
            reference_depth=inverse_reference(
                context, family.reference.to(context.device)
            ).mean(dim=0),
            wettable_mask=(
                torch.ones(family.reference.shape[2], dtype=torch.bool, device=context.device)
                if family.structural_dry_mask is None
                else ~family.structural_dry_mask.to(context.device)
            ),
            lead_bins=lead_bins,
            score_weights=family.weights,
        )
        for stratum, values in cancellation["strata"].items():
            strata_records.setdefault(stratum, []).append(values)
        lead_slopes.append(cancellation["cancellation_regression_slope_per_lead_step"])
        weights_cpu = None if family.weights is None else family.weights.cpu()
        total_grams.append(
            particle_mean_grams([train_mean.cpu() + prior_mean.cpu()], score_weights=[weights_cpu])[0]
        )
        prior_grams.append(
            particle_mean_grams([prior_mean.cpu()], score_weights=[weights_cpu])[0]
        )
        train_grams.append(
            particle_mean_grams([train_mean.cpu()], score_weights=[weights_cpu])[0]
        )

        base_norm = frozen.base_prediction[0] + deterministic
        base_nested = base_norm.unsqueeze(0).expand(args.draws, -1, -1, -1, -1)
        variants_norm = {
            "base": base_nested,
            "base_plus_prior": base_nested + full_prior_t,
            "base_plus_trainable": base_nested + full_train_t,
            "base_plus_both": base_nested + full_train_t + full_prior_t,
        }
        reference_physical = inverse_reference(
            context, family.reference.to(context.device)
        )
        area = (
            torch.ones(base_norm.shape[-2])
            if family.weights is None
            else family.weights[0, :, 0].cpu()
        )
        static_raw = torch.stack([torch.zeros_like(area), area], dim=1).numpy()
        wettable = area.numpy() > 0.0
        family_row = {"family_id": family.family_id}
        for name, nested_norm in variants_norm.items():
            nested = inverse_predictions(context, nested_norm)
            crps, rmse = _skill(nested, reference_physical, family.weights)
            skill_records[name].append({"fair_crps_m": crps, "rmse_m": rmse})
            impact = compute_nested_flood_impact_crps_metrics(
                nested[..., 0].detach().cpu().numpy(),
                reference_physical[..., 0].detach().cpu().numpy(),
                family.geometry[0].cpu().numpy(),
                static_raw,
                wettable,
                {"inundation_threshold_m": 0.1},
                sampling_design="crossed_common_random_numbers",
            )
            impact_row = {
                "total_area_crps_km2": float(np.mean(impact["crps_total_inundated_area_km2"])),
                "peak_area_crps_km2": float(np.mean(impact["crps_peak_inundated_area_km2"])),
                "arrival_crps_steps": float(impact["crps_arrival_time_step"]),
            }
            impact_records[name].append(impact_row)
            family_row[f"{name}_fair_crps_m"] = crps
            family_row[f"{name}_rmse_m"] = rmse
        per_family.append(family_row)
        LOG.info("D3 family %d/50 %s", family_index + 1, family.family_id)

    total_scale = gram_scale_interval(
        torch.stack(total_grams), replicates=args.bootstrap_replicates, seed=args.seed
    )
    prior_scale = gram_scale_interval(
        torch.stack(prior_grams), replicates=args.bootstrap_replicates, seed=args.seed + 1
    )
    train_scale = gram_scale_interval(
        torch.stack(train_grams), replicates=args.bootstrap_replicates, seed=args.seed + 2
    )
    report = {
        **plan,
        "observed_posterior_scale_m": total_scale,
        "scaled_prior_scale_m": prior_scale,
        "trainable_scale_m": train_scale,
        "retention_ratio_std": total_scale["estimate"] / max(prior_scale["estimate"], 1e-15),
        "strata": {
            key: _average_nested(records) for key, records in strata_records.items()
        },
        "cancellation_regression_slope_per_lead_step_mean": float(np.nanmean(lead_slopes)),
        "skill": {key: _average_nested(value) for key, value in skill_records.items()},
        "impact": {key: _average_nested(value) for key, value in impact_records.items()},
        "skill_without_cancellation": {
            "base_plus_prior": "explicitly scored",
            "base_plus_trainable": "explicitly scored",
            "base_plus_both": "explicitly scored",
        },
        "per_family": per_family,
    }
    write_checksummed_artifact(output / "RESULT.json", report)
    torch.save(
        {
            "z_e": z_e.cpu(),
            "basis_values": basis_values.cpu(),
            "basis_covariance": basis_covariance.cpu(),
            "total_grams": torch.stack(total_grams),
            "prior_grams": torch.stack(prior_grams),
            "train_grams": torch.stack(train_grams),
        },
        output / "CANCELLATION.pt",
    )
    (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
