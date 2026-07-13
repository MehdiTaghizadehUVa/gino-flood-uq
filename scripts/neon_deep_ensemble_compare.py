#!/usr/bin/env python3
"""Physical-space deep-ensemble cross-check for NEON epistemic maps."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage1-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", choices=("val", "train", "all"), default="val")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-families", type=int, default=50)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--m-eval", type=int, default=16)
    parser.add_argument("--k-neon", type=int, default=50)
    parser.add_argument("--k-de", type=int, default=50)
    parser.add_argument("--k-chunk", type=int, default=16)
    parser.add_argument("--epistemic-chunk", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-q", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_plan(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.k_neon) != int(args.k_de):
        raise ValueError(
            "NEON and deep-ensemble comparison require identical aleatory bank sizes "
            "because the stable latent-bank seed includes K."
        )
    return {
        "config": str(args.config),
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "stage1_bundle": str(args.stage1_bundle),
        "output_dir": str(args.output_dir),
        "families": str(args.families),
        "val_fraction": float(args.val_fraction),
        "max_families": int(args.max_families),
        "rollout_length": (
            None if args.rollout_length is None else int(args.rollout_length)
        ),
        "m_eval": int(args.m_eval),
        "k_neon": int(args.k_neon),
        "k_de": int(args.k_de),
        "k_chunk": int(args.k_chunk),
        "epistemic_chunk": int(args.epistemic_chunk),
        "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
        "seed": int(args.seed),
        "top_q": float(args.top_q),
        "common_aleatory_latent_bank": True,
        "physical_space": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _corr(x, y) -> float:
    import torch

    x = x.reshape(-1).to(torch.float64)
    y = y.reshape(-1).to(torch.float64)
    xm, ym = x - x.mean(), y - y.mean()
    denominator = torch.sqrt(xm.pow(2).sum() * ym.pow(2).sum()).clamp_min(1.0e-12)
    return float((xm * ym).sum() / denominator)


def _active_values(value, weights):
    import torch

    if weights is None:
        return value.reshape(-1)
    mask = weights.to(device=value.device) > 0
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    mask = torch.broadcast_to(mask, value.shape)
    return value[mask]


def _weighted_mean(value, weights) -> float:
    import torch

    if weights is None:
        return float(value.mean())
    weight = weights.to(device=value.device, dtype=value.dtype)
    if weight.ndim == 3:
        weight = weight.unsqueeze(0)
    weight = torch.broadcast_to(weight, value.shape)
    return float((value * weight).sum() / weight.sum().clamp_min(1.0e-12))


def ensemble_mean_absolute_error(prediction, reference, *, ensemble_dims):
    """Absolute error of one method's own ensemble mean against the reference mean."""

    forecast_mean = prediction.mean(dim=tuple(int(dim) for dim in ensemble_dims))
    reference_mean = reference.mean(dim=1)
    if forecast_mean.shape != reference_mean.shape:
        raise ValueError(
            "forecast and reference means must share [B, T, Nv, C] shape; "
            f"got {tuple(forecast_mean.shape)} and {tuple(reference_mean.shape)}."
        )
    return (forecast_mean - reference_mean).abs()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    plan = resolve_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    import torch

    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.eval.neon import (
        compare_epistemic_maps,
        deep_ensemble_epistemic_variance,
        inverse_transform_on_tensor_device,
    )
    from neuralop.flood.neon import (
        anova_corrected_epistemic_variance,
        load_neon_stage2_checkpoint,
        sample_epistemic_indices,
    )
    from neuralop.flood.train.neon import neon_stage2_eval_forward_chunked
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import (
        make_cached_feature_collector,
        make_feature_collector_from_frozen_model,
    )
    from neuralop.flood.utils.runtime_core import (
        load_config_and_setup,
        parse_target_variables,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("neon_de_compare")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan["stage2_checkpoint_sha256"] = _sha256(Path(args.stage2_checkpoint))
    plan["stage1_bundle_sha256"] = _sha256(Path(args.stage1_bundle))

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_de_compare", "--config_path", str(args.config)]
        config, _device, _logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(
        getattr(config.data, "target_variables", ["wd"])
    )
    stage1 = _load_frozen_stage1(args.stage1_bundle)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared  # type: ignore[attr-defined]
    normalizers = prepared.get("normalizers") or {}
    target_normalizer = normalizers.get("target")
    reference_normalizer = normalizers.get("dynamic")
    if target_normalizer is None or reference_normalizer is None:
        raise RuntimeError("deep-ensemble comparison requires saved physical normalizers.")
    dry_mask = prepared.get("structural_dry_mask")
    models = list(prepared.get("models") or [])
    if len(models) < 2:
        raise ValueError("deep-ensemble comparison requires at least two Stage-1 models.")

    train_families, val_families = build_families_from_config(
        config,
        normalizers,
        target_variables,
        log,
        structural_dry_artifact=(
            None if dry_mask is None else {"dry_mask": dry_mask}
        ),
        rollout_length=args.rollout_length,
        val_fraction=float(args.val_fraction),
        dataset_split="test",
    )
    if args.families == "val":
        families = val_families
    elif args.families == "train":
        families = train_families
    else:
        families = train_families + val_families
    families = sorted(families, key=lambda family: family.family_id)[
        : int(args.max_families)
    ]
    if not families:
        raise ValueError("deep-ensemble comparison selected no families.")

    module, metadata = load_neon_stage2_checkpoint(
        args.stage2_checkpoint, map_location="cpu"
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()
    common_seed = int(args.seed) + 100
    neon_collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=metadata.get(
            "feature_source", "decoder_pre_projection"
        ),
        n_history=int(bundle.n_history),
        latent_dim=int(bundle.fgn_noise_dim),
        generator=torch.Generator().manual_seed(common_seed),
        canonical_k=(
            int(metadata.get("deterministic_head_canonical_k", 32))
            if getattr(module, "deterministic_head_enabled", False)
            else 0
        ),
        canonical_seed=int(metadata.get("deterministic_head_latent_seed", 123)),
        canonical_zero_latent=(
            str(metadata.get("deterministic_head_feature", ""))
            == "fixed_zero_latent"
        ),
        target_normalizer=target_normalizer,
    )
    if args.cache_dir:
        neon_collector = make_cached_feature_collector(
            neon_collector, cache_dir=Path(args.cache_dir) / "neon"
        )
    de_collectors = []
    for model_index, model in enumerate(models):
        collector = make_feature_collector_from_frozen_model(
            model,
            feature_source="decoder_pre_projection",
            n_history=int(bundle.n_history),
            latent_dim=int(bundle.fgn_noise_dim),
            # Separate generators with identical state provide a common latent
            # bank without coupling mutable generator state across models.
            generator=torch.Generator().manual_seed(common_seed),
            target_normalizer=target_normalizer,
        )
        if args.cache_dir:
            collector = make_cached_feature_collector(
                collector,
                cache_dir=Path(args.cache_dir) / f"deep_model_{model_index}",
            )
        de_collectors.append(collector)

    z_generator = torch.Generator().manual_seed(int(args.seed) + 1)
    rows = []
    for family_index, family in enumerate(families):
        batch = neon_collector(family, num_aleatory=int(args.k_neon), latent_bank_id=0)
        z_e = sample_epistemic_indices(
            int(args.m_eval),
            module.epistemic_dim,
            device=batch.features.device,
            dtype=batch.features.dtype,
            generator=z_generator,
        )
        nested_normalized = neon_stage2_eval_forward_chunked(
            module=module,
            base_prediction=batch.base_prediction,
            features=batch.features,
            z_e=z_e,
            k_chunk=int(args.k_chunk),
            epistemic_chunk_size=int(args.epistemic_chunk),
            node_coords=family.geometry,
            canonical_mean_features=batch.canonical_mean_features,
            output_device="cpu",
            output_dtype=torch.float32,
        )
        nested_physical = inverse_transform_on_tensor_device(
            target_normalizer, nested_normalized
        )
        neon_epistemic = anova_corrected_epistemic_variance(nested_physical)

        model_means = []
        all_model_members = []
        for collector in de_collectors:
            de_batch = collector(
                family, num_aleatory=int(args.k_de), latent_bank_id=0
            )
            physical = inverse_transform_on_tensor_device(
                target_normalizer,
                de_batch.base_prediction.detach().to("cpu", torch.float32),
            )
            model_means.append(physical.mean(dim=1))
            all_model_members.append(physical)
        deep_epistemic = deep_ensemble_epistemic_variance(
            torch.stack(model_means, dim=1)
        )
        reference = inverse_transform_on_tensor_device(
            reference_normalizer,
            family.reference.unsqueeze(0).to(torch.float32),
        )
        neon_absolute_error = ensemble_mean_absolute_error(
            nested_physical, reference, ensemble_dims=(1, 2)
        )
        deep_absolute_error = ensemble_mean_absolute_error(
            torch.cat(all_model_members, dim=1), reference, ensemble_dims=(1,)
        )
        active_neon = _active_values(neon_epistemic, family.weights)
        active_deep = _active_values(deep_epistemic, family.weights)
        active_neon_error = _active_values(neon_absolute_error, family.weights)
        active_deep_error = _active_values(deep_absolute_error, family.weights)
        comparison = compare_epistemic_maps(
            active_neon, active_deep, top_q=float(args.top_q)
        )
        rows.append(
            {
                "family_id": family.family_id,
                **comparison,
                "neon_epistemic_abs_error_corr": _corr(
                    active_neon.sqrt(), active_neon_error
                ),
                "deep_epistemic_abs_error_corr": _corr(
                    active_deep.sqrt(), active_deep_error
                ),
                "neon_epistemic_variance_mean_m2": _weighted_mean(
                    neon_epistemic, family.weights
                ),
                "deep_epistemic_variance_mean_m2": _weighted_mean(
                    deep_epistemic, family.weights
                ),
            }
        )
        log.info(
            "family %s complete (%d/%d)",
            family.family_id,
            family_index + 1,
            len(families),
        )

    numeric_keys = [key for key in rows[0] if key != "family_id"]
    aggregate = {
        key: float(torch.tensor([row[key] for row in rows], dtype=torch.float64).mean())
        for key in numeric_keys
    }
    payload = {
        "schema_version": "neon_deep_ensemble_comparison_v2",
        "plan": plan,
        "checkpoint_metadata": metadata,
        "j_models": len(models),
        "aggregate": aggregate,
        "per_family": rows,
    }
    output = out_dir / "deep_ensemble_comparison.json"
    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(output)
    print("NEON DEEP ENSEMBLE COMPARE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
