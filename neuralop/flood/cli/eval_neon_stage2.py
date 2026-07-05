"""CLI entrypoint for NEON Stage-2 nested evaluation.

Evaluates a trained Stage-2 EpiNet checkpoint on grouped-hydrograph families
with the full nested budget (``M_eval`` epistemic x ``K_eval`` aleatory),
producing:

- nested predictive + epistemic metrics (``evaluate_neon_nested``),
- PIT / rank histograms and exceedance-reliability curves through the legacy
  evaluator's counting logic,
- spread-skill diagnostics,
- optional flood-impact CRPS metrics (inundated area / peak / arrival /
  pooled-radius) via ``compute_flood_impact_crps_metrics``,
- optional per-family nested forecast artifacts (common HDF5 schema with
  ``member_epistemic_id`` / ``member_aleatory_id``),
- variance maps (aleatory / epistemic / ANOVA-corrected / total) for the first
  family.

Usage::

    python -m neuralop.flood.cli.eval_neon_stage2 \
        --config <flood eval yaml> --stage2-checkpoint <neon_stage2_best.pt> \
        --stage1-bundle <coastal_fgn_bundle.json> --output-dir <dir> \
        [--families val|train|all] [--m-eval 32] [--k-eval 50] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="eval_neon_stage2",
        description="Nested evaluation of a NEON Stage-2 epistemic correction.",
    )
    parser.add_argument("--config", required=True, help="Flood eval YAML (grouped rollout data).")
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage1-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", choices=("val", "train", "all"), default="val",
                        help="Which side of the family split to evaluate (default: val).")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--m-eval", type=int, default=32)
    parser.add_argument("--k-eval", type=int, default=50)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--rollout-length", type=int, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=(0.1, 0.3, 0.5))
    parser.add_argument("--pit-min-ref-depth", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write-artifacts", action="store_true")
    parser.add_argument("--impact-metrics", action="store_true")
    parser.add_argument("--variance-maps", type=int, default=1,
                        help="Write variance maps for the first N families (default 1).")
    parser.add_argument("--cache-dir", default=None,
                        help="Disk cache for frozen K_eval rollouts (one .pt per family; "
                             "shared across checkpoint evaluations so the frozen model "
                             "is only rolled once per family).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved evaluation plan without loading torch/data.")
    return parser.parse_args(argv)


def resolve_eval_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Torch-free resolved plan for --dry-run and logging."""
    return {
        "config": str(args.config),
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "stage1_bundle": str(args.stage1_bundle),
        "output_dir": str(args.output_dir),
        "families": args.families,
        "val_fraction": float(args.val_fraction),
        "m_eval": int(args.m_eval),
        "k_eval": int(args.k_eval),
        "max_families": None if args.max_families is None else int(args.max_families),
        "rollout_length": None if args.rollout_length is None else int(args.rollout_length),
        "thresholds": [float(t) for t in args.thresholds],
        "pit_min_ref_depth": float(args.pit_min_ref_depth),
        "seed": int(args.seed),
        "write_artifacts": bool(args.write_artifacts),
        "impact_metrics": bool(args.impact_metrics),
        "variance_maps": int(args.variance_maps),
        "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    plan = resolve_eval_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    # ---- Heavy path: lazy imports keep --dry-run torch-free ----
    import logging

    import numpy as np
    import torch

    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.eval.datasets import _load_or_fit_normalizers
    from neuralop.flood.eval.neon import (
        evaluate_neon_nested,
        exceedance_reliability,
        nested_pit_rank_histograms,
        save_nested_forecast_artifact,
        spread_error_diagnostics,
        write_variance_maps,
    )
    from neuralop.flood.neon import (
        load_neon_stage2_checkpoint,
        sample_epistemic_indices,
    )
    from neuralop.flood.train.neon import neon_stage2_eval_forward
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import (
        make_cached_feature_collector,
        make_feature_collector_from_frozen_model,
    )
    from neuralop.flood.utils.runtime_core import load_config_and_setup, parse_target_variables

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("eval_neon_stage2")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("plan: %s", json.dumps(plan, sort_keys=True))

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["eval_neon_stage2", "--config_path", str(args.config)]
        flood_config, _dev, _is_logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(getattr(flood_config.data, "target_variables", ["wd"]))
    normalizers, _norm_path = _load_or_fit_normalizers(flood_config, None, None, log)

    train_fam, val_fam = build_families_from_config(
        flood_config, normalizers, target_variables, log,
        rollout_length=args.rollout_length,
        val_fraction=float(args.val_fraction),
    )
    families = {"val": val_fam, "train": train_fam, "all": train_fam + val_fam}[args.families]
    families = sorted(families, key=lambda f: f.family_id)
    if args.max_families is not None:
        families = families[: int(args.max_families)]
    log.info("evaluating %d families (%s split)", len(families), args.families)

    module, ckpt_meta = load_neon_stage2_checkpoint(args.stage2_checkpoint, map_location="cpu")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    module = module.to(device).eval()
    stage1 = _load_frozen_stage1(args.stage1_bundle)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared  # type: ignore[attr-defined]
    collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=ckpt_meta.get("feature_source", "decoder_pre_projection"),
        n_history=int(bundle.n_history),
        latent_dim=int(bundle.fgn_noise_dim),
        generator=torch.Generator().manual_seed(int(args.seed)),
    )
    if args.cache_dir:
        # Sharing the cache across checkpoint evaluations also fixes the K_eval
        # aleatory draws per family, so checkpoints are compared on identical
        # frozen ensembles rather than through resampling noise.
        collector = make_cached_feature_collector(collector, cache_dir=args.cache_dir)

    per_family: list[dict[str, Any]] = []
    pit_total: dict[str, Any] | None = None
    reliability_sums: dict[str, list[dict[str, float]]] = {}
    spread_corrs: list[float] = []
    impact_rows: list[dict[str, float]] = []
    z_gen = torch.Generator().manual_seed(int(args.seed) + 1)

    for f_idx, fam in enumerate(families):
        batch = collector(fam, num_aleatory=int(args.k_eval))
        z_e = sample_epistemic_indices(
            int(args.m_eval), module.epistemic_dim,
            device=batch.features.device, dtype=batch.features.dtype, generator=z_gen,
        )
        # Chunk the epistemic axis so the eval forward fits on a 40GB card;
        # assemble the nested prediction on CPU.
        chunks = []
        for m in range(int(args.m_eval)):
            pred_m = neon_stage2_eval_forward(
                module=module, base_prediction=batch.base_prediction,
                features=batch.features, z_e=z_e[m : m + 1],
            )
            chunks.append(pred_m.detach().to("cpu", torch.float32))
        prediction = torch.cat(chunks, dim=1)                       # [1, M, K, T, Nv, C]
        reference = fam.reference.unsqueeze(0).to(prediction.dtype) # [1, R, T, Nv, C]
        weights = fam.weights

        row: dict[str, Any] = {"family_id": fam.family_id}
        row.update(evaluate_neon_nested(prediction, reference,
                                        thresholds=tuple(args.thresholds), weights=weights))
        per_family.append(row)

        pit = nested_pit_rank_histograms(
            prediction, reference, seed=int(args.seed),
            min_ref_depth=float(args.pit_min_ref_depth),
        )
        if pit_total is None:
            pit_total = pit
        else:
            pit_total["pit_counts"] = (np.array(pit_total["pit_counts"]) + np.array(pit["pit_counts"])).tolist()
            pit_total["rank_counts"] = (np.array(pit_total["rank_counts"]) + np.array(pit["rank_counts"])).tolist()

        rel = exceedance_reliability(prediction, reference, thresholds=tuple(args.thresholds))
        for key, bins in rel.items():
            if key not in reliability_sums:
                reliability_sums[key] = [
                    {"bin_lo": b["bin_lo"], "bin_hi": b["bin_hi"], "n": 0.0,
                     "sum_forecast_prob": 0.0, "sum_observed_freq": 0.0}
                    for b in bins
                ]
            for acc, b in zip(reliability_sums[key], bins):
                acc["n"] += b["n"]
                acc["sum_forecast_prob"] += b["sum_forecast_prob"]
                acc["sum_observed_freq"] += b["sum_observed_freq"]

        spread = spread_error_diagnostics(prediction, reference)
        spread_corrs.append(spread["spread_error_corr"])
        row["spread_error_corr"] = spread["spread_error_corr"]

        if args.impact_metrics:
            from neuralop.flood.eval.impact_metrics import compute_flood_impact_crps_metrics

            flat = prediction.reshape(1, -1, *prediction.shape[3:])[0, :, :, :, 0]  # [MK, T, Nv]
            static_raw = np.stack(
                [prepared["elevation_raw_np"], prepared["cell_area_m2_np"]], axis=1
            )
            impact = compute_flood_impact_crps_metrics(
                flat.permute(1, 0, 2).numpy(),                     # (T, ens, cells)
                reference[0, :, :, :, 0].permute(1, 0, 2).numpy(), # (T, ref, cells)
                prepared["geometry_raw_np"],
                static_raw=static_raw,
            )
            impact_scalars = {
                k: float(np.nanmean(v)) for k, v in impact.items()
            }
            impact_scalars["family_id"] = fam.family_id
            impact_rows.append(impact_scalars)

        if args.write_artifacts:
            artifact_dir = out_dir / "artifacts"
            artifact_dir.mkdir(exist_ok=True)
            save_nested_forecast_artifact(
                artifact_dir / f"{fam.family_id}.h5",
                hydrograph_id=fam.family_id,
                prediction=prediction,
                ref_members_wd=reference[0, :, :, :, 0].numpy(),
                geometry_raw=prepared["geometry_raw_np"],
                elevation_raw=prepared["elevation_raw_np"],
                metadata={"m_eval": int(args.m_eval), "k_eval": int(args.k_eval),
                          "stage2_checkpoint": str(args.stage2_checkpoint)},
            )

        if f_idx < int(args.variance_maps):
            write_variance_maps(
                prediction, geometry_xy=prepared["geometry_raw_np"],
                output_dir=out_dir / "variance_maps", label=fam.family_id,
            )
        log.info("family %s done (%d/%d)", fam.family_id, f_idx + 1, len(families))

    # ---- Aggregate ----
    scalar_keys = [k for k, v in per_family[0].items() if isinstance(v, float)]
    aggregate = {k: float(np.mean([row[k] for row in per_family])) for k in scalar_keys}
    aggregate["spread_error_corr_mean"] = float(np.mean(spread_corrs))
    reliability_curves = {}
    for key, bins in reliability_sums.items():
        reliability_curves[key] = [
            {
                "bin_lo": b["bin_lo"], "bin_hi": b["bin_hi"], "n": int(b["n"]),
                "forecast_prob": (b["sum_forecast_prob"] / b["n"]) if b["n"] else None,
                "observed_freq": (b["sum_observed_freq"] / b["n"]) if b["n"] else None,
            }
            for b in bins
        ]

    payload = {
        "plan": plan,
        "checkpoint_metadata": {k: v for k, v in ckpt_meta.items() if isinstance(v, (str, int, float, bool))},
        "aggregate": aggregate,
        "per_family": per_family,
        "pit_rank": pit_total,
        "reliability": reliability_curves,
        "impact_metrics": impact_rows,
    }
    metrics_path = out_dir / "neon_eval_metrics.json"
    with metrics_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    log.info("wrote %s", metrics_path)
    print("NEON EVAL OK")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
