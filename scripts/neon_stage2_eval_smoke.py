"""NEON Stage-2 EVAL-side smoke: nested evaluation of a trained EpiNet.

Loads the Stage-2 checkpoint produced by ``neon_stage2_real_ref_smoke.py`` and
runs the full nested-evaluation path against REAL HEC-RAS reference ensembles:

  frozen FGN K-member rollout  ->  base_prediction + features
  EpiNet over M epistemic z_e  ->  nested prediction [B, M, K, T, Nv, C]
  evaluate_neon_nested         ->  RMSE / fair-CRPS / Brier / CSI + variance
  write_variance_maps          ->  aleatory/epistemic/ANOVA/total PNGs

It enables the structural-dry artifact so wettable-area weights flow through the
weighted metric + variance path. If the config has no masked_primary artifact,
it falls back to a reference-derived wettable mask so the weighted code path is
still exercised on real tensors (and logs which path was taken).

Run inside the pytorch container on a GPU node, e.g.::

    srun -A uqgroup -p gpu --gres=gpu:a100:1 -t 0:40:00 --pty \
      apptainer exec --nv <container> python scripts/neon_stage2_eval_smoke.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("neon_eval_smoke")

EVAL_CONFIG = (
    "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn_eval60/"
    "coast_fgn3x20_eval_currentviz_20260506_152059/config/coast_fgn3x20_eval_currentviz.yaml"
)
BUNDLE = (
    "/scratch/jrj6wm/GINO_Model/model_bundles/"
    "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json"
)
CKPT = "/scratch/jrj6wm/GINO_Model/neon_stage2_smoke_out/real_ref/neon_stage2_best.pt"
OUT_DIR = Path("/scratch/jrj6wm/GINO_Model/neon_stage2_smoke_out/eval")

MAX_FAMILIES = 2
ROLLOUT_LENGTH = 6
K_EVAL = 4
M_EVAL = 4


def main() -> int:
    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.eval.datasets import (
        _load_or_fit_normalizers,
        _load_structural_dry_artifact_for_eval,
    )
    from neuralop.flood.eval.neon import evaluate_neon_nested, write_variance_maps
    from neuralop.flood.neon import (
        load_neon_stage2_checkpoint,
        sample_epistemic_indices,
    )
    from neuralop.flood.train.neon import neon_stage2_eval_forward
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import make_feature_collector_from_frozen_model
    from neuralop.flood.utils.runtime_core import (
        load_config_and_setup,
        parse_target_variables,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s", device)

    # --- 1) Flood eval config + normalizers ---
    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_eval_smoke", "--config_path", EVAL_CONFIG]
        flood_config, _dev, _is_logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(
        getattr(flood_config.data, "target_variables", ["wd"])
    )
    normalizers, normalizer_path = _load_or_fit_normalizers(flood_config, None, None, LOG)

    # --- 2) Structural-dry artifact (enables wettable-area weights) ---
    dry_artifact = None
    try:
        _policy, dry_artifact = _load_structural_dry_artifact_for_eval(
            flood_config, normalizer_path=Path(normalizer_path), logger=LOG
        )
    except Exception as exc:  # noqa: BLE001 - smoke tolerates missing artifact
        LOG.warning("structural-dry artifact unavailable (%s); will derive a wettable mask", exc)
    LOG.info("structural_dry_artifact loaded=%s", dry_artifact is not None)

    # --- 3) Real grouped families (with weights if the dry artifact loaded) ---
    train_fam, val_fam = build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        LOG,
        structural_dry_artifact=dry_artifact,
        rollout_length=ROLLOUT_LENGTH,
        max_families=MAX_FAMILIES,
        val_fraction=0.5,
    )
    families = (train_fam + val_fam)
    fam = families[0]
    LOG.info("family id=%s reference=%s weights=%s", fam.family_id,
             tuple(fam.reference.shape),
             None if fam.weights is None else tuple(fam.weights.shape))

    # --- 4) Load trained EpiNet checkpoint + frozen FGN ---
    module, meta = load_neon_stage2_checkpoint(CKPT, map_location=device)
    module = module.to(device).eval()
    feature_source = meta.get("feature_source", "decoder_pre_projection")
    LOG.info("checkpoint loaded: feature_source=%s d_e=%s", feature_source, module.epistemic_dim)

    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared  # type: ignore[attr-defined]
    collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=feature_source,
        n_history=int(bundle.n_history),
        latent_dim=int(bundle.fgn_noise_dim),
        generator=torch.Generator().manual_seed(0),
    )

    # --- 5) Assemble the nested prediction [B, M, K, T, Nv, C] ---
    probe = collector(fam, num_aleatory=K_EVAL)
    z_e = sample_epistemic_indices(
        M_EVAL, module.epistemic_dim,
        device=probe.features.device, dtype=probe.features.dtype,
        generator=torch.Generator().manual_seed(1),
    )
    prediction = neon_stage2_eval_forward(
        module=module, base_prediction=probe.base_prediction, features=probe.features, z_e=z_e
    )
    reference = fam.reference.unsqueeze(0).to(device=prediction.device, dtype=prediction.dtype)
    LOG.info("prediction=%s reference=%s", tuple(prediction.shape), tuple(reference.shape))

    # --- 6) Weights: real wettable mask, or reference-derived fallback ---
    if fam.weights is not None:
        weights = fam.weights.to(device=prediction.device, dtype=prediction.dtype)
        weight_source = "structural_dry_artifact"
    else:
        # wettable = any reference member ever wet (>1cm) over the horizon
        ever_wet = (reference[0] > 0.01).any(dim=0).any(dim=0)  # [Nv, C]
        T = int(prediction.shape[3])
        weights = ever_wet.to(prediction.dtype).unsqueeze(0).expand(T, -1, -1).contiguous()
        weight_source = "reference_derived_wettable"
    LOG.info("weights source=%s shape=%s wettable_cells=%d/%d",
             weight_source, tuple(weights.shape),
             int((weights[0, :, 0] > 0).sum()), int(weights.shape[1]))

    # --- 7) Full nested evaluation bundle ---
    metrics = evaluate_neon_nested(prediction, reference, weights=weights)
    LOG.info("=== NEON nested metrics ===")
    for k in sorted(metrics):
        LOG.info("  %-45s = %s", k, f"{metrics[k]:.6g}")

    # sanity: finiteness + epistemic fraction in [0,1]
    frac = metrics["variance_epistemic_fraction_anova_corrected_mean"]
    assert np.isfinite(list(metrics.values())).all(), "non-finite metric present"
    assert 0.0 <= frac <= 1.0 + 1e-6, f"epistemic fraction out of range: {frac}"

    # --- 8) Variance maps to PNG (real UTM geometry) ---
    geometry_xy = prepared["geometry_raw_np"]  # [Nv, 2]
    paths = write_variance_maps(
        prediction, geometry_xy=geometry_xy, output_dir=OUT_DIR, label="neon_eval_smoke", time_index=0
    )
    LOG.info("variance maps written: %d files -> %s", len(paths), OUT_DIR)
    for p in paths:
        LOG.info("  %s exists=%s", p, Path(p).exists())

    print("EVAL SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
