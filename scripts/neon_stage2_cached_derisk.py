"""De-risk the cached feature collector before the full 12h Stage-2 run.

Small settings (3 families, T=12, 2 epochs, cache_features=True) to confirm:
 - epoch 0 rolls + caches the frozen features,
 - epoch 1 is a cache hit (fast, fp16->fp32 round-trip does not break the EpiNet),
 - shapes/metrics stay finite.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("neon_cached_derisk")

EVAL_CONFIG = (
    "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn_eval60/"
    "coast_fgn3x20_eval_currentviz_20260506_152059/config/coast_fgn3x20_eval_currentviz.yaml"
)
BUNDLE = (
    "/scratch/jrj6wm/GINO_Model/model_bundles/"
    "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json"
)
OUT_DIR = Path("/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/cached_derisk")


def main() -> int:
    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.eval.datasets import _load_or_fit_normalizers
    from neuralop.flood.neon_config import NEONStage2Config
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import run_neon_stage2_training
    from neuralop.flood.utils.runtime_core import load_config_and_setup, parse_target_variables

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s", device)

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["derisk", "--config_path", EVAL_CONFIG]
        flood_config, _d, _l = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(getattr(flood_config.data, "target_variables", ["wd"]))
    normalizers, _ = _load_or_fit_normalizers(flood_config, None, None, LOG)

    train_fam, val_fam = build_families_from_config(
        flood_config, normalizers, target_variables, LOG,
        structural_dry_artifact=None, rollout_length=12, max_families=3, val_fraction=0.34,
    )
    LOG.info("families train=%d val=%d T=%d", len(train_fam), len(val_fam), int(train_fam[0].reference.shape[1]))

    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]

    config = NEONStage2Config(
        enabled=True, feature_source="decoder_pre_projection", dependency="za_dependent",
        d_e=16, m_train=4, k_train=8, m_eval=16, k_eval=50,
        prior_scale="auto_0p10_base_rmse", n_epochs=2, lead_time_dim=0,
    )
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    result = run_neon_stage2_training(
        config=config, stage1_checkpoint=BUNDLE, output_dir=OUT_DIR, data_root=EVAL_CONFIG,
        load_stage1_fn=lambda _c: stage1, build_families_fn=lambda _r, _c: (train_fam, val_fam),
        latent_dim=int(bundle.fgn_noise_dim), n_history=int(bundle.n_history),
        out_channels=len(target_variables), generator=gen, calibrate_prior=True,
        cache_features=True, cache_device="cpu",
    )
    LOG.info("done in %.1fs best_epoch=%s best_val_fit=%.5f", time.time() - t0, result.best_epoch, result.best_val_fit)
    for row in result.history:
        LOG.info("  epoch %d train_fit=%.5f val_fit=%.5f", row["epoch"], row["train_fit"], row["val_fit"])
    assert all(torch.isfinite(torch.tensor(row["val_fit"])) for row in result.history), "non-finite val_fit"
    print("CACHED DERISK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
