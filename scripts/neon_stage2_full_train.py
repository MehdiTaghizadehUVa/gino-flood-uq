"""Full NEON Stage-2 training on the frozen coastal FGN with real HEC-RAS refs.

All grouped-hydrograph families, full forecast horizon (T=94), config-default
scale (M_train=4, K_train=8, d_e=16), 30 epochs, prior-scale auto-calibration,
wettable-area loss and the Stage-1 structural-dry policy. Frozen-FGNO features are cached as fp16 tensors in
an on-disk cache under OUT_DIR, so full 500-family runs do not materialize the
entire feature bank in process memory.

Run as a batch job (see scripts/sbatch_neon_full_train.sh).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("neon_full_train")

EVAL_CONFIG = (
    "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn_eval60/"
    "coast_fgn3x20_eval_currentviz_20260506_152059/config/coast_fgn3x20_eval_currentviz.yaml"
)
BUNDLE = (
    "/scratch/jrj6wm/GINO_Model/model_bundles/"
    "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json"
)
OUT_DIR = Path("/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/real_ref_full")
OUT_DIR = Path(os.environ.get("NEON_STAGE2_OUT_DIR", str(OUT_DIR)))
_CACHE_DIR_ENV = os.environ.get("NEON_STAGE2_CACHE_DIR")
CACHE_DIR = Path(_CACHE_DIR_ENV) if _CACHE_DIR_ENV else OUT_DIR / "feature_cache"
N_EPOCHS = int(os.environ.get("NEON_STAGE2_EPOCHS", "30"))
PROGRESS_INTERVAL = int(os.environ.get("NEON_STAGE2_PROGRESS_INTERVAL", "10"))
RESUME_ENABLED = os.environ.get("NEON_STAGE2_RESUME", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
LATEST_STATE = Path(os.environ.get("NEON_STAGE2_LATEST_STATE", str(OUT_DIR / "neon_stage2_latest.pt")))


def main() -> int:
    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.neon_config import NEONStage2Config
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import run_neon_stage2_training
    from neuralop.flood.utils.runtime_core import (
        load_config_and_setup,
        parse_target_variables,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s", device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Flood eval config + normalizers ---
    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_full_train", "--config_path", EVAL_CONFIG]
        flood_config, _dev, _is_logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(
        getattr(flood_config.data, "target_variables", ["wd"])
    )
    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared  # type: ignore[attr-defined]
    normalizers = prepared["normalizers"]
    dry_mask = prepared.get("structural_dry_mask")
    LOG.info("normalizers from frozen bundle %s", bundle.normalizer_path)

    # --- All grouped training families, full horizon, frozen dry-domain policy ---
    t0 = time.time()
    train_fam, val_fam = build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        LOG,
        structural_dry_artifact=(None if dry_mask is None else {"dry_mask": dry_mask}),
        rollout_length=None,            # full available horizon (T=94)
        max_families=None,              # all hydrographs
        val_fraction=0.1,
        dataset_split="train",
    )
    LOG.info(
        "families built in %.1fs: train=%d val=%d T=%d Nv=%d",
        time.time() - t0, len(train_fam), len(val_fam),
        int(train_fam[0].reference.shape[1]), int(train_fam[0].reference.shape[2]),
    )

    # --- Frozen FGN ---
    latent_dim = int(bundle.fgn_noise_dim)
    n_history = int(bundle.n_history)
    LOG.info("frozen FGN: fgn_noise_dim=%d n_history=%d", latent_dim, n_history)

    # --- Config-default full scale ---
    config = NEONStage2Config(
        enabled=True,
        feature_source="decoder_pre_projection",
        dependency="za_dependent",
        d_e=16,
        m_train=4,
        k_train=8,
        m_eval=16,
        k_eval=50,
        prior_scale="auto_0p10_base_rmse",
        n_epochs=N_EPOCHS,
        lead_time_dim=0,
        progress_log_interval_effective_batches=PROGRESS_INTERVAL,
    )
    LOG.info("config: %s", json.dumps({
        "d_e": config.d_e, "m_train": config.m_train, "k_train": config.k_train,
        "n_epochs": config.n_epochs, "prior_scale": config.prior_scale,
        "feature_source": config.feature_source, "dependency": config.dependency,
        "progress_log_interval_effective_batches": config.progress_log_interval_effective_batches,
    }))
    LOG.info("out_dir=%s", OUT_DIR)
    LOG.info("feature_cache_dir=%s", CACHE_DIR)
    LOG.info("latest_state=%s resume_enabled=%s exists=%s",
             LATEST_STATE, RESUME_ENABLED, LATEST_STATE.exists())

    gen = torch.Generator().manual_seed(0)
    t1 = time.time()
    def load_prepared_stage1(_checkpoint):
        return stage1

    load_prepared_stage1.last_prepared = prepared
    result = run_neon_stage2_training(
        config=config,
        stage1_checkpoint=BUNDLE,
        output_dir=OUT_DIR,
        data_root=EVAL_CONFIG,
        load_stage1_fn=load_prepared_stage1,
        build_families_fn=lambda _root, _cfg: (train_fam, val_fam),
        latent_dim=latent_dim,
        n_history=n_history,
        out_channels=len(target_variables),
        generator=gen,
        calibrate_prior=True,
        cache_features=True,      # CPU fp16 cache of the frozen rollout
        cache_device="cpu",
        cache_dir=CACHE_DIR,
        epistemic_chunk_size=1,   # process one z_e at a time so backward fits on a 40GB A100
        latest_checkpoint_path=LATEST_STATE,
        resume_state_path=LATEST_STATE if RESUME_ENABLED else None,
        structural_dry_policy={
            "policy": "frozen_stage1_bundle",
            "feedback_clamp": dry_mask is not None,
        },
    )
    LOG.info("training done in %.1f min", (time.time() - t1) / 60.0)

    # Persist the full history for later plotting/analysis.
    hist_path = OUT_DIR / "history.json"
    with hist_path.open("w") as fh:
        json.dump({"best_epoch": result.best_epoch,
                   "best_val_fit": result.best_val_fit,
                   "history": result.history}, fh, indent=2)
    LOG.info("best_epoch=%s best_val_fit=%.6f history=%s",
             result.best_epoch, result.best_val_fit, hist_path)
    for row in result.history:
        LOG.info("  epoch %2d  train_fit=%.5f  train_total=%.5f  val_fit=%.5f",
                 row["epoch"], row["train_fit"], row["train_total"], row["val_fit"])
    LOG.info("checkpoint=%s exists=%s",
             OUT_DIR / "neon_stage2_best.pt", (OUT_DIR / "neon_stage2_best.pt").exists())
    print("FULL TRAIN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
