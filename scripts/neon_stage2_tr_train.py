"""NEON Stage-2 training on the TR (training) package with a 450/50 family split.

This is the first plan-compliant reportable run: Stage-2 fits on TR families
only, validates on the 50 held-out TR families (fixed by sorted family ID), and
leaves the 50 TE test families untouched for final evaluation.

Parameterized via environment for the data-ablation grid:
  NEON_N_TRAIN   number of training families from the 450-family fit pool
                 (default 450; ablation grid uses 25/50/100/250/400)
  NEON_OUT_DIR   output directory (default .../tr_n<NEON_N_TRAIN>)
  NEON_CACHE_DIR shared frozen-feature disk cache (default shared across runs,
                 so grid runs reuse the same cached rollouts)

Fixed choices (plan + hardening):
  - families from the TR package config (grouped R=10 references)
  - full horizon T=94, K_train=8, M_train=4, d_e=16, 30 epochs
  - prior auto-calibration at 0.10 x base RMSE, prior_seed recorded
  - fixed validation generator (val_seed) for deterministic model selection
  - m_eval=32, k_eval=50 recorded for downstream nested evaluation
  - normalizer fingerprint (sha256) + structural-dry policy in metadata
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("neon_tr_train")

TR_CONFIG = "/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/config/coast_fgn_neon_tr450.yaml"
BUNDLE = (
    "/scratch/jrj6wm/GINO_Model/model_bundles/"
    "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json"
)

N_TRAIN = int(os.environ.get("NEON_N_TRAIN") or "450")
OUT_DIR = Path(
    os.environ.get("NEON_OUT_DIR")
    or f"/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/tr_n{N_TRAIN}"
)
CACHE_DIR = Path(
    os.environ.get("NEON_CACHE_DIR")
    or "/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/feature_cache_tr_k8"
)

PRIOR_SEED = 20260703
VAL_SEED = 1234


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
    from neuralop.flood.eval.datasets import _load_or_fit_normalizers
    from neuralop.flood.neon_config import NEONStage2Config
    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.train.neon_runner import run_neon_stage2_training
    from neuralop.flood.utils.runtime_core import (
        load_config_and_setup,
        parse_target_variables,
    )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    LOG.info("device=%s N_TRAIN=%d OUT_DIR=%s CACHE_DIR=%s", device, N_TRAIN, OUT_DIR, CACHE_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_tr_train", "--config_path", TR_CONFIG]
        flood_config, _dev, _is_logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(
        getattr(flood_config.data, "target_variables", ["wd"])
    )
    normalizers, normalizer_path = _load_or_fit_normalizers(flood_config, None, None, LOG)

    t0 = time.time()
    train_fam, val_fam = build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        LOG,
        structural_dry_artifact=None,
        rollout_length=None,      # full horizon
        max_families=None,        # all 500 TR families
        val_fraction=0.1,         # deterministic: last 50 by sorted family ID
    )
    # Fixed validation set across all grid runs; training subset = first
    # N_TRAIN of the sorted 450-family fit pool.
    train_fam = sorted(train_fam, key=lambda f: f.family_id)[:N_TRAIN]
    LOG.info(
        "families built in %.1fs: train=%d (of 450 pool) val=%d T=%d Nv=%d | val ids %s..%s",
        time.time() - t0, len(train_fam), len(val_fam),
        int(train_fam[0].reference.shape[1]), int(train_fam[0].reference.shape[2]),
        val_fam[0].family_id, val_fam[-1].family_id,
    )

    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    latent_dim = int(bundle.fgn_noise_dim)
    n_history = int(bundle.n_history)

    config = NEONStage2Config(
        enabled=True,
        feature_source="decoder_pre_projection",
        dependency="za_dependent",
        d_e=16,
        m_train=4,
        k_train=8,
        m_eval=32,
        k_eval=50,
        prior_scale="auto_0p10_base_rmse",
        n_epochs=30,
        lead_time_dim=0,
    )

    normalizer_fingerprint = {
        "path": str(normalizer_path),
        "sha256": _sha256(str(normalizer_path)),
    }
    LOG.info("normalizer fingerprint: %s", normalizer_fingerprint)

    gen = torch.Generator().manual_seed(0)
    t1 = time.time()
    result = run_neon_stage2_training(
        config=config,
        stage1_checkpoint=BUNDLE,
        output_dir=OUT_DIR,
        data_root=TR_CONFIG,
        load_stage1_fn=lambda _ckpt: stage1,
        build_families_fn=lambda _root, _cfg: (train_fam, val_fam),
        latent_dim=latent_dim,
        n_history=n_history,
        out_channels=len(target_variables),
        generator=gen,
        calibrate_prior=True,
        cache_features=True,
        cache_device="cpu",
        cache_dir=CACHE_DIR,
        epistemic_chunk_size=1,
        val_seed=VAL_SEED,
        prior_seed=PRIOR_SEED,
        normalizer_fingerprint=normalizer_fingerprint,
        structural_dry_policy={
            "policy": "none",
            "weights": "wettable_area_from_cell_area" ,
            "note": "no masked_primary artifact in TR config; family weights are cell-area based when static_raw is present, else uniform",
        },
    )
    LOG.info("training done in %.1f min", (time.time() - t1) / 60.0)

    hist_path = OUT_DIR / "history.json"
    with hist_path.open("w") as fh:
        json.dump({
            "n_train": N_TRAIN,
            "val_seed": VAL_SEED,
            "prior_seed": PRIOR_SEED,
            "best_epoch": result.best_epoch,
            "best_val_fit": result.best_val_fit,
            "history": result.history,
        }, fh, indent=2)
    LOG.info("best_epoch=%s best_val_fit=%.6f history=%s",
             result.best_epoch, result.best_val_fit, hist_path)
    for row in result.history:
        LOG.info("  epoch %2d  train_fit=%.5f  train_total=%.5f  val_fit=%.5f",
                 row["epoch"], row["train_fit"], row["train_total"], row["val_fit"])
    print("TR TRAIN OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
