"""NEON Stage-2 GPU smoke against the real coastal FGN with REAL HEC-RAS references.

Unlike ``neon_stage2_smoke.py`` (real model, synthetic reference ensembles), this
smoke sources grouped-hydrograph families from the coastal FGN training package by
converting the eval-style rollout config into a train split view. It then runs one
Stage-2 epoch end-to-end (probe -> build EpiNet -> auto-calibrate prior -> train ->
checkpoint) so we exercise the full real-reference path.

Run inside the pytorch container on a GPU node, e.g.::

    srun -A uqgroup -p gpu --gres=gpu:a100:1 -t 0:40:00 --pty \
      apptainer exec --nv <container> python scripts/neon_stage2_real_ref_smoke.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("neon_real_ref_smoke")

EVAL_CONFIG = (
    "/scratch/jrj6wm/GINO_Model/neuraloperator_runs/coastal_fgn_eval60/"
    "coast_fgn3x20_eval_currentviz_20260506_152059/config/coast_fgn3x20_eval_currentviz.yaml"
)
BUNDLE = (
    "/scratch/jrj6wm/GINO_Model/model_bundles/"
    "coastal_fgn_60_calibrated_v1_20260510/coastal_fgn_bundle.json"
)

# Keep the smoke small/fast.
MAX_FAMILIES = 3
ROLLOUT_LENGTH = 6      # cap AR horizon T for the smoke
VAL_FRACTION = 0.34     # -> ~1 val family of 3


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
    LOG.info("device=%s", device)

    # --- 1) Load the coastal flood eval config (converted to training package below) ---
    saved_argv = list(sys.argv)
    try:
        sys.argv = ["neon_real_ref_smoke", "--config_path", EVAL_CONFIG]
        flood_config, _dev, _is_logger = load_config_and_setup()
    finally:
        sys.argv = saved_argv
    target_variables = parse_target_variables(
        getattr(flood_config.data, "target_variables", ["wd"])
    )
    LOG.info("target_variables=%s n_history=%s skip=%s query_res=%s",
             target_variables,
             getattr(flood_config.data, "n_history", None),
             getattr(flood_config.data, "skip_before_timestep", None),
             getattr(flood_config.data, "query_res", None))

    # --- 2) Normalizers (loaded from training path; never refit in eval) ---
    normalizers, normalizer_path = _load_or_fit_normalizers(
        flood_config, None, None, LOG
    )
    LOG.info("normalizers loaded from %s (keys=%s)", normalizer_path, sorted(normalizers.keys()))

    # --- 3) Build REAL grouped-hydrograph families (capped for the smoke) ---
    train_fam, val_fam = build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        LOG,
        rollout_length=ROLLOUT_LENGTH,
        max_families=MAX_FAMILIES,
        val_fraction=VAL_FRACTION,
        dataset_split="train",
    )
    LOG.info("families: train=%d val=%d", len(train_fam), len(val_fam))
    f0 = train_fam[0]
    LOG.info(
        "family[0] id=%s reference=%s static=%s geometry=%s query=%s boundary=%s init_hist=%s weights=%s",
        f0.family_id,
        tuple(f0.reference.shape),
        tuple(f0.static.shape),
        tuple(f0.geometry.shape),
        tuple(f0.query_points.shape),
        tuple(f0.boundary_sequence.shape),
        tuple(f0.initial_histories.shape),
        None if f0.weights is None else tuple(f0.weights.shape),
    )

    # --- 4) Load the frozen coastal FGN (serving loader, tolerant of dt drift) ---
    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    latent_dim = int(bundle.fgn_noise_dim)
    n_history = int(bundle.n_history)
    LOG.info("frozen FGN loaded: fgn_noise_dim=%d n_history=%d", latent_dim, n_history)

    # --- 5) Small Stage-2 config; one epoch end-to-end ---
    config = NEONStage2Config(
        enabled=True,
        feature_source="decoder_pre_projection",
        dependency="za_dependent",
        d_e=4,
        m_train=2,
        k_train=2,
        m_eval=4,
        k_eval=4,
        prior_scale="auto_0p10_base_rmse",
        n_epochs=1,
        lead_time_dim=0,
    )

    gen = torch.Generator().manual_seed(0)
    out_dir = Path("/scratch/jrj6wm/GINO_Model/neon_stage2_smoke_out/real_ref")
    cache_dir = out_dir / "feature_cache"
    LOG.info("feature_cache_dir=%s", cache_dir)
    result = run_neon_stage2_training(
        config=config,
        stage1_checkpoint=BUNDLE,
        output_dir=out_dir,
        data_root=EVAL_CONFIG,
        load_stage1_fn=lambda _ckpt: stage1,
        build_families_fn=lambda _root, _cfg: (train_fam, val_fam),
        latent_dim=latent_dim,
        n_history=n_history,
        out_channels=len(target_variables),
        generator=gen,
        calibrate_prior=True,
        cache_features=True,
        cache_device="cpu",
        cache_dir=cache_dir,
    )

    LOG.info("TRAIN OK best_epoch=%s best_val_fit=%.4f", result.best_epoch, result.best_val_fit)
    ckpt = out_dir / "neon_stage2_best.pt"
    cache_files = sorted(cache_dir.glob("*.pt"))
    if not cache_files:
        raise RuntimeError(f"Expected disk-backed frozen-feature cache files in {cache_dir}")
    LOG.info("feature cache files=%d first=%s", len(cache_files), cache_files[0])
    LOG.info("checkpoint exists=%s at %s", ckpt.exists(), ckpt)
    print("REAL-REF SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
