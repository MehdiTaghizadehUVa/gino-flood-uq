"""NEON Stage-2 training on the TR (training) package with a 450/50 family split.

This is the first plan-compliant reportable run: Stage-2 fits on TR families
only, validates on the 50 held-out TR families (fixed by sorted family ID), and
leaves the 50 TE test families untouched for final evaluation.

Parameterized via environment for the data-ablation grid:
  NEON_N_TRAIN   number of training families from the 450-family fit pool
                 (default 450; ablation grid uses 25/50/100/250/400)
  NEON_OUT_DIR   output directory (default .../tr_n<NEON_N_TRAIN>)
  NEON_SUBSET_REPLICATE seeded nested-subset replicate (optional; 0--4 grid)
  NEON_CACHE_DIR shared frozen-feature disk cache (default shared across runs,
                 so grid runs reuse the same cached rollouts)

Fixed choices (plan + hardening):
  - families from the TR package config (grouped R=10 references)
  - full horizon T=94, K_train=8, M_train=4, d_e=D_E, 30 epochs
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
from dataclasses import asdict
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
_SUBSET_REPLICATE_ENV = os.environ.get("NEON_SUBSET_REPLICATE")
LADDER_RUNG = os.environ.get("NEON_LADDER_RUNG") or "B3"
SUBSET_REPLICATE = int(_SUBSET_REPLICATE_ENV or "0")
SUBSET_BASE_SEED = int(os.environ.get("NEON_SUBSET_BASE_SEED") or "20260712")
_DEFAULT_OUT_NAME = (
    f"tr_n{N_TRAIN}_rep{SUBSET_REPLICATE}_{LADDER_RUNG.lower()}"
    if _SUBSET_REPLICATE_ENV is not None
    else f"tr_n{N_TRAIN}_{LADDER_RUNG.lower()}"
)
OUT_DIR = Path(
    os.environ.get("NEON_OUT_DIR")
    or f"/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/{_DEFAULT_OUT_NAME}"
)
CACHE_DIR = Path(
    os.environ.get("NEON_CACHE_DIR")
    or "/scratch/jrj6wm/GINO_Model/neon_stage2_full_train/feature_cache_tr_k8"
)

PRIOR_SEED = 20260703
PRIOR_SCALE = os.environ.get("NEON_PRIOR_SCALE") or "auto_0p10_base_rmse"
D_E = int(os.environ.get("NEON_D_E") or "16")
N_EPOCHS = int(os.environ.get("NEON_EPOCHS") or "30")
VAL_SEED = 1234


def _ladder_overrides(
    rung: str, *, de_spread_multiplier: float | None = None
) -> dict:
    rung = str(rung).strip().upper()
    if rung not in {"B0", "B1A", "B1B", "B2", "B3", "B4", "B5"}:
        raise ValueError(f"unsupported NEON ladder rung {rung!r}.")
    values = {
        "bootstrap_distribution": "tempered_exponential",
        "member_bootstrap_enabled": True,
        "epistemic_basis": "identity",
        "concat_index": True,
        "prior_rff_dim": 32,
        "deterministic_head": False,
        "selection_metric": "per_epistemic_fit",
        "selection_enforce_rmse": False,
        "selection_min_retention": 0.0,
    }
    if rung in {"B1A", "B1B", "B2", "B3", "B4", "B5"}:
        values["member_bootstrap_enabled"] = False
    if rung in {"B1B", "B2", "B3", "B4", "B5"}:
        values["bootstrap_distribution"] = "probit_exponential"
    if rung in {"B2", "B3", "B4", "B5"}:
        values.update(
            epistemic_basis="hermite_random_projection",
            concat_index=False,
            prior_rff_dim=0,
            deterministic_head=True,
            deterministic_head_feature="canonical_aleatory_mean",
        )
    if rung in {"B3", "B4", "B5"}:
        values.update(selection_metric="mixture_crps", selection_enforce_rmse=True)
    if rung == "B4":
        multiplier = float(
            de_spread_multiplier
            if de_spread_multiplier is not None
            else (os.environ.get("NEON_DE_SPREAD_MULTIPLIER") or "1.0")
        )
        if multiplier not in {0.5, 1.0, 2.0}:
            raise ValueError("B4 requires NEON_DE_SPREAD_MULTIPLIER in {0.5,1.0,2.0}.")
        values["prior_scale"] = {
            "mode": "de_spread_target",
            "target_std_m": 0.046 * multiplier,
        }
    if rung == "B5":
        values.update(
            epistemic_index_mode="dirichlet_particles",
            epistemic_basis="identity",
            concat_index=False,
            prior_rff_dim=0,
            d_e=16,
            dirichlet_num_particles=16,
            m_eval=16,
        )
    return values


def _resolved_ladder_config(
    rung: str,
    *,
    prior_scale: str,
    d_e: int,
    n_epochs: int,
    de_spread_multiplier: float | None = None,
):
    """Resolve and validate one attribution-ladder configuration."""
    from neuralop.flood.neon_config import NEONStage2Config

    config_kwargs = dict(
        enabled=True,
        feature_source="decoder_pre_projection",
        dependency="za_dependent",
        d_e=int(d_e),
        m_train=4,
        k_train=8,
        m_eval=16,
        k_eval=50,
        prior_scale=prior_scale,
        n_epochs=int(n_epochs),
        lead_time_dim=0,
    )
    config_kwargs.update(
        _ladder_overrides(
            rung, de_spread_multiplier=de_spread_multiplier
        )
    )
    return NEONStage2Config(**config_kwargs).validate()


def _write_preflight_manifest(
    path: Path,
    *,
    config,
    rung: str,
    n_train: int,
    output_dir: Path,
    cache_dir: Path,
    subset_replicate: int,
) -> None:
    """Atomically persist the exact validated configuration before sbatch."""
    payload = {
        "schema_version": "neon_repair_preflight_v1",
        "ladder_rung": str(rung).upper(),
        "n_train": int(n_train),
        "subset_replicate": int(subset_replicate),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "config": asdict(config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    config = _resolved_ladder_config(
        LADDER_RUNG,
        prior_scale=PRIOR_SCALE,
        d_e=D_E,
        n_epochs=N_EPOCHS,
    )
    if os.environ.get("NEON_PLAN_ONLY") == "1":
        preflight_path = Path(
            os.environ.get("NEON_PREFLIGHT_PATH") or OUT_DIR / "preflight.json"
        )
        _write_preflight_manifest(
            preflight_path,
            config=config,
            rung=LADDER_RUNG,
            n_train=N_TRAIN,
            output_dir=OUT_DIR,
            cache_dir=CACHE_DIR,
            subset_replicate=SUBSET_REPLICATE,
        )
        print(preflight_path)
        return 0

    from neuralop.flood.cli.train_neon_stage2 import _load_frozen_stage1
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
    stage1 = _load_frozen_stage1(BUNDLE)
    bundle = _load_frozen_stage1.last_bundle  # type: ignore[attr-defined]
    prepared = _load_frozen_stage1.last_prepared  # type: ignore[attr-defined]
    normalizers = prepared["normalizers"]
    normalizer_path = bundle.normalizer_path
    dry_mask = prepared.get("structural_dry_mask")

    t0 = time.time()
    train_fam, val_fam = build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        LOG,
        structural_dry_artifact=(None if dry_mask is None else {"dry_mask": dry_mask}),
        rollout_length=None,      # full horizon
        max_families=None,        # all 500 TR families
        val_fraction=0.1,         # deterministic: last 50 by sorted family ID
    )
    # Fixed validation set; each replicate uses one seeded permutation whose
    # prefixes define nested N=25,...,400 subsets.
    fit_pool = sorted(train_fam, key=lambda f: f.family_id)
    subset_generator = torch.Generator().manual_seed(
        SUBSET_BASE_SEED + SUBSET_REPLICATE
    )
    subset_order = torch.randperm(len(fit_pool), generator=subset_generator).tolist()
    train_fam = [fit_pool[index] for index in subset_order[:N_TRAIN]]
    LOG.info(
        "families built in %.1fs: train=%d (of 450 pool) val=%d T=%d Nv=%d | val ids %s..%s",
        time.time() - t0, len(train_fam), len(val_fam),
        int(train_fam[0].reference.shape[1]), int(train_fam[0].reference.shape[2]),
        val_fam[0].family_id, val_fam[-1].family_id,
    )

    latent_dim = int(bundle.fgn_noise_dim)
    n_history = int(bundle.n_history)

    normalizer_fingerprint = {
        "path": str(normalizer_path),
        "sha256": _sha256(str(normalizer_path)),
    }
    LOG.info("normalizer fingerprint: %s", normalizer_fingerprint)

    gen = torch.Generator().manual_seed(0)
    t1 = time.time()
    def load_prepared_stage1(_checkpoint):
        return stage1

    load_prepared_stage1.last_prepared = prepared
    result = run_neon_stage2_training(
        config=config,
        stage1_checkpoint=BUNDLE,
        output_dir=OUT_DIR,
        data_root=TR_CONFIG,
        load_stage1_fn=load_prepared_stage1,
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
            "policy": "frozen_stage1_bundle",
            "feedback_clamp": dry_mask is not None,
            "weights": "wettable_area_from_cell_area",
        },
    )
    LOG.info("training done in %.1f min", (time.time() - t1) / 60.0)

    hist_path = OUT_DIR / "history.json"
    with hist_path.open("w") as fh:
        json.dump({
            "n_train": N_TRAIN,
            "ladder_rung": LADDER_RUNG,
            "val_seed": VAL_SEED,
            "prior_seed": PRIOR_SEED,
            "subset_replicate": SUBSET_REPLICATE,
            "subset_seed": SUBSET_BASE_SEED + SUBSET_REPLICATE,
            "train_family_ids": [family.family_id for family in train_fam],
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
