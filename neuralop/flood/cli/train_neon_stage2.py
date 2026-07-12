"""CLI entrypoint for NEON Stage-2 epistemic FGNO training.

The plan-resolution layer (:func:`resolve_training_plan`) is torch-free so the
config wiring can be validated with ``--dry-run`` without loading a checkpoint
or dataset. The real training path lazily imports the torch-side helpers and
the grouped-hydrograph family loader.

Usage::

    python -m neuralop.flood.cli.train_neon_stage2 --config <neon.yaml> [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from neuralop.flood.neon_config import NEONStage2Config, load_neon_config


def resolve_training_plan(config: NEONStage2Config) -> dict[str, Any]:
    """Return a torch-free, fully-resolved training plan dict from a config.

    This is the single source of truth for "what will Stage-2 training do",
    used both by ``--dry-run`` and by the real training path.
    """
    config.validate()
    return {
        "feature_source": config.feature_source,
        "dependency": config.dependency,
        "objective": config.objective,
        "d_e": int(config.d_e),
        "m_train": int(config.m_train),
        "k_train": int(config.k_train),
        "m_eval": int(config.m_eval),
        "k_eval": int(config.k_eval),
        "n_epochs": int(config.n_epochs),
        "lead_time_dim": int(config.lead_time_dim),
        "branch_type": config.branch_type,
        "train_hidden_channels": int(config.train_hidden_channels),
        "prior_hidden_channels": int(config.prior_hidden_channels),
        "prior_rff_dim": int(config.prior_rff_dim),
        "prior_rff_lengthscale": float(config.prior_rff_lengthscale),
        "prior_rff_include_lead": bool(config.prior_rff_include_lead),
        "spatial_weights": config.spatial_weights,
        "lead_time_weights": config.lead_time_weights,
        "reference_term_for_logging": bool(config.reference_term_for_logging),
        "prior_scale_mode": "auto" if config.uses_auto_prior_scale else "explicit",
        "prior_scale_fraction": float(config.prior_scale_fraction),
        "alpha": None if config.alpha is None else float(config.alpha),
        "loss_weights": config.to_loss_weights_dict(),
        "optimizer": {
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
        },
        "stage1_checkpoint_dir": config.stage1_checkpoint_dir,
        "stage2_checkpoint_dir": config.stage2_checkpoint_dir,
    }


def _load_config_file(config_path: Path) -> NEONStage2Config:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is a runtime dep
        raise RuntimeError("Reading a NEON config requires PyYAML.") from exc
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return load_neon_config(raw)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train_neon_stage2",
        description="Train a NEON Stage-2 epistemic correction on a frozen FGNO.",
    )
    parser.add_argument("--config", required=True, help="Path to a NEON Stage-2 YAML config.")
    parser.add_argument(
        "--stage1-checkpoint",
        default=None,
        help="Frozen Stage-1 FGNO checkpoint (overrides neon.stage1_checkpoint_dir).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Stage-2 output dir (overrides neon.stage2_checkpoint_dir).",
    )
    parser.add_argument("--data-root", default=None, help="Grouped-hydrograph family data root.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print the training plan without loading model/data.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    config = _load_config_file(Path(args.config))
    plan = resolve_training_plan(config)

    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    # ---- Real training path (lazy imports keep --dry-run torch-free) ----
    stage1_ckpt = args.stage1_checkpoint or config.stage1_checkpoint_dir
    output_dir = args.output_dir or config.stage2_checkpoint_dir
    if not stage1_ckpt:
        raise SystemExit("A Stage-1 checkpoint is required (--stage1-checkpoint or neon.stage1_checkpoint_dir).")
    if not output_dir:
        raise SystemExit("An output dir is required (--output-dir or neon.stage2_checkpoint_dir).")
    if not args.data_root:
        raise SystemExit("A grouped-hydrograph --data-root is required for training.")

    # The heavy wiring lives in the training package; imported lazily so the
    # config/plan surface stays testable without torch.
    from neuralop.flood.train.neon_runner import run_neon_stage2_training  # type: ignore

    run_neon_stage2_training(
        config=config,
        stage1_checkpoint=stage1_ckpt,
        output_dir=output_dir,
        data_root=args.data_root,
        load_stage1_fn=_load_frozen_stage1,
        build_families_fn=_build_grouped_families,
        latent_dim=_infer_latent_dim(config),
    )
    return 0


def _infer_latent_dim(config: NEONStage2Config) -> int:
    """FGN aleatory latent dim (fgn_noise_dim). Overridable via config metadata."""
    return int(getattr(config, "fgn_noise_dim", 32) or 32)


def _load_frozen_stage1(stage1_checkpoint: Any):  # pragma: no cover - GPU/infra path
    """Load a frozen Stage-1 coastal FGNO from a serving model-bundle JSON.

    Reuses the serving inference loader (validated by the NEON GPU smoke): loads
    the bundle, builds ProductionFGNInferenceService, prepares the models and
    domain assets, and returns the first frozen FGN model. ``stage1_checkpoint``
    is the path to a ``coastal_fgn_bundle.json``. Serving metadata drift (e.g.
    dt_seconds) is tolerated so the real weights load regardless.
    """
    import json
    from pathlib import Path as _Path

    from neuralop.flood.serving.inference import ProductionFGNInferenceService
    from neuralop.flood.serving.model_bundle import load_model_bundle

    try:
        bundle = load_model_bundle(str(stage1_checkpoint))
    except Exception:
        with open(stage1_checkpoint) as handle:
            raw = json.load(handle)
        raw["dt_seconds"] = 900  # tolerate serving dt metadata drift
        patched = str(_Path(stage1_checkpoint).with_name("coastal_fgn_bundle_neon.json"))
        with open(patched, "w") as handle:
            json.dump(raw, handle)
        bundle = load_model_bundle(patched)

    import torch as _torch

    device = "cuda:0" if _torch.cuda.is_available() else "cpu"
    service = ProductionFGNInferenceService(bundle, device=device)
    prepared = service._ensure_loaded()
    # Cache the prepared assets so a companion family loader can reuse them.
    _load_frozen_stage1.last_prepared = prepared  # type: ignore[attr-defined]
    _load_frozen_stage1.last_bundle = bundle       # type: ignore[attr-defined]
    return prepared["models"][0]


def _build_grouped_families(data_root: Any, config: NEONStage2Config):  # pragma: no cover
    """Build (train, val) NEONFamilySample splits from grouped-hydrograph data.

    ``data_root`` is the path to a coastal flood eval-style YAML config. Its
    ``rollout_data`` section may point at the test package, but the NEON family
    converter switches to the training package described by ``data.train_root``
    for Stage-2 training. Normalizers are taken from the frozen Stage-1 bundle
    prepared by :func:`_load_frozen_stage1` (so the references are normalized
    in the same space as the model).
    """
    import logging
    import sys as _sys

    from neuralop.flood.train.neon_families import build_families_from_config
    from neuralop.flood.utils.runtime_core import (
        load_config_and_setup,
        parse_target_variables,
    )

    prepared = getattr(_load_frozen_stage1, "last_prepared", None)
    if prepared is None:
        raise RuntimeError(
            "_build_grouped_families needs the frozen Stage-1 bundle's normalizers; "
            "call _load_frozen_stage1 (load_stage1_fn) before building families."
        )
    normalizers = prepared["normalizers"]

    # Load the flood eval config via the shared runtime loader (reads
    # --config_path from argv), isolating our own argv so it does not clash.
    saved_argv = list(_sys.argv)
    try:
        _sys.argv = [saved_argv[0] if saved_argv else "neon", "--config_path", str(data_root)]
        flood_config, _device, _is_logger = load_config_and_setup()
    finally:
        _sys.argv = saved_argv

    target_variables = parse_target_variables(
        getattr(flood_config.data, "target_variables", ["wd"])
    )
    logger = logging.getLogger("neon_stage2.families")
    return build_families_from_config(
        flood_config,
        normalizers,
        target_variables,
        logger,
        rollout_length=getattr(config, "rollout_length", None),
        max_families=getattr(config, "max_families", None),
        val_fraction=float(getattr(config, "val_fraction", 0.1)),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
