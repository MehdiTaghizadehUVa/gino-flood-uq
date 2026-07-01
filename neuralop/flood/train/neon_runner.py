"""Production orchestration for NEON Stage-2 training (Gap 4).

Composes the tested Stage-2 building blocks into a single ``run_neon_stage2_training``
pipeline. The two infrastructure-dependent steps -- loading the frozen Stage-1
FGNO checkpoint and building the grouped-hydrograph family splits -- are
injected callables, so the orchestration is fully testable with fakes while the
real adapters are validated by a GPU smoke against a concrete checkpoint/dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple

import torch

from neuralop.flood.neon import (
    NEONStage2LossWeights,
    base_rmse_from_reference,
    calibrate_prior_scale,
    freeze_stage1_model,
    sample_epistemic_indices,
)
from neuralop.flood.train.neon import (
    NEONFamilySample,
    build_epinet_from_config,
    build_neon_stage2_metadata,
    build_neon_stage2_optimizer,
    collect_frozen_fgno_rollout_features,
    train_neon_stage2_epochs,
)

# load_stage1_fn(checkpoint) -> frozen-able nn.Module
LoadStage1Fn = Callable[[Any], torch.nn.Module]
# build_families_fn(data_root, config) -> (train_families, val_families)
BuildFamiliesFn = Callable[[Any, Any], Tuple[Sequence[NEONFamilySample], Sequence[NEONFamilySample]]]


def make_feature_collector_from_frozen_model(
    stage1_model: torch.nn.Module,
    *,
    feature_source: str,
    n_history: int,
    latent_dim: int,
    generator: Optional[torch.Generator] = None,
):
    """Return a ``feature_collector`` that AR-rolls the frozen FGNO per family.

    The returned callable matches the training loop's collector contract:
    ``(family, *, num_aleatory, generator) -> FrozenFGNOFeatureBatch``. It samples
    ``num_aleatory`` persistent aleatory latents, warm-starts from the family's
    ``initial_histories`` (broadcast to K if a single shared history is given),
    and rolls out to the family's reference horizon ``T``.
    """

    def collector(family: NEONFamilySample, *, num_aleatory: int, generator=generator):
        if family.initial_histories is None:
            raise ValueError(f"family {family.family_id!r} lacks initial_histories for rollout.")
        latents = torch.randn(int(num_aleatory), int(latent_dim), generator=generator)
        init = family.initial_histories
        if init.ndim == 3:  # [n_history, Nv, C] -> broadcast to K members
            init = init.unsqueeze(0).expand(int(num_aleatory), -1, -1, -1).contiguous()
        return collect_frozen_fgno_rollout_features(
            stage1_model=stage1_model,
            static=family.static,
            geometry=family.geometry,
            query_points=family.query_points,
            boundary_sequence=family.boundary_sequence,
            initial_histories=init,
            aleatory_latents=latents,
            rollout_length=int(family.reference.shape[1]),  # T from the reference ensemble
            n_history=int(n_history),
            feature_source=feature_source,
        )

    return collector


def run_neon_stage2_training(
    *,
    config: Any,
    stage1_checkpoint: Any,
    output_dir: Any,
    data_root: Any,
    load_stage1_fn: LoadStage1Fn,
    build_families_fn: BuildFamiliesFn,
    latent_dim: int,
    n_history: int = 3,
    out_channels: int = 1,
    hidden_channels: int = 64,
    generator: Optional[torch.Generator] = None,
    calibrate_prior: bool = True,
    normalizer_fingerprint: Optional[dict] = None,
    structural_dry_policy: Any = None,
    stage1_alias: str = "best_model",
):
    """End-to-end NEON Stage-2 training orchestration.

    Steps: load + freeze Stage-1, build family splits, build the feature
    collector, probe the frozen feature width, construct the EpiNet, optionally
    auto-calibrate the prior scale, and run the family-level epoch loop saving
    the best checkpoint with structured metadata. Returns the
    :class:`NEONTrainingResult`.
    """
    stage1 = load_stage1_fn(stage1_checkpoint)
    freeze_stage1_model(stage1)

    train_families, val_families = build_families_fn(data_root, config)
    if not train_families:
        raise ValueError("build_families_fn produced no training families.")

    collector = make_feature_collector_from_frozen_model(
        stage1,
        feature_source=config.feature_source,
        n_history=n_history,
        latent_dim=latent_dim,
        generator=generator,
    )

    # Probe the frozen feature width from one family to size the EpiNet.
    probe = collector(train_families[0], num_aleatory=max(2, int(config.k_train)), generator=generator)
    feature_channels = int(probe.features.shape[-1])

    module = build_epinet_from_config(
        config,
        feature_channels=feature_channels,
        out_channels=int(out_channels),
        hidden_channels=int(hidden_channels),
    )

    if calibrate_prior and config.uses_auto_prior_scale:
        z_e = sample_epistemic_indices(
            int(config.m_train), int(config.d_e),
            device=probe.features.device, dtype=probe.features.dtype, generator=generator,
        )
        rmse = base_rmse_from_reference(
            probe.base_prediction,
            train_families[0].reference.unsqueeze(0).to(probe.features.dtype),
        )
        alpha = calibrate_prior_scale(
            module=module, features=probe.features, z_e=z_e,
            base_rmse=rmse, target_fraction=float(config.prior_scale_fraction),
        )
        module.set_prior_scale(alpha)

    optimizer = build_neon_stage2_optimizer(
        module, learning_rate=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    loss_weights = NEONStage2LossWeights(**config.to_loss_weights_dict())
    metadata = build_neon_stage2_metadata(
        stage1_checkpoint_path=str(stage1_checkpoint),
        stage1_checkpoint_alias=stage1_alias,
        normalizer_fingerprint=normalizer_fingerprint or {},
        structural_dry_policy=structural_dry_policy,
        feature_source=config.feature_source,
        dependency=config.dependency,
        d_a=int(latent_dim),
        d_e=int(config.d_e),
        k_train=int(config.k_train),
        m_train=int(config.m_train),
        k_eval=int(config.k_eval),
        m_eval=int(config.m_eval),
        alpha=module.alpha,
        prior_seed=None,
        loss_weights=config.to_loss_weights_dict(),
        optimizer_settings={
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
        },
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "neon_stage2_best.pt"

    return train_neon_stage2_epochs(
        module=module,
        optimizer=optimizer,
        train_families=train_families,
        val_families=val_families,
        feature_collector=collector,
        n_epochs=int(config.n_epochs),
        m_train=int(config.m_train),
        k_train=int(config.k_train),
        d_e=int(config.d_e),
        loss_weights=loss_weights,
        generator=generator,
        stage1_model=stage1,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
    )
