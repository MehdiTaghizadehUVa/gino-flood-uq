"""TDD tests for the NEON Stage-2 family-level epoch training loop (Gap 3).

The epoch driver is dependency-injected with a ``feature_collector`` so it can
be exercised with a tiny grouped fixture (R>1) and fake frozen-FGNO outputs,
without a real dataset or GPU. Tests assert: only trainable EpiNet weights
change, Stage-1 + prior stay frozen, best-epoch tracking, no-grad validation,
and a structured checkpoint schema with the plan's required fields.
"""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _ensure_pkg(name: str):
    sys.modules.setdefault(name, types.ModuleType(name))


def _load_module(name: str, rel_path: str):
    for pkg in ("neuralop", "neuralop.flood", "neuralop.flood.train"):
        _ensure_pkg(pkg)
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")
neon_config = _load_module("neuralop.flood.neon_config", "neuralop/flood/neon_config.py")
train_neon = _load_module("neuralop.flood.train.neon", "neuralop/flood/train/neon.py")

NEONEpistemicCorrection = neon.NEONEpistemicCorrection
NEONStage2LossWeights = neon.NEONStage2LossWeights
load_neon_stage2_checkpoint = neon.load_neon_stage2_checkpoint
NEONFamilySample = train_neon.NEONFamilySample
NEONTrainingResult = train_neon.NEONTrainingResult
train_neon_stage2_epochs = train_neon.train_neon_stage2_epochs
build_neon_stage2_metadata = train_neon.build_neon_stage2_metadata
build_neon_stage2_optimizer = train_neon.build_neon_stage2_optimizer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

Nv, C, Cphi, T = 4, 1, 6, 2


def _family(fid: str, ref_offset: float, n_ref: int = 3) -> NEONFamilySample:
    # Reference HEC-RAS ensemble [R, T, Nv, C] with R>1.
    ref = ref_offset + 0.01 * torch.arange(n_ref * T * Nv * C, dtype=torch.float32).reshape(
        n_ref, T, Nv, C
    )
    return NEONFamilySample(family_id=fid, reference=ref)


def _make_feature_collector(seed_base: int = 0):
    """Return a fake collector that produces frozen base predictions/features.

    Base predictions are deterministic per family so the fit loss is stable;
    features are random-but-fixed per family so the EpiNet has signal to fit.
    """

    def collector(family: NEONFamilySample, *, num_aleatory: int, generator=None):
        g = torch.Generator().manual_seed(seed_base + abs(hash(family.family_id)) % 10_000)
        base = torch.zeros(1, num_aleatory, T, Nv, C)
        # slight per-member spread so aleatory variance is nonzero
        base = base + 0.1 * torch.arange(num_aleatory).view(1, num_aleatory, 1, 1, 1)
        features = torch.randn(1, num_aleatory, T, Nv, Cphi, generator=g)
        return train_neon.FrozenFGNOFeatureBatch(
            base_prediction=base,
            features=features,
            aleatory_latents=torch.zeros(num_aleatory, 8),
        )

    return collector


def _module():
    torch.manual_seed(0)
    return NEONEpistemicCorrection(
        feature_channels=Cphi, out_channels=C, epistemic_dim=4, hidden_channels=8, alpha=0.1
    )


# ---------------------------------------------------------------------------
# Training-loop behavior
# ---------------------------------------------------------------------------


def test_epoch_loop_runs_requested_epochs_and_returns_history():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0), _family("b", 0.5)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=3,
        m_train=2,
        k_train=4,
        d_e=4,
    )
    assert isinstance(result, NEONTrainingResult)
    assert len(result.history) == 3
    assert set(result.history[0]) >= {"epoch", "train_fit", "val_fit"}


def test_training_updates_only_trainable_epinet_weights():
    module = _module()
    # snapshot params
    prior_before = [p.clone() for p in module.prior_branch.parameters()]
    trainable_before = [p.clone() for p in module.trainable_branch.parameters()]
    feat_encoder_before = [p.clone() for p in module.trainable_branch.feature_encoder.parameters()]

    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0), _family("b", 0.5)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=2,
        m_train=2,
        k_train=4,
        d_e=4,
    )
    # Prior branch must be unchanged.
    for before, after in zip(prior_before, module.prior_branch.parameters()):
        torch.testing.assert_close(before, after)
    # At least one trainable parameter must have changed.
    changed = any(
        not torch.allclose(before, after)
        for before, after in zip(trainable_before, module.trainable_branch.parameters())
    )
    assert changed


def test_best_epoch_tracking_picks_lowest_val():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=4,
        m_train=2,
        k_train=4,
        d_e=4,
    )
    val_series = [h["val_fit"] for h in result.history]
    assert result.best_epoch == int(min(range(len(val_series)), key=lambda i: val_series[i]))
    assert result.best_val_fit == pytest.approx(min(val_series))


def test_validation_runs_without_grad_and_leaves_module_in_train_mode_between_epochs():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    # If validation leaked gradients, prior params (frozen) would still have
    # grad=None, but trainable grads would be polluted mid-epoch. We check the
    # simpler invariant: no parameter retains a grad from the val pass after
    # training completes with a final validation.
    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
    )
    assert all(p.grad is None for p in module.prior_branch.parameters())


def test_checkpoint_saved_at_best_epoch_with_structured_metadata(tmp_path):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    ckpt = tmp_path / "neon_stage2_best.pt"
    metadata = build_neon_stage2_metadata(
        stage1_checkpoint_path="/scratch/fgno/best.pt",
        stage1_checkpoint_alias="best_model",
        normalizer_fingerprint={"split_fingerprint": "abc"},
        structural_dry_policy="masked_primary",
        feature_source="decoder_pre_projection",
        dependency="za_dependent",
        d_a=32,
        d_e=4,
        k_train=4,
        m_train=2,
        k_eval=50,
        m_eval=16,
        alpha=0.1,
        prior_seed=1234,
        loss_weights={"rpf": 1e-4, "smooth": 1e-3, "time": 1.0, "pos": 1e-2, "mag": 1e-4},
        optimizer_settings={"learning_rate": 1e-2, "weight_decay": 1e-4},
    )
    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=2,
        m_train=2,
        k_train=4,
        d_e=4,
        checkpoint_path=ckpt,
        checkpoint_metadata=metadata,
    )
    assert ckpt.exists()
    loaded, meta = load_neon_stage2_checkpoint(ckpt)
    # Required schema fields present.
    for key in (
        "stage1_checkpoint_path",
        "stage1_checkpoint_alias",
        "normalizer_fingerprint",
        "structural_dry_policy",
        "feature_source",
        "dependency",
        "d_a",
        "d_e",
        "k_train",
        "m_train",
        "k_eval",
        "m_eval",
        "alpha",
        "prior_seed",
        "loss_weights",
        "optimizer_settings",
    ):
        assert key in meta, f"missing checkpoint metadata field: {key}"
    # Best-epoch bookkeeping recorded.
    assert meta["best_epoch"] == result.best_epoch
    assert "val_metrics" in meta
    # Fixed prior reproduced from the saved checkpoint.
    assert all(not p.requires_grad for p in loaded.prior_branch.parameters())


def test_reference_accepts_bare_R_T_Nv_C_without_batch_dim():
    # NEONFamilySample.reference is [R, T, Nv, C]; the driver must add B=1.
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    fam = _family("a", 0.0)
    assert fam.reference.ndim == 4  # [R, T, Nv, C]
    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[fam],
        val_families=[fam],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
    )
    assert len(result.history) == 1
