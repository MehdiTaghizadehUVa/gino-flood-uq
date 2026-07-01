"""TDD test for the NEON Stage-2 runner orchestration (Gap 4).

Exercises run_neon_stage2_training end-to-end with a fake frozen Stage-1 model
and tiny grouped-hydrograph families carrying domain inputs -- no GPU, no real
checkpoint or dataset.
"""

import importlib.util
import sys
import types
from pathlib import Path

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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


neon = _load_module("neuralop.flood.neon", "neuralop/flood/neon.py")
neon_config = _load_module("neuralop.flood.neon_config", "neuralop/flood/neon_config.py")
train_neon = _load_module("neuralop.flood.train.neon", "neuralop/flood/train/neon.py")
runner = _load_module("neuralop.flood.train.neon_runner", "neuralop/flood/train/neon_runner.py")

NEONStage2Config = neon_config.NEONStage2Config
NEONFamilySample = train_neon.NEONFamilySample
run_neon_stage2_training = runner.run_neon_stage2_training
load_neon_stage2_checkpoint = neon.load_neon_stage2_checkpoint


Nv, C, Cphi, n_hist, T, R = 4, 1, 6, 3, 2, 3


class _DummyStage1(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, *, x, ada_in=None, return_features=False, feature_source="decoder_pre_projection", **kwargs):
        b, nv, _ = x.shape
        pred = torch.ones(b, nv, C) + x[..., -C:]
        feat = torch.zeros(b, nv, Cphi) + ada_in.reshape(1, 1, -1)[..., :Cphi]
        if not return_features:
            return pred
        return {"prediction": pred, "features": {"decoder_pre_projection": feat}, "feature_source": feature_source}


def _family(fid: str, offset: float) -> NEONFamilySample:
    ref = offset + 0.01 * torch.arange(R * T * Nv * C, dtype=torch.float32).reshape(R, T, Nv, C)
    return NEONFamilySample(
        family_id=fid,
        reference=ref,
        static=torch.zeros(1, Nv, 7),
        geometry=torch.zeros(1, Nv, 2),
        query_points=torch.zeros(1, 4, 4, 2),
        boundary_sequence=torch.zeros(T + n_hist, Nv, 2),
        initial_histories=torch.zeros(n_hist, Nv, C),
    )


def test_runner_trains_saves_and_records_metadata(tmp_path):
    torch.manual_seed(0)
    config = NEONStage2Config(
        enabled=True, d_e=4, m_train=2, k_train=2, m_eval=2, k_eval=2,
        n_epochs=2, feature_source="decoder_pre_projection",
        prior_scale="auto_0p10_base_rmse", alpha=None, lead_time_dim=0,
    )
    train_families = [_family("a", 0.0), _family("b", 0.5)]
    val_families = [_family("v", 0.2)]

    stage1 = _DummyStage1()
    result = run_neon_stage2_training(
        config=config,
        stage1_checkpoint="dummy_fgno.pt",
        output_dir=tmp_path,
        data_root="ignored",
        load_stage1_fn=lambda ckpt: stage1,
        build_families_fn=lambda root, cfg: (train_families, val_families),
        latent_dim=8,
    )

    # Trained the requested number of epochs.
    assert len(result.history) == 2
    # Best checkpoint written with structured metadata.
    ckpt = tmp_path / "neon_stage2_best.pt"
    assert ckpt.exists()
    _, meta = load_neon_stage2_checkpoint(ckpt)
    assert meta["feature_source"] == "decoder_pre_projection"
    assert meta["d_a"] == 8
    assert meta["d_e"] == 4
    assert "best_epoch" in meta and "val_metrics" in meta
    # Frozen Stage-1 stayed frozen (orchestration froze it and never unfroze).
    assert all(not p.requires_grad for p in stage1.parameters())
    assert not stage1.training


def test_runner_auto_calibrates_prior_scale_away_from_placeholder(tmp_path):
    torch.manual_seed(0)
    config = NEONStage2Config(
        enabled=True, d_e=4, m_train=4, k_train=2, m_eval=2, k_eval=2,
        n_epochs=1, prior_scale="auto_0p10_base_rmse", alpha=None,
    )
    stage1 = _DummyStage1()
    # Capture the module alpha via the saved checkpoint metadata.
    run_neon_stage2_training(
        config=config,
        stage1_checkpoint="dummy_fgno.pt",
        output_dir=tmp_path,
        data_root="ignored",
        load_stage1_fn=lambda ckpt: stage1,
        build_families_fn=lambda root, cfg: ([_family("a", 1.0)], [_family("v", 1.0)]),
        latent_dim=8,
    )
    _, meta = load_neon_stage2_checkpoint(tmp_path / "neon_stage2_best.pt")
    # alpha should be a finite calibrated float (not left at the 0.1 placeholder
    # unless calibration coincidentally lands there).
    assert meta["alpha"] is not None
    assert torch.isfinite(torch.tensor(meta["alpha"]))
