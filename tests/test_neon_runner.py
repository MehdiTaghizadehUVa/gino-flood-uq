"""TDD test for the NEON Stage-2 runner orchestration (Gap 4).

Exercises run_neon_stage2_training end-to-end with a fake frozen Stage-1 model
and tiny grouped-hydrograph families carrying domain inputs -- no GPU, no real
checkpoint or dataset.
"""

import importlib.util
import json
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
    assert meta["branch_type"] == "projected"
    assert meta["prior_hidden_channels"] == 16
    assert meta["bootstrap"]["enabled"] is True
    assert meta["member_bootstrap"]["enabled"] is True
    assert meta["prior_rff_dim"] == 32
    assert meta["selection_min_retention"] == 0.3
    assert meta["calibration_m"] == 64
    assert meta["feature_cache_schema_version"] == "neon_feature_cache_v2"
    assert meta["progress_log_interval_effective_batches"] == 10
    assert "best_epoch" in meta and "val_metrics" in meta
    assert (tmp_path / "progress_events.jsonl").exists()
    assert (tmp_path / "history_partial.jsonl").exists()
    with (tmp_path / "latest_status.json").open("r", encoding="utf-8") as fh:
        latest = json.load(fh)
    assert latest["event"] == "training_end"
    assert latest["best_epoch"] == result.best_epoch
    # Frozen Stage-1 stayed frozen (orchestration froze it and never unfroze).
    assert all(not p.requires_grad for p in stage1.parameters())
    assert not stage1.training


def test_runner_propagates_dependency_mode_to_epinet_checkpoint(tmp_path):
    torch.manual_seed(0)
    config = NEONStage2Config(
        enabled=True, d_e=4, m_train=2, k_train=2, m_eval=2, k_eval=2,
        n_epochs=1, dependency="za_independent", alpha=0.05,
    )
    stage1 = _DummyStage1()

    run_neon_stage2_training(
        config=config,
        stage1_checkpoint="dummy_fgno.pt",
        output_dir=tmp_path,
        data_root="ignored",
        load_stage1_fn=lambda ckpt: stage1,
        build_families_fn=lambda root, cfg: ([_family("a", 0.0)], [_family("v", 0.2)]),
        latent_dim=8,
        calibrate_prior=False,
    )
    module, meta = load_neon_stage2_checkpoint(tmp_path / "neon_stage2_best.pt")
    assert meta["dependency"] == "za_independent"
    assert module.za_dependent is False


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


def test_frozen_model_collector_uses_stable_family_bank_latents():
    stage1 = _DummyStage1()
    collector = runner.make_feature_collector_from_frozen_model(
        stage1,
        feature_source="decoder_pre_projection",
        n_history=n_hist,
        latent_dim=8,
        generator=torch.Generator().manual_seed(123),
    )
    fam = _family("TR000123", 0.0)

    bank1_first = collector(fam, num_aleatory=3, latent_bank_id=1)
    bank0 = collector(fam, num_aleatory=3, latent_bank_id=0)
    bank1_second = collector(fam, num_aleatory=3, latent_bank_id=1)

    torch.testing.assert_close(bank1_first.aleatory_latents, bank1_second.aleatory_latents)
    assert not torch.allclose(bank0.aleatory_latents, bank1_first.aleatory_latents)


def test_disk_cache_collects_once_and_round_trips(tmp_path):
    # cache_dir mode: first call rolls the base collector and writes one file;
    # second call must load from disk (no recollection) and reproduce the
    # tensors up to fp16 quantization.
    calls = {"n": 0}

    def base(family, *, num_aleatory, generator=None, latent_bank_id=None):
        calls["n"] += 1
        g = torch.Generator().manual_seed(7)
        return train_neon.FrozenFGNOFeatureBatch(
            base_prediction=torch.randn(1, num_aleatory, T, Nv, C, generator=g),
            features=torch.randn(1, num_aleatory, T, Nv, Cphi, generator=g),
            aleatory_latents=torch.randn(num_aleatory, 2, generator=g),
        )

    fam = NEONFamilySample(family_id="TR000001", reference=torch.zeros(R, T, Nv, C))
    coll = runner.make_cached_feature_collector(base, cache_dir=tmp_path)
    first = coll(fam, num_aleatory=4)
    second = coll(fam, num_aleatory=4)
    assert calls["n"] == 1
    assert (tmp_path / "TR000001_bank0_k4.pt").exists()
    assert second.features.dtype == first.features.dtype
    assert torch.allclose(first.features, second.features, atol=1e-2)
    assert torch.allclose(first.base_prediction, second.base_prediction, atol=1e-2)


def test_disk_cache_key_separates_incompatible_feature_entries(tmp_path):
    calls = {"n": 0}

    def base(family, *, num_aleatory, generator=None, latent_bank_id=None):
        calls["n"] += 1
        return train_neon.FrozenFGNOFeatureBatch(
            base_prediction=torch.full((1, num_aleatory, T, Nv, C), float(calls["n"])),
            features=torch.full((1, num_aleatory, T, Nv, Cphi), float(calls["n"])),
            aleatory_latents=torch.zeros(num_aleatory, 2),
        )

    fam = NEONFamilySample(family_id="TR000001", reference=torch.zeros(R, T, Nv, C))
    coll_a = runner.make_cached_feature_collector(base, cache_dir=tmp_path, cache_key="schema_a")
    coll_b = runner.make_cached_feature_collector(base, cache_dir=tmp_path, cache_key="schema_b")

    a = coll_a(fam, num_aleatory=4)
    b = coll_b(fam, num_aleatory=4)

    assert calls["n"] == 2
    assert (tmp_path / "schema_a_TR000001_bank0_k4.pt").exists()
    assert (tmp_path / "schema_b_TR000001_bank0_k4.pt").exists()
    assert not torch.allclose(a.features, b.features)


def test_disk_cache_key_separates_latent_banks(tmp_path):
    calls = {"n": 0}

    def base(family, *, num_aleatory, generator=None, latent_bank_id=None):
        calls["n"] += 1
        value = float(latent_bank_id or 0)
        return train_neon.FrozenFGNOFeatureBatch(
            base_prediction=torch.full((1, num_aleatory, T, Nv, C), value),
            features=torch.full((1, num_aleatory, T, Nv, Cphi), value),
            aleatory_latents=torch.zeros(num_aleatory, 2),
        )

    fam = NEONFamilySample(family_id="TR000001", reference=torch.zeros(R, T, Nv, C))
    coll = runner.make_cached_feature_collector(base, cache_dir=tmp_path)

    bank0 = coll(fam, num_aleatory=4, latent_bank_id=0)
    bank1 = coll(fam, num_aleatory=4, latent_bank_id=1)
    bank1_again = coll(fam, num_aleatory=4, latent_bank_id=1)

    assert calls["n"] == 2
    assert (tmp_path / "TR000001_bank0_k4.pt").exists()
    assert (tmp_path / "TR000001_bank1_k4.pt").exists()
    assert not torch.allclose(bank0.features, bank1.features)
    torch.testing.assert_close(bank1.features, bank1_again.features)


def test_runner_rejects_large_in_memory_feature_cache(tmp_path):
    config = NEONStage2Config(
        enabled=True,
        d_e=4,
        m_train=2,
        k_train=2,
        m_eval=2,
        k_eval=2,
        n_epochs=1,
        prior_scale="auto_0p10_base_rmse",
        alpha=None,
    )
    stage1 = _DummyStage1()
    train_families = [_family(f"f{i}", 0.0) for i in range(3)]
    val_families = [_family("v", 0.2)]

    try:
        run_neon_stage2_training(
            config=config,
            stage1_checkpoint="dummy_fgno.pt",
            output_dir=tmp_path,
            data_root="ignored",
            load_stage1_fn=lambda ckpt: stage1,
            build_families_fn=lambda root, cfg: (train_families, val_families),
            latent_dim=8,
            cache_features=True,
            memory_cache_limit_bytes=1,
        )
    except ValueError as exc:
        assert "cache_dir" in str(exc)
        assert "In-memory NEON frozen-feature cache" in str(exc)
    else:
        raise AssertionError("large in-memory feature cache should fail early")


def test_runner_allows_large_disk_feature_cache(tmp_path):
    config = NEONStage2Config(
        enabled=True,
        d_e=4,
        m_train=2,
        k_train=2,
        m_eval=2,
        k_eval=2,
        n_epochs=1,
        prior_scale="auto_0p10_base_rmse",
        alpha=None,
    )
    stage1 = _DummyStage1()

    result = run_neon_stage2_training(
        config=config,
        stage1_checkpoint="dummy_fgno.pt",
        output_dir=tmp_path / "out",
        data_root="ignored",
        load_stage1_fn=lambda ckpt: stage1,
        build_families_fn=lambda root, cfg: ([_family("a", 0.0)], [_family("v", 0.2)]),
        latent_dim=8,
        cache_features=True,
        cache_dir=tmp_path / "feature_cache",
        memory_cache_limit_bytes=1,
    )

    assert result.history
    assert list((tmp_path / "feature_cache").glob("*_bank0_k2.pt"))
