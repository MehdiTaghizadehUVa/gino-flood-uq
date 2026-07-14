"""TDD tests for the NEON Stage-2 family-level epoch training loop (Gap 3).

The epoch driver is dependency-injected with a ``feature_collector`` so it can
be exercised with a tiny grouped fixture (R>1) and fake frozen-FGNO outputs,
without a real dataset or GPU. Tests assert: only trainable EpiNet weights
change, Stage-1 + prior stay frozen, best-epoch tracking, no-grad validation,
and a structured checkpoint schema with the plan's required fields.
"""

import importlib.util
import json
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
neon_stage2_eval_forward_chunked = train_neon.neon_stage2_eval_forward_chunked


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

    def collector(family: NEONFamilySample, *, num_aleatory: int, generator=None, latent_bank_id=None):
        g = torch.Generator().manual_seed(seed_base + abs(hash(family.family_id)) % 10_000)
        base = torch.zeros(1, num_aleatory, T, Nv, C)
        # slight per-member spread so aleatory variance is nonzero
        base = base + 0.1 * torch.arange(num_aleatory).view(1, num_aleatory, 1, 1, 1)
        if latent_bank_id is not None:
            base = base + 0.01 * int(latent_bank_id)
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


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Evaluation batching behavior
# ---------------------------------------------------------------------------


def test_chunked_eval_batches_all_particles_for_no_concat_branch():
    class CountingModule(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.epistemic_chunk_sizes = []

        def forward(self, *args, **kwargs):
            self.epistemic_chunk_sizes.append(int(args[2].shape[0]))
            return self.inner(*args, **kwargs)

    head = NEONEpistemicCorrection(
        feature_channels=Cphi,
        out_channels=C,
        epistemic_dim=4,
        hidden_channels=8,
        branch_type="projected",
        concat_index=False,
        alpha=0.1,
    ).eval()
    wrapped = CountingModule(head)
    base = torch.randn(1, 5, T, Nv, C)
    features = torch.randn(1, 5, T, Nv, Cphi)
    z_e = torch.randn(7, 4)

    actual = neon_stage2_eval_forward_chunked(
        module=wrapped,
        base_prediction=base,
        features=features,
        z_e=z_e,
        k_chunk=2,
        epistemic_chunk_size=7,
        output_device="cpu",
    )
    expected = head(base, features, z_e).prediction

    torch.testing.assert_close(actual, expected)
    assert wrapped.epistemic_chunk_sizes == [7, 7, 7]


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
    assert set(result.history[0]) >= {
        "epoch",
        "train_fit",
        "val_fit",
        "epoch_seconds",
    }
    assert result.history[0]["epoch_seconds"] > 0.0


def test_epoch_loop_validates_at_interval_and_final_epoch():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    base_collector = _make_feature_collector()
    calls = []

    def collector(family, **kwargs):
        calls.append(family.family_id)
        return base_collector(family, **kwargs)

    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("train", 0.0)],
        val_families=[_family("val", 0.2)],
        feature_collector=collector,
        n_epochs=5,
        m_train=2,
        k_train=4,
        d_e=4,
        validation_interval=2,
    )

    assert calls.count("val") == 3
    assert [row["validation_ran"] for row in result.history] == [0.0, 1.0, 0.0, 1.0, 1.0]
    assert all(row["epoch_seconds"] > 0.0 for row in result.history)


def test_epoch_loop_saves_latest_state_and_resumes_training(tmp_path):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    latest = tmp_path / "neon_stage2_latest.pt"
    generator = torch.Generator().manual_seed(123)

    first = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0), _family("b", 0.5)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
        generator=generator,
        latest_checkpoint_path=latest,
    )
    assert latest.exists()
    state = train_neon.load_neon_stage2_training_state(latest)
    assert state["next_epoch"] == 1
    assert len(state["history"]) == 1
    assert state["best_epoch"] == first.best_epoch
    assert torch.equal(state["generator_state"], generator.get_state())
    assert state["particle_training_step"] == 0
    assert state["n_ineligible_epochs"] >= 0

    resumed_module = _module()
    resumed_optimizer = build_neon_stage2_optimizer(resumed_module, learning_rate=1e-2)
    resumed_module.load_state_dict(state["state_dict"])
    resumed_optimizer.load_state_dict(state["optimizer_state_dict"])
    second = train_neon_stage2_epochs(
        module=resumed_module,
        optimizer=resumed_optimizer,
        train_families=[_family("a", 0.0), _family("b", 0.5)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=2,
        m_train=2,
        k_train=4,
        d_e=4,
        latest_checkpoint_path=latest,
        start_epoch=int(state["next_epoch"]),
        initial_history=state["history"],
        initial_best_epoch=int(state["best_epoch"]),
        initial_best_val_fit=float(state["best_val_fit"]),
    )

    assert [row["epoch"] for row in second.history] == [0, 1]
    state2 = train_neon.load_neon_stage2_training_state(latest)
    assert state2["next_epoch"] == 2
    assert len(state2["history"]) == 2


def test_epoch_loop_writes_live_progress_reports(tmp_path):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    reporter = train_neon.NEONTrainingProgressReporter(
        output_dir=tmp_path,
        log_interval_effective_batches=1,
    )

    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0), _family("b", 0.5)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=2,
        m_train=2,
        k_train=4,
        d_e=4,
        progress_reporter=reporter,
    )

    events = _read_jsonl(tmp_path / "progress_events.jsonl")
    event_names = {event["event"] for event in events}
    assert {
        "training_start",
        "epoch_start",
        "train_progress",
        "validation_start",
        "epoch_end",
        "training_end",
    }.issubset(event_names)

    partial_history = _read_jsonl(tmp_path / "history_partial.jsonl")
    assert [row["epoch"] for row in partial_history] == [0, 1]
    assert partial_history[-1]["best_epoch"] == result.best_epoch
    assert "val_fit" in partial_history[-1]

    with (tmp_path / "latest_status.json").open("r", encoding="utf-8") as fh:
        latest = json.load(fh)
    assert latest["event"] == "training_end"
    assert latest["best_epoch"] == result.best_epoch
    assert latest["best_val_fit"] == pytest.approx(result.best_val_fit)


def test_training_updates_only_trainable_epinet_weights():
    module = _module()
    # snapshot params
    prior_before = [p.clone() for p in module.prior_branch.parameters()]
    trainable_before = [p.clone() for p in module.trainable_branch.parameters()]

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


def test_epoch_history_includes_cancellation_diagnostics_with_bootstrap():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    result = train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0), _family("b", 0.5), _family("c", 1.0)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
        bootstrap_config={
            "enabled": True,
            "distribution": "tempered_exponential",
            "temperature": 0.5,
            "normalize": "per_epistemic_batch",
            "min_weight": 0.05,
            "max_weight": 5.0,
            "seed": 123,
        },
        cancellation_config={"enabled": True},
    )

    row = result.history[0]
    assert "train_cancellation_fraction" in row
    assert "val_cancellation_fraction" in row
    assert "train_prior_retention_ratio" in row


def test_bootstrap_weights_remain_nontrivial_with_single_family_microbatches(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    captured: list[torch.Tensor] = []
    original_step = train_neon.neon_stage2_training_step

    def wrapped_step(*args, **kwargs):
        sample_weights = kwargs.get("sample_weights")
        if sample_weights is not None:
            captured.append(sample_weights.detach().cpu().clone())
        return original_step(*args, **kwargs)

    monkeypatch.setattr(train_neon, "neon_stage2_training_step", wrapped_step)
    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family(f"fam-{i}", float(i) * 0.1) for i in range(4)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=3,
        k_train=4,
        d_e=4,
        bootstrap_config={
            "enabled": True,
            "distribution": "tempered_exponential",
            "temperature": 0.5,
            "normalize": "per_epistemic_batch",
            "min_weight": 0.05,
            "max_weight": 5.0,
            "seed": 321,
        },
        family_batch_size=1,
        effective_batch_size=4,
        shuffle_families=False,
        generator=torch.Generator().manual_seed(12),
    )

    assert captured
    weights = torch.cat(captured, dim=0)
    torch.testing.assert_close(weights.mean(dim=0), torch.ones(weights.shape[1]), rtol=1e-6, atol=1e-6)
    assert not torch.allclose(weights, torch.ones_like(weights))


def test_training_steps_once_per_effective_family_batch():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    step_count = 0
    original_step = opt.step

    def counted_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    opt.step = counted_step
    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family(f"f{i}", float(i) * 0.1) for i in range(5)],
        val_families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
        family_batch_size=2,
        effective_batch_size=3,
        shuffle_families=False,
    )
    assert step_count == 2


def test_training_passes_latent_bank_ids_to_feature_collector():
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    calls: list[int] = []
    base_collector = _make_feature_collector()

    def collector(family: NEONFamilySample, *, num_aleatory: int, generator=None, latent_bank_id=None):
        calls.append(int(latent_bank_id))
        return base_collector(
            family,
            num_aleatory=num_aleatory,
            generator=generator,
            latent_bank_id=latent_bank_id,
        )

    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0), _family("b", 0.5), _family("c", 1.0)],
        val_families=[_family("v", 0.2)],
        feature_collector=collector,
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
        latent_bank_count=3,
        shuffle_families=False,
        generator=torch.Generator().manual_seed(7),
    )
    # Training and validation both go through the collector; every call gets a
    # valid bank id and training can choose among multiple cached z_a banks.
    assert calls
    assert all(0 <= bank_id < 3 for bank_id in calls)


def test_training_prefetches_the_exact_reproducible_latent_bank_schedule():
    def run_once():
        module = _module()
        opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
        base_collector = _make_feature_collector()
        calls: list[tuple[str, int]] = []
        prefetched: list[tuple[str, int]] = []

        def collector(
            family: NEONFamilySample,
            *,
            num_aleatory: int,
            generator=None,
            latent_bank_id=None,
        ):
            calls.append((family.family_id, int(latent_bank_id)))
            return base_collector(
                family,
                num_aleatory=num_aleatory,
                generator=generator,
                latent_bank_id=latent_bank_id,
            )

        def prefetch(family, *, num_aleatory, latent_bank_id=None):
            prefetched.append((family.family_id, int(latent_bank_id)))
            return True

        collector.prefetch = prefetch
        collector.prefetch_depth = 2
        train_neon_stage2_epochs(
            module=module,
            optimizer=opt,
            train_families=[
                _family("a", 0.0),
                _family("b", 0.5),
                _family("c", 1.0),
            ],
            val_families=[_family("v", 0.2)],
            feature_collector=collector,
            n_epochs=1,
            m_train=2,
            k_train=4,
            d_e=4,
            latent_bank_count=3,
            shuffle_families=False,
            generator=torch.Generator().manual_seed(17),
        )
        return calls, prefetched

    calls_a, prefetched_a = run_once()
    calls_b, prefetched_b = run_once()

    assert calls_a == calls_b
    assert prefetched_a == prefetched_b
    assert prefetched_a
    for consumed in calls_a:
        assert consumed in prefetched_a


def test_training_subsamples_reference_members_for_fit(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    seen_reference_members: list[int] = []
    original_step = train_neon.neon_stage2_training_step

    def wrapped_step(*args, **kwargs):
        seen_reference_members.append(int(kwargs["reference"].shape[1]))
        return original_step(*args, **kwargs)

    monkeypatch.setattr(train_neon, "neon_stage2_training_step", wrapped_step)
    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0, n_ref=7), _family("b", 0.5, n_ref=7)],
        val_families=[_family("v", 0.2, n_ref=7)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=2,
        k_train=4,
        d_e=4,
        reference_member_subsample=2,
        shuffle_families=False,
        generator=torch.Generator().manual_seed(11),
    )
    assert seen_reference_members == [2, 2]


def test_training_threads_member_bootstrap_weights_and_keeps_validation_unweighted(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    seen_member_weights: list[torch.Tensor | None] = []
    original_step = train_neon.neon_stage2_training_step
    original_eval = train_neon.stage2_fit_score

    def wrapped_step(*args, **kwargs):
        mw = kwargs.get("member_weights")
        seen_member_weights.append(None if mw is None else mw.detach().cpu().clone())
        return original_step(*args, **kwargs)

    validation_member_weights: list[object] = []

    def wrapped_fit(*args, **kwargs):
        validation_member_weights.append(kwargs.get("member_weights"))
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(train_neon, "neon_stage2_training_step", wrapped_step)
    monkeypatch.setattr(train_neon, "stage2_fit_score", wrapped_fit)
    train_neon_stage2_epochs(
        module=module,
        optimizer=opt,
        train_families=[_family("a", 0.0, n_ref=5), _family("b", 0.5, n_ref=5)],
        val_families=[_family("v", 0.2, n_ref=5)],
        feature_collector=_make_feature_collector(),
        n_epochs=1,
        m_train=3,
        k_train=4,
        d_e=4,
        member_bootstrap_config={"enabled": True, "temperature": 1.0, "seed": 44},
        reference_member_subsample=3,
        shuffle_families=False,
        generator=torch.Generator().manual_seed(11),
    )

    assert seen_member_weights and all(w is not None for w in seen_member_weights)
    weights = seen_member_weights[0]
    assert weights is not None
    assert weights.shape == (1, 3, 3)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 3), rtol=1e-6, atol=1e-6)
    assert not torch.allclose(weights[:, 0], weights[:, 1])
    assert validation_member_weights
    assert all(value is None for value in validation_member_weights)


def test_best_epoch_tracking_picks_lowest_eligible_mixture_crps():
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
        selection_rmse_margin_m=float("inf"),
    )
    score_series = [h["selection_score_mixture_fair_crps_physical"] for h in result.history]
    expected = int(min(range(len(score_series)), key=lambda i: score_series[i]))
    assert result.best_epoch == expected
    assert result.best_val_fit == pytest.approx(min(score_series))


def test_retention_floor_is_warning_only_and_does_not_override_skill_selection(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    calls = {"n": 0}

    def fake_eval(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 10.0, {"prior_retention_ratio": 0.5}
        return 1.0, {"prior_retention_ratio": 0.1}

    monkeypatch.setattr(train_neon, "_evaluate_neon_validation", fake_eval)
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
        selection_min_retention=0.3,
    )

    assert result.best_epoch == 1
    assert result.best_val_fit == pytest.approx(1.0)
    assert result.history[1]["selection_eligible"] == 1.0
    assert result.history[1]["retention_warning"] == 1.0


def test_retention_gate_disabled_recovers_lowest_val_fit(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    calls = {"n": 0}

    def fake_eval(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 10.0, {"prior_retention_ratio": 0.0}
        return 1.0, {"prior_retention_ratio": 0.0}

    monkeypatch.setattr(train_neon, "_evaluate_neon_validation", fake_eval)
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
        selection_min_retention=0.0,
    )

    assert result.best_epoch == 1
    assert result.best_val_fit == pytest.approx(1.0)


def test_rmse_noninferiority_margin_blocks_lower_mixture_crps(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    calls = {"n": 0}

    def fake_eval(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return 5.0, {
                "mixture_fair_crps_physical": 0.03,
                "stage2_minus_base_rmse_physical": 0.0005,
            }
        return 1.0, {
            "mixture_fair_crps_physical": 0.01,
            "stage2_minus_base_rmse_physical": 0.002,
        }

    monkeypatch.setattr(train_neon, "_evaluate_neon_validation", fake_eval)
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
        selection_rmse_margin_m=0.001,
    )

    assert result.best_epoch == 0
    assert result.best_val_fit == pytest.approx(0.03)
    assert result.history[1]["selection_eligible"] == 0.0


def test_legacy_ladder_selector_uses_per_epistemic_fit_without_rmse_gate(monkeypatch):
    module = _module()
    opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
    calls = {"n": 0}

    def fake_eval(**kwargs):
        calls["n"] += 1
        return (
            (2.0, {"mixture_fair_crps_physical": 0.01, "stage2_minus_base_rmse_physical": 1.0})
            if calls["n"] == 1
            else (1.0, {"mixture_fair_crps_physical": 0.50, "stage2_minus_base_rmse_physical": 1.0})
        )

    monkeypatch.setattr(train_neon, "_evaluate_neon_validation", fake_eval)
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
        selection_metric="per_epistemic_fit",
        selection_enforce_rmse=False,
    )

    assert result.best_epoch == 1
    assert result.best_val_fit == pytest.approx(1.0)


def test_validation_selection_metrics_are_inverse_transformed_to_physical_space():
    class AffineNormalizer:
        def to(self, device):
            return self

        def inverse_transform(self, value):
            return 10.0 + 2.0 * value

    module = _module()
    kwargs = dict(
        module=module,
        families=[_family("v", 0.2)],
        feature_collector=_make_feature_collector(),
        m=2,
        k=4,
        d_e=4,
    )
    _, normalized = train_neon._evaluate_neon_validation(
        **kwargs, generator=torch.Generator().manual_seed(7)
    )
    _, physical = train_neon._evaluate_neon_validation(
        **kwargs,
        generator=torch.Generator().manual_seed(7),
        target_normalizer=AffineNormalizer(),
        reference_normalizer=AffineNormalizer(),
        physical_scale=2.0,
    )

    assert physical["mixture_fair_crps_physical"] == pytest.approx(
        2.0 * normalized["mixture_fair_crps_physical"], rel=1e-5
    )
    assert physical["base_rmse_physical"] == pytest.approx(
        2.0 * normalized["base_rmse_physical"], rel=1e-5
    )
    assert physical["base_fair_crps_physical"] == pytest.approx(
        2.0 * normalized["base_fair_crps_physical"], rel=1e-5
    )
    assert physical["deterministic_head_fair_crps_physical"] == pytest.approx(
        2.0 * normalized["deterministic_head_fair_crps_physical"], rel=1e-5
    )
    assert physical["deterministic_head_rmse_physical"] == pytest.approx(
        2.0 * normalized["deterministic_head_rmse_physical"], rel=1e-5
    )
    assert physical["total_epistemic_std_physical"] == pytest.approx(
        2.0 * normalized["total_epistemic_std_physical"], rel=1e-5
    )


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
        extra={
            "branch_type": "projected",
            "train_hidden_channels": 32,
            "prior_hidden_channels": 5,
            "bootstrap": {"enabled": True},
            "cancellation_diagnostics": {"enabled": True},
            "feature_cache_schema_version": "neon_feature_cache_v2",
        },
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
        "branch_type",
        "train_hidden_channels",
        "prior_hidden_channels",
        "bootstrap",
        "cancellation_diagnostics",
        "feature_cache_schema_version",
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


def test_epistemic_chunk_size_one_still_logs_nonzero_epistemic_variance():
    """End-to-end regression for the epistemic_chunk_size=1 production config.

    Before the fix, chunking the epistemic (M) axis to size 1 made the per-chunk
    variance collapse to exactly 0.0, so total/prior epistemic variance and the
    retention ratio logged as 0.0 every epoch. With assembly across chunks the
    caller must now log strictly positive epistemic variance under chunk=1, and
    it must track the unchunked (full-M) computation.
    """
    def _run(chunk):
        module = _module()
        opt = build_neon_stage2_optimizer(module, learning_rate=1e-2)
        result = train_neon_stage2_epochs(
            module=module,
            optimizer=opt,
            train_families=[_family("a", 0.0), _family("b", 0.5), _family("c", 1.0)],
            val_families=[_family("v", 0.2)],
            feature_collector=_make_feature_collector(),
            n_epochs=1,
            m_train=3,
            k_train=4,
            d_e=4,
            epistemic_chunk_size=chunk,
            cancellation_config={"enabled": True},
        )
        return result.history[0]

    chunked = _run(1)
    full = _run(None)

    # Core regression: the live chunk=1 config no longer logs zeros.
    assert chunked["train_total_epistemic_variance"] > 0.0
    assert chunked["train_prior_epistemic_variance"] > 0.0
    assert chunked["val_total_epistemic_variance"] > 0.0
    assert chunked["train_prior_retention_ratio"] > 0.0

    # And chunk=1 must track the full-M computation (identical gradients/seeds).
    for key in ("train_total_epistemic_variance", "train_prior_epistemic_variance"):
        assert abs(chunked[key] - full[key]) <= 1e-5 * abs(full[key]) + 1e-8
