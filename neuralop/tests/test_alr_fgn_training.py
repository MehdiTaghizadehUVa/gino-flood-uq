import torch
from torch.utils.data import Dataset

from neuralop.flood.data.normalization_impl import NormalizedDatasetOnTheFly
from neuralop.flood.data.reference_dispersion import ReferenceDispersionTable
from neuralop.flood.losses import FloodMaskedCRPSLoss
from neuralop.flood.train.alr_fgn import (
    DirichletFamilyBootstrap,
    AnchoredLowRankFGNTrainer,
    clamp_nested_feedback,
    make_nested_particle_batch,
    particle_bootstrap_crps,
    particle_bootstrap_mean_mse,
    split_alr_family_indices,
    update_nested_history,
    residual_decomposition_components,
    residual_centering_monitor,
    PhysicalRMSE,
)
from neuralop.losses.probabilistic_losses import CRPSLoss


def test_dirichlet_family_bootstrap_is_persistent_and_checkpointable():
    family_ids = ["F001", "F002", "F003", "F004"]
    bootstrap = DirichletFamilyBootstrap(
        family_ids=family_ids,
        num_particles=3,
        seed=17,
    )

    selected = bootstrap.weights_for(["F003", "F001"])
    assert selected.shape == (3, 2)
    assert torch.all(selected > 0)
    torch.testing.assert_close(bootstrap.weights.mean(dim=1), torch.ones(3))

    restored = DirichletFamilyBootstrap(
        family_ids=family_ids,
        num_particles=3,
        seed=999,
    )
    restored.load_state_dict(bootstrap.state_dict())

    assert restored.family_ids == family_ids
    assert restored.seed == 17
    torch.testing.assert_close(restored.weights_for(["F003", "F001"]), selected)


class _OneFamilyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        del index
        return {
            "geometry": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "static": torch.zeros(2, 1),
            "boundary": torch.zeros(1, 2, 1),
            "dynamic": torch.zeros(1, 2, 1),
            "target": torch.zeros(2, 1),
            "run_id": "Flood_coastal_TR000001_sim03",
            "family_id": "Flood_coastal_TR000001",
        }


def test_normalized_dataset_preserves_family_identity_metadata():
    wrapped = NormalizedDatasetOnTheFly(
        _OneFamilyDataset(),
        normalizers={},
        query_res=[2, 2],
    )

    sample = wrapped[0]

    assert sample["run_id"] == "Flood_coastal_TR000001_sim03"
    assert sample["family_id"] == "Flood_coastal_TR000001"


def test_nested_particle_batch_uses_common_aleatory_draws_and_stable_ids():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    latent_bank = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

    nested = make_nested_particle_batch(
        x,
        latent_bank=latent_bank,
        num_particles=3,
    )

    assert nested.values.shape == (12, 3)
    assert nested.latents.shape == (12, 4)
    assert nested.particle_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert nested.aleatory_ids.tolist() == [0, 0, 1, 1] * 3
    for particle in range(1, 3):
        start = particle * 4
        torch.testing.assert_close(nested.latents[start : start + 4], nested.latents[:4])


def test_particle_bootstrap_crps_scores_aleatory_members_within_each_particle():
    predictions = torch.tensor(
        [
            [[0.0, 2.0], [2.0, 4.0]],
            [[1.0, 3.0], [5.0, 7.0]],
        ]
    ).unsqueeze(-1).unsqueeze(-1)  # [M=2, K=2, B=2, N=1, C=1]
    target = torch.tensor([[[1.0]], [[3.0]]])
    family_weights = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    loss_fn = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=2, reduction="mean"),
    )

    result = particle_bootstrap_crps(
        predictions,
        target,
        family_weights=family_weights,
        loss_fn=loss_fn,
    )

    expected_by_particle = []
    for particle in range(2):
        weights = family_weights[particle].view(2, 1, 1).expand_as(target)
        expected_by_particle.append(
            loss_fn(predictions[particle], target, spatial_weights=weights)
        )
    expected = torch.stack(expected_by_particle)
    torch.testing.assert_close(result.per_particle, expected)
    torch.testing.assert_close(result.mean, expected.mean())


class _AffineNormalizer:
    def transform(self, values):
        return (values - 10.0) / 2.0

    def inverse_transform(self, values):
        return values * 2.0 + 10.0


def test_nested_feedback_is_clamped_before_each_particle_member_history_update():
    physical = torch.tensor(
        [
            [[[-1.0], [5.0], [3.0]]],
            [[[2.0], [4.0], [6.0]]],
            [[[1.0], [7.0], [8.0]]],
            [[[9.0], [2.0], [4.0]]],
        ]
    ).reshape(2, 2, 1, 3, 1)
    predicted = _AffineNormalizer().transform(physical)
    clamped = clamp_nested_feedback(
        predicted,
        structural_dry_mask=torch.tensor([[False, True, False]]),
        target_normalizer=_AffineNormalizer(),
        water_depth_index=0,
    )
    history = torch.full((2, 2, 1, 2, 3, 1), -99.0)

    updated = update_nested_history(history, clamped)

    clamped_physical = _AffineNormalizer().inverse_transform(clamped)
    assert torch.all(clamped_physical[..., 0] >= 0)
    assert torch.count_nonzero(clamped_physical[..., 1, :]) == 0
    torch.testing.assert_close(updated[..., -1, :, :], clamped)
    assert not torch.equal(updated[0, 0], updated[1, 1])


def test_residual_decomposition_is_signed_and_invariant_to_common_depth_shift():
    stochastic = torch.tensor(
        [[[[[1.0]], [[4.0]]], [[[3.0]], [[0.0]]]]]
    )  # [M=1,K=2,B=2,N=1,C=1]
    mean = torch.tensor([[[[2.0]], [[2.0]]]])
    target = torch.tensor([[[1.0]], [[5.0]]])
    reference_mean = torch.tensor([[[3.0]], [[4.0]]])

    original = residual_decomposition_components(
        stochastic, mean, target, reference_mean
    )
    shifted = residual_decomposition_components(
        stochastic + 7.0,
        mean + 7.0,
        target + 7.0,
        reference_mean + 7.0,
    )

    torch.testing.assert_close(original.model_residuals, shifted.model_residuals)
    torch.testing.assert_close(original.reference_residual, shifted.reference_residual)
    assert torch.any(original.model_residuals < 0)
    assert torch.any(original.reference_residual < 0)


def test_particle_mean_mse_uses_particle_family_weights_and_excludes_dry_cells():
    predictions = torch.tensor(
        [
            [[[[1.0], [99.0]]], [[[3.0], [99.0]]]],
            [[[[3.0], [99.0]]], [[[1.0], [99.0]]]],
        ]
    ).reshape(2, 2, 2, 1)
    reference_mean = torch.zeros(2, 2, 1)
    family_weights = torch.tensor([[3.0, 1.0], [1.0, 3.0]])

    result = particle_bootstrap_mean_mse(
        predictions,
        reference_mean,
        family_weights=family_weights,
        structural_dry_mask=torch.tensor([[False, True], [False, True]]),
    )

    torch.testing.assert_close(result.per_particle, torch.tensor([3.0, 3.0]))
    torch.testing.assert_close(result.mean, torch.tensor(3.0))


def test_residual_centering_monitor_reports_rms_mc_error_and_exceedance_fraction():
    stochastic = torch.tensor(
        [[
            [[[-1.0], [2.0]]],
            [[[1.0], [2.0]]],
            [[[-1.0], [2.0]]],
            [[[1.0], [2.0]]],
        ]]
    )
    mean = torch.zeros(1, 1, 2, 1)

    monitor = residual_centering_monitor(stochastic, mean)

    torch.testing.assert_close(
        monitor.discrepancy_rms,
        torch.tensor(2.0).sqrt(),
    )
    torch.testing.assert_close(monitor.exceeds_two_se_fraction, torch.tensor(0.5))
    assert monitor.mc_se_rms > 0


class _RecordingALRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Parameter(torch.tensor(0.1))
        self.anchored_low_rank_enabled = True
        self.anchored_low_rank_num_particles = 2
        self.last_particle_ids = None
        self.last_latents = None
        self.adapters_only = None
        self.anchored_low_rank_active = True
        self.x_calls = []
        self.latent_calls = []

    def forward(self, *, x, ada_in, particle_ids, **kwargs):
        del kwargs
        self.last_particle_ids = particle_ids.detach().clone()
        self.last_latents = ada_in.detach().clone()
        self.latent_calls.append(ada_in.detach().clone())
        self.x_calls.append(x.detach().clone())
        particle_delta = particle_ids[:, None, None] * 0.2 if self.anchored_low_rank_active else 0.0
        return x[..., :1] + self.shared + particle_delta + ada_in[:, None, :1]

    def anchored_low_rank_offset_penalty(self):
        return self.shared.square()

    def set_anchored_low_rank_training_phase(self, *, adapters_only):
        self.adapters_only = bool(adapters_only)

    def set_anchored_low_rank_active(self, active):
        self.anchored_low_rank_active = bool(active)


def test_alr_trainer_vectorizes_particles_and_keeps_common_aleatory_bank():
    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001", "F002"], num_particles=2, seed=23
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=2,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        anchor_penalty_weight=1.0e-3,
        adapter_warmup_epochs=1,
        target_normalizer=_AffineNormalizer(),
        use_progress_bar=False,
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer.regularizer = None
    trainer.epoch = 0
    sample = {
        "x": torch.zeros(2, 1, 1),
        "y": torch.zeros(2, 1, 1),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
        "family_id": ["F001", "F002"],
    }
    loss_fn = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=2, reduction="mean"),
    )

    loss, metrics = trainer.train_one_batch(0, sample, loss_fn)

    assert torch.isfinite(loss)
    assert model.last_particle_ids.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    latent = model.last_latents.reshape(2, 2, 2, 1)
    torch.testing.assert_close(latent[0], latent[1])
    assert set(metrics) >= {
        "alr_particle_crps_0",
        "alr_particle_crps_1",
        "alr_anchor_displacement_norm",
    }
    trainer.on_epoch_start(0)
    assert model.adapters_only is True
    trainer.on_epoch_start(1)
    assert model.adapters_only is False

    eval_loss = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=4, reduction="mean"),
    )
    trainer.n_samples = 0
    eval_metrics, output = trainer.eval_one_batch(
        sample,
        {"crps": eval_loss},
        return_output=True,
    )
    assert torch.isfinite(eval_metrics["crps"])
    assert torch.isfinite(eval_metrics["alr_particle_crps_0"])
    assert torch.isfinite(eval_metrics["alr_particle_crps_1"])
    assert torch.isfinite(eval_metrics["alr_particle_correlation_mean"])
    assert torch.isfinite(eval_metrics["alr_anchor_displacement_norm"])
    assert output.shape == sample["y"].shape


def test_residual_decomposition_trainer_uses_an_explicit_zero_latent_mean_path():
    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001"], num_particles=2, seed=23
    )
    table = ReferenceDispersionTable(
        family_ids=["F001"],
        dispersion=torch.ones(1, 2, 1),
        reference_mean=torch.zeros(1, 2, 1),
        reference_mean_variance=torch.full((1, 2, 1), 0.01),
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        anchor_penalty_weight=0.0,
        adapter_warmup_epochs=1,
        target_normalizer=_IdentityNormalizer(),
        reference_dispersion=table,
        residual_decomposition_enabled=True,
        mean_loss_weight=1.0,
        residual_crps_weight=1.0,
        residual_gradient_probe_batches=1,
        use_progress_bar=False,
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer.regularizer = None
    trainer.epoch = 0
    trainer._sample_common_latents = lambda **kwargs: torch.tensor(
        [[[0.5]], [[-0.5]]], dtype=kwargs["dtype"]
    )
    sample = {
        "x": torch.zeros(1, 1, 1),
        "y": torch.zeros(1, 1, 1),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
        "family_id": ["F001"],
        "time_index": torch.tensor([0]),
    }
    loss_fn = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=2, reduction="mean"),
    )

    loss, metrics = trainer.train_one_batch(0, sample, loss_fn)

    assert torch.isfinite(loss)
    assert any(torch.count_nonzero(latent) == 0 for latent in model.latent_calls)
    assert set(metrics) >= {
        "alr_mean_mse_m2",
        "alr_residual_crps_m",
        "alr_reference_mean_se_rms_m",
        "alr_mean_residual_gradient_cosine",
    }

    first_bank = trainer._residual_monitor_latent_bank(
        batch_size=2, dtype=torch.float32
    )
    second_bank = trainer._residual_monitor_latent_bank(
        batch_size=2, dtype=torch.float32
    )
    torch.testing.assert_close(first_bank, second_bank)
    torch.testing.assert_close(first_bank[:, 0], first_bank[:, 1])

    trainer._residual_monitor_active = True
    eval_losses, _ = trainer.eval_one_batch(
        sample,
        {
            "crps": FloodMaskedCRPSLoss(
                policy="legacy_full_domain",
                base_loss=CRPSLoss(n_samples=4, reduction="mean"),
            )
        },
    )
    assert set(eval_losses) >= {
        "alr_monitor_mean_rmse_m",
        "alr_monitor_residual_crps_m",
        "alr_monitor_centering_discrepancy_rms_m",
        "alr_monitor_centering_mc_se_rms_m",
        "alr_monitor_centering_exceeds_2se_fraction",
    }

    trainer.configure_selection_contract(base_rmse=10.0)
    epoch_crps = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=4, reduction="mean"),
    )
    epoch_metrics = trainer.evaluate_all(
        0,
        {"crps": epoch_crps, "rmse": PhysicalRMSE()},
        {"test": [sample]},
    )
    assert "train_alr_mean_mse_m2" in epoch_metrics
    assert "train_alr_mean_residual_gradient_cosine" in epoch_metrics


class _IdentityNormalizer:
    def transform(self, values):
        return values

    def inverse_transform(self, values):
        return values


def test_alr_autoregressive_training_feeds_back_each_nested_trajectory():
    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001"], num_particles=2, seed=29
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=2,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        anchor_penalty_weight=0.0,
        adapter_warmup_epochs=1,
        target_normalizer=_IdentityNormalizer(),
        ar_rollout_steps=2,
        ar_finetune_start_epoch=0,
        use_progress_bar=False,
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer.regularizer = None
    trainer.epoch = 0
    trainer._sample_common_latents = lambda **kwargs: torch.tensor(
        [[[0.5]], [[1.0]]], dtype=kwargs["dtype"]
    )
    sample = {
        "x": torch.zeros(1, 1, 3),
        "y": torch.zeros(1, 1, 1),
        "static": torch.zeros(1, 1, 1),
        "boundary": torch.zeros(1, 1, 1, 1),
        "dynamic": torch.zeros(1, 1, 1, 1),
        "target_sequence": torch.zeros(1, 2, 1, 1),
        "boundary_sequence": torch.zeros(1, 2, 1, 1),
        "structural_dry_mask": torch.zeros(1, 1, dtype=torch.bool),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
        "family_id": ["F001"],
    }
    loss_fn = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=2, reduction="mean"),
    )

    loss, _ = trainer.train_one_batch(0, sample, loss_fn)

    assert torch.isfinite(loss)
    assert len(model.x_calls) == 2
    second_step_dynamic = model.x_calls[1][..., -1]
    assert torch.unique(second_step_dynamic).numel() > 1


def test_residual_decomposition_ar_rollout_keeps_a_separate_mean_history():
    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001"], num_particles=2, seed=29
    )
    table = ReferenceDispersionTable(
        family_ids=["F001"],
        dispersion=torch.ones(1, 3, 1),
        reference_mean=torch.zeros(1, 3, 1),
        reference_mean_variance=torch.full((1, 3, 1), 0.01),
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        anchor_penalty_weight=0.0,
        adapter_warmup_epochs=1,
        target_normalizer=_IdentityNormalizer(),
        reference_dispersion=table,
        residual_decomposition_enabled=True,
        ar_rollout_steps=2,
        ar_finetune_start_epoch=0,
        use_progress_bar=False,
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    trainer.regularizer = None
    trainer.epoch = 0
    trainer._sample_common_latents = lambda **kwargs: torch.tensor(
        [[[0.5]], [[-0.5]]], dtype=kwargs["dtype"]
    )
    sample = {
        "x": torch.zeros(1, 1, 3),
        "y": torch.zeros(1, 1, 1),
        "static": torch.zeros(1, 1, 1),
        "boundary": torch.zeros(1, 1, 1, 1),
        "dynamic": torch.zeros(1, 1, 1, 1),
        "target_sequence": torch.zeros(1, 2, 1, 1),
        "boundary_sequence": torch.zeros(1, 2, 1, 1),
        "structural_dry_mask": torch.zeros(1, 1, dtype=torch.bool),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
        "family_id": ["F001"],
        "time_index": torch.tensor([0]),
    }
    loss_fn = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=2, reduction="mean"),
    )

    loss, metrics = trainer.train_one_batch(0, sample, loss_fn)

    assert torch.isfinite(loss)
    assert len(model.x_calls) == 4
    assert model.x_calls[1].shape[0] == 2
    assert model.x_calls[3].shape[0] == 2
    torch.testing.assert_close(
        model.x_calls[3][..., -1].reshape(-1), torch.tensor([0.1, 0.3])
    )
    assert "alr_mean_mse_m2" in metrics


class _IndexedFamilies:
    sample_index = [
        ("F001_sim00", 1),
        ("F001_sim01", 1),
        ("F002_sim00", 1),
        ("F003_sim00", 1),
        ("F004_sim00", 1),
    ]


def test_alr_family_split_has_no_member_leakage_and_nested_training_subsets():
    full = split_alr_family_indices(
        _IndexedFamilies(), validation_family_count=1, seed=31
    )
    limited = split_alr_family_indices(
        _IndexedFamilies(), validation_family_count=1, seed=31, train_family_limit=2
    )

    assert set(full.train_family_ids).isdisjoint(full.validation_family_ids)
    assert set(limited.train_family_ids).issubset(full.train_family_ids)
    assert limited.validation_family_ids == full.validation_family_ids
    for index in full.train_indices:
        run_id, _ = _IndexedFamilies.sample_index[index]
        assert run_id.rpartition("_sim")[0] in full.train_family_ids


def test_alr_validation_selection_enforces_physical_rmse_noninferiority():
    from neuralop.flood.train.alr_fgn import PhysicalRMSE

    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001"], num_particles=2, seed=37
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        eval_aleatory_samples=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        deterministic_eval=True,
        eval_seed=41,
        use_progress_bar=False,
    )
    trainer.configure_selection_contract(base_rmse=0.0, margin=0.001)
    sample = {
        "x": torch.zeros(1, 1, 1),
        "y": torch.zeros(1, 1, 1),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
        "family_id": ["F001"],
    }
    crps = FloodMaskedCRPSLoss(
        policy="legacy_full_domain",
        base_loss=CRPSLoss(n_samples=4, reduction="mean"),
    )

    metrics = trainer.evaluate_all(
        epoch=0,
        eval_losses={"crps": crps, "rmse": PhysicalRMSE()},
        test_loaders={"test": [sample]},
    )

    assert torch.isfinite(torch.tensor(metrics["test_crps_unconstrained"]))
    assert metrics["test_rmse_gate_passed"] == 0.0
    assert metrics["test_crps"] == float("inf")


def test_alr_base_rmse_measurement_bypasses_adapters_and_restores_them():
    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001"], num_particles=2, seed=43
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        eval_aleatory_samples=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        deterministic_eval=True,
        eval_seed=47,
        use_progress_bar=False,
    )
    trainer._sample_common_latents = lambda **kwargs: torch.zeros(
        2, 1, 1, dtype=kwargs["dtype"]
    )
    sample = {
        "x": torch.zeros(1, 1, 1),
        "y": torch.zeros(1, 1, 1),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
        "family_id": ["F001"],
    }

    baseline = trainer.measure_frozen_base_rmse([sample])

    torch.testing.assert_close(torch.tensor(baseline), torch.tensor(0.1))
    assert model.anchored_low_rank_active is True


def test_alr_validation_chunks_nested_members_without_changing_outputs():
    model = _RecordingALRModel()
    bootstrap = DirichletFamilyBootstrap(
        family_ids=["F001", "F002"], num_particles=2, seed=24
    )
    trainer = AnchoredLowRankFGNTrainer(
        model=model,
        n_epochs=1,
        device="cpu",
        fgn_noise_dim=1,
        crps_n_samples=2,
        eval_aleatory_samples=3,
        eval_member_chunk_size=2,
        num_particles=2,
        family_bootstrap=bootstrap,
        use_progress_bar=False,
    )
    trainer._sample_common_latents = lambda **kwargs: torch.tensor(
        [[[0.1], [0.2]], [[0.3], [0.4]], [[0.5], [0.6]]],
        dtype=kwargs["dtype"],
    )
    sample = {
        "x": torch.zeros(2, 1, 1),
        "input_geom": torch.zeros(1, 1, 2),
        "latent_queries": torch.zeros(1, 1, 1, 2),
        "output_queries": torch.zeros(1, 1, 2),
    }

    chunked = trainer._forward_nested_x(sample, aleatory_samples=3)

    assert chunked.shape == (2, 3, 2, 1, 1)
    assert len(model.x_calls) == 3
    assert all(call.shape[0] <= 4 for call in model.x_calls)
    expected_particle_delta = torch.tensor([0.0, 0.2]).view(2, 1, 1, 1, 1)
    expected_latent = trainer._sample_common_latents(
        count=3, batch_size=2, dtype=sample["x"].dtype
    ).view(1, 3, 2, 1, 1)
    torch.testing.assert_close(
        chunked,
        0.1 + expected_particle_delta + expected_latent,
    )
