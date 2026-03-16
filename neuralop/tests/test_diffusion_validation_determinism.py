import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralop.diffusion import ConditioningConfig, ConditionalDDOForecaster, PointRFFGaussianProcessSampler
from neuralop.flood.train.diffusion_loop import _evaluate_validation
from neuralop.flood.train.diffusion_runtime import DistContext
from neuralop.flood.utils.diffusion_script_utils import load_checkpoint_bundle, save_checkpoint_sidecars


class _DummyDenoiser(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, 1, bias=False)

    def forward(self, input_geom, latent_queries, output_queries, x, ada_in=None):
        return self.lin(x)


class _DiffusionValDataset(Dataset):
    def __init__(self, item):
        self.item = item

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return self.item


def _make_item(n_history: int = 2, n_cells: int = 6):
    torch.manual_seed(5)
    return {
        "static": torch.randn(n_cells, 2),
        "boundary": torch.randn(n_history, n_cells, 1),
        "dynamic": torch.randn(n_history, n_cells, 1),
        "geometry": torch.rand(n_cells, 2),
        "query_points": torch.rand(4, 4, 2),
        "target": torch.randn(n_cells, 1),
    }


def test_diffusion_validation_loss_is_repeatable_with_deterministic_eval_seed():
    denoiser = _DummyDenoiser(in_features=9)
    gp = PointRFFGaussianProcessSampler(gp_type="independent", sigma=1.0, rff_features=16)
    forecaster = ConditionalDDOForecaster(
        denoiser=denoiser,
        gp_sampler=gp,
        conditioning=ConditioningConfig(
            add_noisy_target=True,
            add_time_features=True,
            time_feature_type="sincos",
            time_injection="channel",
        ),
        sampler_num_steps=4,
    )

    loader = DataLoader(_DiffusionValDataset(_make_item()), batch_size=1, shuffle=False)
    dist_ctx = DistContext(use_distributed=False, rank=0, local_rank=0, world_size=1)

    val_1 = _evaluate_validation(
        forecaster=forecaster,
        loader=loader,
        device=torch.device("cpu"),
        target_norm=None,
        dist_ctx=dist_ctx,
        max_batches=2,
        deterministic_eval=True,
        eval_seed=77,
        epoch=3,
    )
    val_2 = _evaluate_validation(
        forecaster=forecaster,
        loader=loader,
        device=torch.device("cpu"),
        target_norm=None,
        dist_ctx=dist_ctx,
        max_batches=2,
        deterministic_eval=True,
        eval_seed=77,
        epoch=3,
    )

    assert val_1["val_loss"] == val_2["val_loss"]
    assert val_1["val_loss_full_domain"] == val_2["val_loss_full_domain"]


def test_load_checkpoint_bundle_merges_rng_state_from_legacy_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    legacy_payload = {
        "epoch": 1,
        "global_step": 4,
        "best_val_loss": 0.5,
        "normalizer_path": str(tmp_path / "normalizers.pt"),
        "target_variables": ["wd"],
        "gino_config": {"fno_hidden_channels": 16},
        "diffusion_hparams": {"sampler_num_steps": 4},
        "denoiser_state_dict": {"lin.weight": torch.randn(1, 3)},
        "time_mlp_state_dict": {"proj.weight": torch.randn(4, 4)},
        "optimizer_state_dict": {"state": {0: {"momentum_buffer": torch.randn(3)}}},
        "scheduler_state_dict": {"last_epoch": 1},
        "rng_state": {"format_version": 1, "world_size": 1, "rank_states": [{"torch_cpu": torch.get_rng_state()}]},
    }
    torch.save(legacy_payload, checkpoint_path)
    save_checkpoint_sidecars(
        checkpoint_path,
        denoiser_state_dict=legacy_payload["denoiser_state_dict"],
        metadata={k: legacy_payload[k] for k in (
            "epoch",
            "global_step",
            "best_val_loss",
            "normalizer_path",
            "target_variables",
            "gino_config",
            "diffusion_hparams",
        )},
        extra_state_dicts={"time_mlp_state_dict": legacy_payload["time_mlp_state_dict"]},
    )

    bundle = load_checkpoint_bundle(
        checkpoint_path,
        map_location="cpu",
        allow_unsafe_legacy_load=True,
        merge_legacy_training_state=True,
    )

    assert "optimizer_state_dict" in bundle
    assert "scheduler_state_dict" in bundle
    assert "rng_state" in bundle
