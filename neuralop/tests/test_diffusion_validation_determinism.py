import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralop.diffusion import ConditioningConfig, ConditionalDDOForecaster, PointRFFGaussianProcessSampler
from neuralop.flood.train.diffusion_loop import _evaluate_validation
from neuralop.flood.train.diffusion_runtime import DistContext


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
