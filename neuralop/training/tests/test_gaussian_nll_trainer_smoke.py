import torch
from torch import nn

import pytest

from neuralop.losses.probabilistic_losses import GaussianNLLLoss

try:
    from scripts.train_gino_flood_train_rollout_animation_WV import GaussianNLLTrainer
except Exception as exc:  # pragma: no cover - environment dependency guard
    pytest.skip(f"Gaussian trainer script import unavailable: {exc}", allow_module_level=True)


class DummyGaussianModel(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.logvar = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x=None, **kwargs):
        if x is None:
            raise ValueError("DummyGaussianModel expects x.")
        mu = x[..., : self.out_channels] + self.bias.view(1, 1, -1)
        logvar = self.logvar.view(1, 1, -1).expand_as(mu)
        return torch.cat([mu, logvar], dim=-1)


def _setup_trainer(out_channels: int, ar_rollout_steps: int = 1):
    model = DummyGaussianModel(out_channels=out_channels)
    trainer = GaussianNLLTrainer(
        model=model,
        n_epochs=1,
        data_processor=None,
        device=torch.device("cpu"),
        wandb_log=False,
        verbose=False,
        logger=None,
        use_progress_bar=False,
        scheduler_monitor="train_err",
        eval_interval=1,
        rel_l2_loss_fn=None,
        ar_finetune_start_epoch=0,
        ar_rollout_steps=ar_rollout_steps,
        ar_curriculum_epochs_per_step=0,
        gaussian_min_logvar=-9.0,
        gaussian_max_logvar=4.0,
    )
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer.regularizer = None
    trainer.epoch = 0
    trainer.n_samples = 0
    return trainer


def test_gaussian_trainer_single_step_smoke():
    torch.manual_seed(0)
    B, N, C = 2, 7, 2
    trainer = _setup_trainer(out_channels=C, ar_rollout_steps=1)
    training_loss = GaussianNLLLoss(
        reduction="mean",
        min_logvar=-9.0,
        max_logvar=4.0,
        logvar_reg_weight=0.0,
    )
    sample = {
        "x": torch.randn(B, N, 4),
        "y": torch.randn(B, N, C),
    }
    loss, _ = trainer.train_one_batch(0, sample, training_loss)
    assert torch.isfinite(loss)
    loss.backward()
    trainer.optimizer.step()


def test_gaussian_trainer_ar_smoke():
    torch.manual_seed(1)
    B, N, C = 2, 6, 2
    H, T, BC = 3, 2, 1
    trainer = _setup_trainer(out_channels=C, ar_rollout_steps=T)
    training_loss = GaussianNLLLoss(
        reduction="mean",
        min_logvar=-9.0,
        max_logvar=4.0,
        logvar_reg_weight=0.0,
    )
    sample = {
        "y": torch.randn(B, N, C),
        "dynamic": torch.randn(B, H, N, C),
        "boundary": torch.randn(B, H, N, BC),
        "static": torch.randn(B, N, 2),
        "input_geom": torch.randn(1, N, 2),
        "latent_queries": torch.randn(1, 4, 4, 2),
        "output_queries": torch.randn(1, N, 2),
        "target_sequence": torch.randn(B, T, N, C),
        "boundary_sequence": torch.randn(B, T, N, BC),
    }
    loss, _ = trainer.train_one_batch(0, sample, training_loss)
    assert torch.isfinite(loss)
    loss.backward()
    trainer.optimizer.step()
