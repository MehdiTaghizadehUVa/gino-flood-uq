from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuralop.flood.eval.mc_dropout import (
    enable_mc_dropout_only,
    evaluate_mc_dropout_one_step,
    validate_mc_dropout_config,
)
from neuralop.flood.processing.wv_impl import FloodGINODataProcessor


class _MeanSquaredLoss:
    reduction = "mean"

    def __call__(self, pred, y, **kwargs):
        return ((pred - y) ** 2).mean()


class _StrictMeanSquaredLoss:
    reduction = "mean"

    def __call__(self, pred, *, y, structural_dry_mask=None):
        del structural_dry_mask
        return ((pred - y) ** 2).mean()


class _FloodProcessorDropoutModel(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.BatchNorm1d(4)

    def forward(self, input_geom, latent_queries, output_queries, x, **kwargs):
        del input_geom, latent_queries, output_queries, kwargs
        # Touch BatchNorm so the test proves MC dropout does not globally train the model.
        _ = self.norm(torch.zeros(x.shape[0], 4, device=x.device))
        return self.dropout(x[..., -1:])


class _DropoutModel(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.norm = nn.BatchNorm1d(4)
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, **kwargs):
        y = self.dropout(x)
        # Touch BatchNorm without changing its eval-mode running statistics.
        _ = self.norm(torch.zeros(x.shape[0], 4, device=x.device))
        return y


def _cfg(**overrides):
    gino = {
        "use_fgn_noise": False,
        "output_distribution": "deterministic",
        "fno_channel_mlp_dropout": 0.05,
    }
    opt = {"training_loss": "l2"}
    uq = {
        "method": "mc_dropout",
        "mc_samples": 4,
        "mc_dropout": {
            "dropout_probability": 0.05,
            "activate_modules": "dropout_only",
            "seed": 123,
        },
    }
    data = {"target_variables": ["wd"]}
    structural_dry = {"policy": "masked_primary"}
    for section, values in overrides.items():
        locals()[section].update(values)
    return SimpleNamespace(
        gino=SimpleNamespace(**gino),
        opt=SimpleNamespace(**opt),
        uq=SimpleNamespace(
            method=uq["method"],
            mc_samples=uq["mc_samples"],
            mc_dropout=SimpleNamespace(**uq["mc_dropout"]),
        ),
        data=SimpleNamespace(**data),
        structural_dry=SimpleNamespace(**structural_dry),
        distributed=SimpleNamespace(seed=123),
    )


def test_validate_mc_dropout_rejects_mixed_uq_modes():
    with pytest.raises(ValueError, match="use_fgn_noise"):
        validate_mc_dropout_config(_cfg(gino={"use_fgn_noise": True}))
    with pytest.raises(ValueError, match="output_distribution"):
        validate_mc_dropout_config(_cfg(gino={"output_distribution": "gaussian"}))
    with pytest.raises(ValueError, match="training_loss"):
        validate_mc_dropout_config(_cfg(opt={"training_loss": "crps"}), require_training_loss_l2=True)
    with pytest.raises(ValueError, match="fno_channel_mlp_dropout"):
        validate_mc_dropout_config(_cfg(gino={"fno_channel_mlp_dropout": 0.1}))


def test_enable_mc_dropout_only_keeps_non_dropout_modules_eval():
    model = _DropoutModel(p=0.2)
    model.train()
    count = enable_mc_dropout_only(model)

    assert count == 1
    assert model.training is False
    assert model.dropout.training is True
    assert model.norm.training is False


def test_mc_one_step_eval_reproducible_but_stochastic():
    cfg = _cfg(
        gino={"fno_channel_mlp_dropout": 0.5},
        uq={"mc_dropout": {"dropout_probability": 0.5}},
    )
    model = _DropoutModel(p=0.5)
    sample = {
        "x": torch.ones(1, 64, 1),
        "y": torch.zeros(1, 64, 1),
        "structural_dry_mask": torch.tensor([False, True] * 32),
    }
    data_loader = [sample]
    logger = __import__("logging").getLogger("test_mc_dropout")

    first = evaluate_mc_dropout_one_step(
        model=model,
        data_processor=None,
        data_loader=data_loader,
        eval_losses={"l2": _MeanSquaredLoss()},
        config=cfg,
        device=torch.device("cpu"),
        logger=logger,
    )
    second = evaluate_mc_dropout_one_step(
        model=model,
        data_processor=None,
        data_loader=data_loader,
        eval_losses={"l2": _MeanSquaredLoss()},
        config=cfg,
        device=torch.device("cpu"),
        logger=logger,
    )

    assert first == second
    assert first["test_mc_spread_mean"] > 0.0
    assert first["test_mc_crps"] >= 0.0
    assert 0.0 <= first["test_mc_coverage_wd_90"] <= 1.0


def test_mc_one_step_eval_passes_only_loss_kwargs():
    cfg = _cfg(gino={"fno_channel_mlp_dropout": 0.5}, uq={"mc_dropout": {"dropout_probability": 0.5}})
    model = _DropoutModel(p=0.5)
    sample = {
        "x": torch.ones(1, 8, 1),
        "y": torch.zeros(1, 8, 1),
        "structural_dry_mask": torch.tensor([False, True] * 4),
        "static": torch.randn(1, 8, 2),
        "point_weights": torch.ones(1, 8, 1),
    }

    metrics = evaluate_mc_dropout_one_step(
        model=model,
        data_processor=None,
        data_loader=[sample],
        eval_losses={"strict_l2": _StrictMeanSquaredLoss()},
        config=cfg,
        device=torch.device("cpu"),
        logger=__import__("logging").getLogger("test_mc_dropout_kwargs"),
    )

    assert "test_strict_l2" in metrics


def test_mc_one_step_eval_keeps_dropout_active_after_wrapped_processor_eval():
    cfg = _cfg(
        gino={"fno_channel_mlp_dropout": 0.5},
        uq={"mc_dropout": {"dropout_probability": 0.5}},
    )
    model = _FloodProcessorDropoutModel(p=0.5)
    processor = FloodGINODataProcessor(device="cpu", target_norm=None, inverse_test=True)
    processor.wrap(model)
    n_cells = 8
    sample = {
        "dynamic": torch.ones(1, 3, n_cells, 1),
        "boundary": torch.zeros(1, 3, n_cells, 1),
        "static": torch.zeros(1, n_cells, 1),
        "geometry": torch.zeros(1, n_cells, 2),
        "query_points": torch.zeros(1, 4, 4, 2),
        "target": torch.zeros(1, n_cells, 1),
        "structural_dry_mask": torch.tensor([False, True] * (n_cells // 2)),
    }

    metrics = evaluate_mc_dropout_one_step(
        model=model,
        data_processor=processor,
        data_loader=[sample],
        eval_losses={"l2": _MeanSquaredLoss()},
        config=cfg,
        device=torch.device("cpu"),
        logger=__import__("logging").getLogger("test_mc_dropout_processor"),
    )

    assert model.training is False
    assert model.norm.training is False
    assert model.dropout.training is True
    assert metrics["test_mc_spread_mean"] > 0.0


def test_rollout_uses_dropout_only_mode_and_clamps_before_feedback():
    repo = __import__("pathlib").Path(__file__).resolve().parents[2]
    text = (repo / "neuralop" / "flood" / "eval" / "rollout.py").read_text()

    assert "enable_mc_dropout_only(model)" in text
    assert "mc_dropout_seed_context" in text
    assert "MC-dropout rollout is incompatible with Gaussian or FGN" in text
    assert "clamp_structural_dry_normalized_values" in text
    assert text.find("clamp_structural_dry_normalized_values") < text.find("current_dynamics[ens_idx] = torch.cat")
