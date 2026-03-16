import pytest
import torch

from neuralop.flood.losses import (
    FloodMaskedCRPSLoss,
    FloodMaskedGaussianNLLLoss,
    FloodMaskedRelLpLoss,
)


class _RecordingLoss:
    reduction = "mean"

    def __init__(self, value=1.0):
        self.value = value
        self.last_weights = None

    def __call__(self, *args, spatial_weights=None, **kwargs):
        del args, kwargs
        self.last_weights = spatial_weights
        return torch.tensor(float(self.value))


def test_masked_rel_l2_excludes_structurally_dry_cells():
    y = torch.tensor([[[1.0], [0.0]]], dtype=torch.float32)
    y_pred = torch.tensor([[[2.0], [100.0]]], dtype=torch.float32)
    dry_mask = torch.tensor([False, True])

    legacy = FloodMaskedRelLpLoss(policy="legacy_full_domain")
    masked = FloodMaskedRelLpLoss(policy="masked_primary")

    legacy_val = legacy(y_pred, y, structural_dry_mask=dry_mask).item()
    masked_val = masked(y_pred, y, structural_dry_mask=dry_mask).item()

    assert masked_val == pytest.approx(1.0)
    assert legacy_val > masked_val


def test_masked_probabilistic_wrappers_pass_binary_spatial_weights():
    dry_mask = torch.tensor([False, True])
    y = torch.ones(1, 2, 1)
    pred_samples = torch.randn(3, 1, 2, 1)
    packed_pred = torch.randn(1, 2, 2)

    crps_base = _RecordingLoss(value=1.5)
    crps = FloodMaskedCRPSLoss(policy="masked_primary", base_loss=crps_base)
    out_crps = crps(pred_samples, y, structural_dry_mask=dry_mask)
    assert out_crps.item() == pytest.approx(1.5)
    assert crps_base.last_weights is not None
    assert torch.equal(
        crps_base.last_weights[0, :, 0].to(dtype=torch.bool),
        torch.tensor([True, False]),
    )

    gaussian_base = _RecordingLoss(value=2.5)
    gaussian = FloodMaskedGaussianNLLLoss(
        policy="masked_primary",
        base_loss=gaussian_base,
    )
    out_gaussian = gaussian(packed_pred, y, structural_dry_mask=dry_mask)
    assert out_gaussian.item() == pytest.approx(2.5)
    assert gaussian_base.last_weights is not None
    assert torch.equal(
        gaussian_base.last_weights[0, :, 0].to(dtype=torch.bool),
        torch.tensor([True, False]),
    )
