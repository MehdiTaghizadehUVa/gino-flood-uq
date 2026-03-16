import pytest
import torch

from neuralop.flood.data.normalization_impl import fit_normalizers_streaming


class _TinyDataset:
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return self.sample


def test_masked_primary_normalization_only_changes_water_state_channels():
    sample = {
        "geometry": torch.tensor([[0.0, 0.0], [10.0, 10.0]], dtype=torch.float32),
        "static": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        "boundary": torch.tensor(
            [[[5.0], [500.0]], [[6.0], [600.0]]], dtype=torch.float32
        ),
        "dynamic": torch.tensor(
            [[[2.0], [200.0]], [[4.0], [400.0]]], dtype=torch.float32
        ),
        "target": torch.tensor([[1.0], [100.0]], dtype=torch.float32),
        "structural_dry_mask": torch.tensor([False, True]),
    }
    dataset = _TinyDataset(sample)

    legacy = fit_normalizers_streaming(
        dataset,
        chunk_size=1,
        expect_target=True,
        structural_dry_policy="legacy_full_domain",
    )
    masked = fit_normalizers_streaming(
        dataset,
        chunk_size=1,
        expect_target=True,
        structural_dry_policy="masked_primary",
    )

    assert torch.allclose(legacy["geometry"].mean, masked["geometry"].mean)
    assert torch.allclose(legacy["static"].mean, masked["static"].mean)
    assert torch.allclose(legacy["boundary"].mean, masked["boundary"].mean)

    legacy_target_mean = legacy["target"].mean.squeeze().item()
    masked_target_mean = masked["target"].mean.squeeze().item()
    assert legacy_target_mean == pytest.approx(50.5)
    assert masked_target_mean == pytest.approx((1.0 + 2.0 + 4.0) / 3.0)
    assert masked["dynamic"] is masked["target"]
