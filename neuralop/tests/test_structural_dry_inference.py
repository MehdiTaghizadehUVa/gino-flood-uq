import torch
from torch import nn

from neuralop.flood.data.structural_dry import (
    apply_structural_dry_zero_mask,
    clamp_structural_dry_normalized_values,
)
from neuralop.flood.processing.wv_impl import FloodGINODataProcessor
from neuralop.flood.train.fgn import FGNTrainer


class AffineNormalizer:
    def __init__(self, mean=10.0, std=2.0):
        self.mean = torch.tensor(float(mean), dtype=torch.float32)
        self.std = torch.tensor(float(std), dtype=torch.float32)
        self.eps = 1e-7

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def inverse_transform(self, x):
        return x * self.std + self.mean

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)


class _DummyModel(nn.Module):
    def forward(self, x=None, ada_in=None, **kwargs):
        return x[..., :1] + ada_in.mean(dim=1, keepdim=True).unsqueeze(1) * 0.0


class _RecordingProcessor:
    def __init__(self):
        self.seen_masks = []

    def preprocess(self, sample):
        return sample

    def postprocess(self, out, sample):
        self.seen_masks.append(sample.get("structural_dry_mask"))
        return out, sample


def test_apply_structural_dry_zero_mask_zeroes_only_dry_cells():
    values = torch.tensor([[[1.0], [2.0], [3.0]]])
    dry_mask = torch.tensor([False, True, False])
    got = apply_structural_dry_zero_mask(values, structural_dry_mask=dry_mask)
    expected = torch.tensor([[[1.0], [0.0], [3.0]]])
    assert torch.equal(got, expected)


def test_clamp_structural_dry_normalized_values_zeroes_in_physical_space():
    norm = AffineNormalizer(mean=10.0, std=2.0)
    values = torch.tensor([[[-4.5], [0.5], [1.0]]])
    dry_mask = torch.tensor([False, True, False])
    got = clamp_structural_dry_normalized_values(
        values,
        structural_dry_mask=dry_mask,
        normalizer=norm,
    )
    expected = torch.tensor([[[-4.5], [-5.0], [1.0]]])
    assert torch.allclose(got, expected)


def test_postprocess_clamps_structural_dry_outputs_in_physical_space():
    norm = AffineNormalizer(mean=10.0, std=2.0)
    processor = FloodGINODataProcessor(
        device="cpu",
        target_norm=norm,
        inverse_test=True,
        output_distribution="deterministic",
    )
    processor.eval()
    out = torch.tensor([[[1.0], [0.5], [-4.0]]])
    sample = {
        "y": torch.zeros_like(out),
        "structural_dry_mask": torch.tensor([False, True, False]),
    }
    got, _ = processor.postprocess(out, sample)
    expected = torch.tensor([[[12.0], [0.0], [2.0]]])
    assert torch.allclose(got, expected)


def test_fgn_eval_one_batch_passes_structural_dry_mask_to_postprocess():
    trainer = object.__new__(FGNTrainer)
    trainer.data_processor = _RecordingProcessor()
    trainer.device = torch.device("cpu")
    trainer.n_samples = 0
    trainer.crps_n_samples = 2
    trainer.fgn_noise_dim = 4
    trainer.model = _DummyModel()

    sample = {
        "x": torch.ones(1, 3, 1),
        "y": torch.ones(1, 3, 1),
        "input_geom": torch.zeros(1, 3, 2),
        "latent_queries": torch.zeros(1, 2, 2, 2),
        "output_queries": torch.zeros(1, 3, 2),
        "structural_dry_mask": torch.tensor([False, True, False]),
    }

    _, _ = trainer.eval_one_batch(sample, {}, return_output=True)
    assert len(trainer.data_processor.seen_masks) == 2
    for seen in trainer.data_processor.seen_masks:
        assert seen is not None
        assert torch.equal(seen, sample["structural_dry_mask"])
