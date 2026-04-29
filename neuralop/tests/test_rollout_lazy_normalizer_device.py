import torch

from neuralop.flood.eval.datasets import _transform_with_current_normalizer_device


class RecordingNormalizer:
    def __init__(self, device: str):
        self.mean = torch.empty((), device=device)
        self.std = torch.empty((), device=device)
        self.seen_device = None

    def transform(self, x):
        self.seen_device = x.device
        return torch.zeros(x.shape, dtype=x.dtype, device="cpu")


def test_lazy_rollout_transform_uses_current_normalizer_device_not_stale_fallback():
    normalizer = RecordingNormalizer("meta")
    value = torch.ones(2, 3)

    out = _transform_with_current_normalizer_device(
        normalizer,
        value,
        fallback_device=torch.device("cpu"),
    )

    assert normalizer.seen_device.type == "meta"
    assert out.device.type == "cpu"
    assert out.shape == value.shape
