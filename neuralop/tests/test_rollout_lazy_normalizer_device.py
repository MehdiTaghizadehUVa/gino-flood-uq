import torch

from neuralop.flood.eval.datasets import (
    _extract_raw_render_context_from_sample,
    _transform_with_current_normalizer_device,
)


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



def test_raw_render_context_preserves_utm_geometry_and_elevation():
    raw = {
        "geometry": torch.tensor([[379000.0, 4072000.0], [379030.0, 4072030.0]]),
        "static": torch.tensor([[1.5, 99.0], [2.5, 100.0]]),
    }

    context = _extract_raw_render_context_from_sample(raw)

    assert torch.equal(context["geometry_raw"], raw["geometry"])
    assert torch.equal(context["static_raw"], raw["static"])
    assert torch.equal(context["elevation_raw"], torch.tensor([1.5, 2.5]))
    raw["geometry"][0, 0] = -1.0
    assert context["geometry_raw"][0, 0].item() == 379000.0
