import torch

from .. import resample as resample_module
from ..resample import _should_use_bicubic_resample, resample


class _DummyTensor:
    def __init__(self, *, is_cuda: bool):
        self.is_cuda = is_cuda


def test_resample():
    a = torch.randn(10, 20, 40, 50)

    res_scale = [2, 3]
    axis = [-2, -1]

    b = resample(a, res_scale, axis)
    assert b.shape[-1] == 3 * a.shape[-1] and b.shape[-2] == 2 * a.shape[-2]

    a = torch.randn((10, 20, 40, 50, 60))

    res_scale = [0.5, 3, 4]
    axis = [-3, -2, -1]
    b = resample(a, res_scale, axis)

    assert b.shape[-1] == 4 * a.shape[-1] and b.shape[-2] == 3 * a.shape[-2] and b.shape[-3] == int(0.5 * a.shape[-3])


def test_should_use_bicubic_resample_allows_fast_path_without_deterministic_cuda(monkeypatch):
    monkeypatch.setattr(resample_module, "_deterministic_algorithms_enabled", lambda: False)
    assert _should_use_bicubic_resample(_DummyTensor(is_cuda=True), [-2, -1]) is True
    assert _should_use_bicubic_resample(_DummyTensor(is_cuda=False), [-2, -1]) is True


def test_should_use_bicubic_resample_disables_fast_path_for_deterministic_cuda(monkeypatch):
    monkeypatch.setattr(resample_module, "_deterministic_algorithms_enabled", lambda: True)
    assert _should_use_bicubic_resample(_DummyTensor(is_cuda=True), [-2, -1]) is False
    assert _should_use_bicubic_resample(_DummyTensor(is_cuda=False), [-2, -1]) is True
    assert _should_use_bicubic_resample(_DummyTensor(is_cuda=True), [-3, -2, -1]) is False
