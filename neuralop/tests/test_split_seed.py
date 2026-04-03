from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset, random_split

from neuralop.flood.utils.runtime_core import make_split_generator, resolve_split_seed


def test_resolve_split_seed_defaults_to_model_seed():
    cfg = SimpleNamespace()
    assert resolve_split_seed(cfg, 123) == 123


def test_resolve_split_seed_honors_explicit_value():
    cfg = SimpleNamespace(split_seed=17)
    assert resolve_split_seed(cfg, 123) == 17


def test_explicit_split_seed_reproduces_splits_across_model_seeds():
    dataset = TensorDataset(torch.arange(12))
    cfg = SimpleNamespace(split_seed=5)
    split_seed_a = resolve_split_seed(cfg, 123)
    split_seed_b = resolve_split_seed(cfg, 999)
    train_a, _ = random_split(dataset, [10, 2], generator=make_split_generator(split_seed_a))
    train_b, _ = random_split(dataset, [10, 2], generator=make_split_generator(split_seed_b))
    assert list(train_a.indices) == list(train_b.indices)


def test_fallback_model_seed_changes_split_when_split_seed_not_set():
    dataset = TensorDataset(torch.arange(12))
    train_a, _ = random_split(dataset, [10, 2], generator=make_split_generator(resolve_split_seed(SimpleNamespace(), 123)))
    train_b, _ = random_split(dataset, [10, 2], generator=make_split_generator(resolve_split_seed(SimpleNamespace(), 124)))
    assert list(train_a.indices) != list(train_b.indices)
