import pytest
import torch
from torch import nn

from neuralop.layers.gno_block import GNOBlock
from neuralop.layers.neighbor_search import NeighborSearch


class DummyIntegralTransform(nn.Module):
    def forward(self, y, neighbors, x, f_y, weighting_fn):
        return torch.zeros((x.shape[0], 1), dtype=x.dtype, device=x.device)


class NeighborSearchSpy(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, data, queries, radius, compute_norm):
        self.calls.append({
            "radius": radius,
            "compute_norm": compute_norm,
            "data_shape": tuple(data.shape),
            "queries_shape": tuple(queries.shape),
        })
        n_queries = queries.shape[0]
        n_data = data.shape[0]
        return {
            "neighbors_index": (torch.arange(n_queries, device=queries.device) % n_data).to(torch.int64),
            "neighbors_row_splits": torch.arange(n_queries + 1, device=queries.device, dtype=torch.int64),
        }


def _make_block():
    block = GNOBlock(
        in_channels=1,
        out_channels=1,
        coord_dim=2,
        radius=1.0,
        use_open3d_neighbor_search=False,
        pos_embedding_type=None,
    )
    block.integral_transform = DummyIntegralTransform()
    return block


def test_precompute_static_components_populates_cache_without_tensor_buffer_errors():
    block = _make_block()
    spy = NeighborSearchSpy()
    block.neighbor_search = spy
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    block.precompute_static_components(y, x)

    assert block._is_cached is True
    assert isinstance(block.cached_neighbors, dict)
    assert torch.equal(block.cached_y_original, y)
    assert torch.equal(block.cached_x_original, x)
    assert len(spy.calls) == 1


def test_first_forward_lazily_precomputes_and_reuses_neighbors():
    block = _make_block()
    spy = NeighborSearchSpy()
    block.neighbor_search = spy
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    block(y, x)
    assert block._is_cached is True
    assert block._is_verified is True
    assert len(spy.calls) == 1

    block(y.clone(), x.clone())
    assert len(spy.calls) == 1


def test_cached_reuse_refreshes_on_stale_inputs_in_forward():
    block = _make_block()
    spy = NeighborSearchSpy()
    block.neighbor_search = spy
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    block(y, x)
    x_changed = x.clone()
    x_changed[0, 0] += 0.25

    block(y, x_changed)

    assert len(spy.calls) == 2
    assert torch.equal(block.cached_x_original, x_changed)


def test_strict_verifier_detects_stale_inputs():
    block = _make_block()
    spy = NeighborSearchSpy()
    block.neighbor_search = spy
    y = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    x = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    block.precompute_static_components(y, x)
    x_changed = x.clone()
    x_changed[0, 0] += 0.25

    with pytest.raises(ValueError, match="Input tensor x has changed since precomputation"):
        block._verify_cached_components(y, x_changed)

    assert len(spy.calls) == 1


def test_neighbor_search_false_does_not_compute_norm():
    search = NeighborSearch(use_open3d=False)
    data = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    queries = torch.tensor([[0.0, 0.0], [0.5, 0.0]], dtype=torch.float32)

    no_norm = search(data, queries, radius=1.5, compute_norm=False)
    with_norm = search(data, queries, radius=1.5, compute_norm=True)

    assert "norm" not in no_norm
    assert "norm" in with_norm
