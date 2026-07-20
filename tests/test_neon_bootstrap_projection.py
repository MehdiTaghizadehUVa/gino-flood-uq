from __future__ import annotations

import torch

from neuralop.flood.neon import (
    fixed_bootstrap_model_projection,
    match_projected_trainable_hidden_channels,
)


def test_fixed_bootstrap_model_projection_is_deterministic_and_orthonormal():
    first = fixed_bootstrap_model_projection(128, 16, seed=91)
    second = fixed_bootstrap_model_projection(128, 16, seed=91)
    different = fixed_bootstrap_model_projection(128, 16, seed=92)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert first.shape == (128, 16)
    assert torch.allclose(first.T @ first, torch.eye(16), atol=2.0e-6, rtol=0.0)


def test_fixed_projection_preserves_standard_normal_model_index_covariance():
    projection = fixed_bootstrap_model_projection(128, 16, seed=7)
    generator = torch.Generator().manual_seed(8)
    bootstrap_index = torch.randn(50_000, 128, generator=generator)
    model_index = bootstrap_index @ projection
    covariance = torch.cov(model_index.T)

    assert torch.allclose(model_index.mean(0), torch.zeros(16), atol=0.02, rtol=0.0)
    assert torch.allclose(covariance, torch.eye(16), atol=0.035, rtol=0.0)


def test_parameter_match_resolver_minimizes_trainable_mlp_count_gap():
    hidden = match_projected_trainable_hidden_channels(
        input_dim=64,
        source_hidden_channels=16,
        source_basis_dim=128,
        target_basis_dim=32,
        out_channels=1,
        n_hidden_layers=2,
    )
    assert hidden > 16

    def count(h: int, basis: int) -> int:
        return (64 * h + h) + (h * h + h) + (h * basis + basis)

    target = count(16, 128)
    selected_gap = abs(count(hidden, 32) - target)
    assert selected_gap <= abs(count(hidden - 1, 32) - target)
    assert selected_gap <= abs(count(hidden + 1, 32) - target)
