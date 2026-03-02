import pytest
import torch

from ..normalization_layers import AdaIN


def test_adain_b1_parity_with_affine_group_norm():
    torch.manual_seed(0)
    embed_dim = 6
    channels = 4
    layer = AdaIN(embed_dim=embed_dim, in_channels=channels)

    x = torch.randn(1, channels, 5, 5)
    e = torch.randn(embed_dim)
    out = layer(x, embedding=e)

    affine = layer.mlp(e.unsqueeze(0)).squeeze(0)
    weight, bias = torch.split(affine, channels, dim=0)
    ref = torch.nn.functional.group_norm(
        x, channels, weight=weight, bias=bias, eps=layer.eps
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_adain_per_sample_affine_changes_outputs():
    torch.manual_seed(1)
    embed_dim = 5
    channels = 3
    layer = AdaIN(embed_dim=embed_dim, in_channels=channels)

    x_single = torch.randn(1, channels, 4, 4)
    x = x_single.expand(2, -1, -1, -1).clone()
    e = torch.stack([torch.zeros(embed_dim), torch.ones(embed_dim)], dim=0)

    out = layer(x, embedding=e)
    assert not torch.allclose(out[0], out[1], atol=1e-7, rtol=1e-6)


def test_adain_broadcast_embedding_matches_explicit_batch():
    torch.manual_seed(2)
    embed_dim = 7
    channels = 2
    layer = AdaIN(embed_dim=embed_dim, in_channels=channels)

    x = torch.randn(3, channels, 6, 6)
    e1 = torch.randn(1, embed_dim)
    e3 = e1.expand(3, -1)

    out_broadcast = layer(x, embedding=e1)
    out_explicit = layer(x, embedding=e3)
    assert torch.allclose(out_broadcast, out_explicit, atol=1e-7, rtol=1e-6)


def test_adain_embedding_batch_mismatch_raises():
    torch.manual_seed(3)
    embed_dim = 4
    channels = 2
    layer = AdaIN(embed_dim=embed_dim, in_channels=channels)

    x = torch.randn(2, channels, 4, 4)
    e_bad = torch.randn(3, embed_dim)
    with pytest.raises(ValueError, match="must be 1 or match x batch"):
        _ = layer(x, embedding=e_bad)
