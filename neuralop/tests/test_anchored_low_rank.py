import torch

from neuralop.layers.anchored_low_rank import (
    AnchoredLowRankDenseAdapter,
    AnchoredLowRankSpectralAdapter,
)
from neuralop.layers.channel_mlp import ChannelMLP, LinearChannelMLP
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.gno_block import GNOBlock
from neuralop.models.gino import GINO


def test_dense_adapter_matches_explicit_particle_weights():
    base_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
    adapter = AnchoredLowRankDenseAdapter(
        in_features=4,
        out_features=3,
        num_particles=2,
        rank=2,
        reference_weight=base_weight,
        anchor_relative_norm=0.05,
        seed=7,
    )
    with torch.no_grad():
        adapter.offset_a.copy_(torch.randn_like(adapter.offset_a) * 0.01)
        adapter.offset_b.copy_(torch.randn_like(adapter.offset_b) * 0.01)

    x = torch.randn(4, 5, 4)
    particle_ids = torch.tensor([0, 1, 0, 1])
    actual = adapter(x, particle_ids, channels_last=True)

    delta = adapter.explicit_delta_weight(particle_ids)
    expected = torch.einsum("bni,boi->bno", x, delta)
    torch.testing.assert_close(actual, expected)


def test_dense_adapter_keeps_anchors_frozen_and_isolates_particle_gradients():
    adapter = AnchoredLowRankDenseAdapter(
        in_features=3,
        out_features=2,
        num_particles=3,
        rank=2,
        reference_weight=torch.ones(2, 3),
        anchor_relative_norm=0.01,
        seed=11,
    )
    x = torch.randn(2, 3, requires_grad=True)
    adapter(x, torch.tensor([1, 1]), channels_last=True).sum().backward()

    assert adapter.anchor_a.requires_grad is False
    assert adapter.anchor_b.requires_grad is False
    assert adapter.anchor_a.grad is None
    assert adapter.anchor_b.grad is None
    assert torch.count_nonzero(adapter.offset_a.grad[1]) > 0
    assert torch.count_nonzero(adapter.offset_b.grad[1]) > 0
    assert torch.count_nonzero(adapter.offset_a.grad[[0, 2]]) == 0
    assert torch.count_nonzero(adapter.offset_b.grad[[0, 2]]) == 0


def test_spectral_adapter_matches_explicit_complex_weight_contraction():
    reference_weight = torch.randn(3, 2, 4, 3, dtype=torch.cfloat)
    adapter = AnchoredLowRankSpectralAdapter(
        in_channels=3,
        out_channels=2,
        n_modes=(4, 3),
        num_particles=2,
        rank=2,
        reference_weight=reference_weight,
        anchor_relative_norm=0.02,
        seed=13,
    )
    with torch.no_grad():
        adapter.offset_a.copy_(torch.randn_like(adapter.offset_a) * 0.01)
        adapter.offset_b.copy_(torch.randn_like(adapter.offset_b) * 0.01)

    x = torch.randn(3, 3, 4, 3, dtype=torch.cfloat)
    particle_ids = torch.tensor([1, 0, 1])
    actual = adapter(x, particle_ids)
    delta = adapter.explicit_delta_weight(particle_ids, mode_shape=x.shape[2:])
    expected = torch.einsum("bi...,bio...->bo...", x, delta)
    torch.testing.assert_close(actual, expected)


def test_adapter_requires_explicit_valid_particle_ids():
    adapter = AnchoredLowRankDenseAdapter(
        in_features=2,
        out_features=2,
        num_particles=2,
        rank=1,
        reference_weight=torch.eye(2),
        anchor_relative_norm=0.01,
        seed=3,
    )
    x = torch.ones(2, 2)

    for particle_ids in (None, torch.tensor([0]), torch.tensor([0, 2])):
        try:
            adapter(x, particle_ids, channels_last=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid particle IDs to fail: {particle_ids}")


def test_channel_mlp_disabled_path_is_exact_and_adds_no_state_keys():
    mlp = ChannelMLP(3, out_channels=2, hidden_channels=5, n_layers=2, n_dim=2)
    x = torch.randn(4, 3, 6, 7)
    expected = mlp(x)
    state_keys = set(mlp.state_dict())

    actual = mlp(x, particle_ids=None)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(mlp.state_dict()) == state_keys
    assert not any("anchored_low_rank" in key for key in state_keys)


def test_channel_mlp_adapter_matches_layerwise_explicit_computation():
    mlp = ChannelMLP(3, out_channels=2, hidden_channels=4, n_layers=2, n_dim=1)
    mlp.enable_anchored_low_rank(
        layer_indices=[0, 1],
        num_particles=2,
        rank=2,
        anchor_relative_norm=0.03,
        seed=19,
    )
    x = torch.randn(3, 3, 5)
    ids = torch.tensor([0, 1, 0])

    hidden = mlp.fcs[0](x) + mlp.anchored_low_rank["0"](
        x, ids, channels_last=False
    )
    hidden = mlp.non_linearity(hidden)
    expected = mlp.fcs[1](hidden) + mlp.anchored_low_rank["1"](
        hidden, ids, channels_last=False
    )

    torch.testing.assert_close(mlp(x, particle_ids=ids), expected)


def test_linear_channel_mlp_can_adapt_only_final_layers():
    mlp = LinearChannelMLP([3, 5, 4, 2])
    mlp.enable_anchored_low_rank(
        layer_indices=[1, 2],
        num_particles=2,
        rank=2,
        anchor_relative_norm=0.01,
        seed=23,
    )
    assert set(mlp.anchored_low_rank) == {"1", "2"}

    x = torch.randn(2, 7, 3)
    ids = torch.tensor([0, 1])
    out = mlp(x, particle_ids=ids)
    assert out.shape == (2, 7, 2)


def test_spectral_conv_zero_norm_adapter_preserves_output_exactly():
    conv = SpectralConv(
        in_channels=2,
        out_channels=3,
        n_modes=(4, 4),
        factorization="Dense",
    )
    x = torch.randn(2, 2, 8, 8)
    expected = conv(x)
    conv.enable_anchored_low_rank(
        num_particles=2,
        rank=2,
        anchor_relative_norm=0.0,
        seed=29,
    )

    actual = conv(x, particle_ids=torch.tensor([0, 1]))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_spectral_conv_adapter_requires_particle_ids_when_enabled():
    conv = SpectralConv(2, 2, n_modes=(4, 4), factorization="Dense")
    conv.enable_anchored_low_rank(
        num_particles=2,
        rank=1,
        anchor_relative_norm=0.01,
        seed=31,
    )
    try:
        conv(torch.randn(2, 2, 8, 8))
    except ValueError as exc:
        assert "particle_ids" in str(exc)
    else:
        raise AssertionError("Expected an enabled spectral adapter to require particle_ids.")


def test_fno_blocks_adapt_only_requested_final_layers():
    blocks = FNOBlocks(
        in_channels=4,
        out_channels=4,
        n_modes=(4, 4),
        n_layers=4,
        factorization="Dense",
        norm=None,
    )
    blocks.enable_anchored_low_rank(
        last_n_blocks=2,
        num_particles=2,
        rank=2,
        anchor_relative_norm=0.01,
        seed=37,
        adapt_spectral=True,
        adapt_pointwise=True,
    )

    assert blocks.convs[0].anchored_low_rank is None
    assert blocks.convs[1].anchored_low_rank is None
    assert blocks.convs[2].anchored_low_rank is not None
    assert blocks.convs[3].anchored_low_rank is not None
    assert not blocks.channel_mlp[0].anchored_low_rank
    assert not blocks.channel_mlp[1].anchored_low_rank
    assert blocks.channel_mlp[2].anchored_low_rank
    assert blocks.channel_mlp[3].anchored_low_rank


def test_fno_zero_norm_adapters_preserve_multiblock_forward():
    blocks = FNOBlocks(
        in_channels=3,
        out_channels=3,
        n_modes=(4, 4),
        n_layers=3,
        factorization="Dense",
        norm=None,
    )
    x = torch.randn(2, 3, 8, 8)
    expected = x
    for layer_idx in range(3):
        expected = blocks(expected, layer_idx)

    blocks.enable_anchored_low_rank(
        last_n_blocks=2,
        num_particles=2,
        rank=2,
        anchor_relative_norm=0.0,
        seed=41,
        adapt_spectral=True,
        adapt_pointwise=True,
    )
    actual = x
    ids = torch.tensor([0, 1])
    for layer_idx in range(3):
        actual = blocks(actual, layer_idx, particle_ids=ids)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_output_gno_vectorizes_particle_specific_decoder_kernels():
    block = GNOBlock(
        in_channels=1,
        out_channels=1,
        coord_dim=1,
        radius=10.0,
        transform_type="linear",
        pos_embedding_type=None,
        channel_mlp_layers=[4, 3],
        use_open3d_neighbor_search=False,
        use_torch_scatter_reduce=False,
    )
    block.enable_anchored_low_rank(
        final_n_layers=2,
        num_particles=2,
        rank=2,
        anchor_relative_norm=0.02,
        seed=43,
    )
    y = torch.tensor([[0.0], [1.0], [2.0]])
    x = torch.tensor([[0.5], [1.5]])
    f_y = torch.randn(2, 3, 1)
    ids = torch.tensor([0, 1])

    batched = block(y=y, x=x, f_y=f_y, particle_ids=ids)
    separate = torch.cat(
        [
            block(y=y, x=x, f_y=f_y[i : i + 1], particle_ids=ids[i : i + 1])
            for i in range(2)
        ],
        dim=0,
    )

    torch.testing.assert_close(batched, separate)


def _tiny_gino(*, anchored_low_rank=None):
    return GINO(
        in_channels=2,
        out_channels=1,
        projection_channels=6,
        gno_coord_dim=2,
        gno_radius=10.0,
        in_gno_transform_type="linear",
        out_gno_transform_type="linear",
        gno_pos_embed_type=None,
        fno_n_modes=(4, 4),
        fno_hidden_channels=4,
        fno_n_layers=3,
        in_gno_channel_mlp_hidden_layers=[5],
        out_gno_channel_mlp_hidden_layers=[6, 5],
        gno_use_open3d=False,
        gno_use_torch_scatter=False,
        fno_factorization="Dense",
        fno_norm="ada_in",
        use_fgn_noise=True,
        fgn_noise_dim=3,
        anchored_low_rank=anchored_low_rank,
    )


def _tiny_gino_inputs(batch_size=2):
    input_geom = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    axis = torch.linspace(0.0, 1.0, 4)
    gx, gy = torch.meshgrid(axis, axis, indexing="ij")
    latent_queries = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    output_queries = input_geom.clone()
    x = torch.randn(batch_size, 3, 2)
    z = torch.randn(batch_size, 3)
    return dict(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        ada_in=z,
    )


def test_gino_disabled_path_has_no_alr_parameters_and_accepts_standard_forward():
    model = _tiny_gino(anchored_low_rank={"enabled": False})
    out = model(**_tiny_gino_inputs())

    assert out.shape == (2, 3, 1)
    assert model.anchored_low_rank_enabled is False
    assert model.anchored_low_rank_parameter_counts()["adapter_trainable"] == 0
    assert not any("anchored_low_rank" in name for name, _ in model.named_parameters())


def test_gino_alr_particles_modify_operator_and_respect_parameter_gate():
    cfg = {
        "enabled": True,
        "num_particles": 2,
        "rank": 2,
        "anchor_seed": 47,
        "anchor_relative_norm": 0.03,
        "fno_last_n_blocks": 2,
        "adapt_spectral": True,
        "adapt_pointwise": True,
        "adapt_output_gno": True,
        "adapt_output_projection": True,
        "adapt_forcing_encoder": False,
    }
    model = _tiny_gino(anchored_low_rank=cfg)
    inputs = _tiny_gino_inputs(batch_size=2)
    inputs["x"][1].copy_(inputs["x"][0])
    inputs["ada_in"][1].copy_(inputs["ada_in"][0])
    out = model(**inputs, particle_ids=torch.tensor([0, 1]))

    assert out.shape == (2, 3, 1)
    assert not torch.allclose(out[0], out[1])
    counts = model.anchored_low_rank_parameter_counts()
    assert counts["adapter_trainable"] > 0
    assert counts["adapter_trainable_fraction"] < 0.25


def test_gino_alr_requires_particle_ids():
    model = _tiny_gino(
        anchored_low_rank={
            "enabled": True,
            "num_particles": 2,
            "rank": 1,
            "anchor_relative_norm": 0.01,
            "anchor_seed": 53,
            "fno_last_n_blocks": 2,
        }
    )
    try:
        model(**_tiny_gino_inputs())
    except ValueError as exc:
        assert "particle_ids" in str(exc)
    else:
        raise AssertionError("ALR-GINO must require explicit particle_ids.")
