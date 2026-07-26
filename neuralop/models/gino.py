from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your base classes & layers from wherever they reside.
# For example:
from .base_model import BaseModel
from ..layers.channel_mlp import ChannelMLP
from ..layers.embeddings import SinusoidalEmbedding
from ..layers.fno_block import FNOBlocks
from ..layers.spectral_convolution import SpectralConv
from ..layers.gno_block import GNOBlock
from ..layers.anchored_low_rank import (
    AnchoredLowRankDenseAdapter,
    AnchoredLowRankSpectralAdapter,
)


def _config_value(config, key, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)

class GINO(BaseModel):
    """
    GINO: Geometry-Informed Neural Operator, with an optional autoregressive
    residual connection for PDE time-stepping.

    By default, GINO uses:
      1) An input GNOBlock (self.gno_in)
      2) A lifting MLP
      3) A series of FNO blocks in a latent 2D/3D domain
      4) An output GNOBlock (self.gno_out)
      5) A final projection MLP to the desired output dimension
      6) (Optionally) a skip connection from x[..., :out_channels] to the output

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., WD, VX, VY, plus optionally more).
    out_channels : int
        Number of output channels to predict (e.g. 3 for WD, VX, VY).
    latent_feature_channels : int, optional
        If not None, additional channels appended before FNO blocks.
    projection_channels : int, optional
        Number of hidden channels in the final projection MLP, default=256.
    gno_coord_dim : int, optional
        The dimension of coordinates that the GNO blocks expect, default=3.
        E.g. 2D geometry => set gno_coord_dim=2. If open3d is used, it expects 3D.
    gno_radius : float, optional
        Neighborhood search radius for GNO blocks, default=0.033.
    in_gno_transform_type : {'linear','nonlinear','residual',...}, optional
        The transform type for the input GNO block, default='linear'.
    out_gno_transform_type : str, optional
        The transform type for the output GNO block, default='linear'.
    gno_pos_embed_type : {'transformer','nerf'} or None, optional
        The positional embedding style used within the GNO blocks, default='transformer'.
    fno_in_channels : int, optional
        The channel dimension that the FNO expects *after* the input GNO, default=3.
    fno_n_modes : tuple, optional
        The number of Fourier modes along each dimension, default=(16,16,16).
    fno_hidden_channels : int, optional
        The hidden channel size in each FNO layer, default=64.
    fno_lifting_channel_ratio : int, optional
        Multiplier to define how many channels to use in the MLP-lifting layer, default=2.
    fno_n_layers : int, optional
        Number of FNO layers, default=4.

    # GNO details
    gno_embed_channels : int, default=32
        Dim of the optional positional embedding in GNO.
    gno_embed_max_positions : int, default=10000
        Max positions for the sinusoidal embedding if pos_embed_type='transformer'.
    in_gno_channel_mlp_hidden_layers : list, default=[80,80,80]
        Widths of hidden layers in the input GNO block’s channel MLP.
    out_gno_channel_mlp_hidden_layers : list, default=[512,256]
        Widths of hidden layers in the output GNO block’s channel MLP.
    gno_channel_mlp_non_linearity : callable, default=F.gelu
        Nonlinear function used in the GNO channel MLP.
    gno_use_open3d : bool, default=True
        Whether to use open3d’s neighbor search if available (3D only).
    gno_use_torch_scatter : bool, default=True
        Whether to use torch_scatter-based integral transform ops in GNO.
    out_gno_tanh : str or None, default=None
        If 'latent_embed', 'both', etc., applies tanh after GNO out. By default, no tanh.

    # NEW weighting function arguments
    gno_in_weighting_fn : Optional[str], default=None
        Name of the weighting function for the input GNO block (e.g. "linear", "bump", etc.).
    gno_in_wt_fn_scale : float, default=1.0
        Scale factor for the input GNO weighting function.
    gno_out_weighting_fn : Optional[str], default=None
        Name of the weighting function for the output GNO block.
    gno_out_wt_fn_scale : float, default=1.0
        Scale factor for the output GNO weighting function.

    # FNO details
    fno_resolution_scaling_factor : float or None, default=None
        If not None, up/down scale the size of the latent grid for the FNO.
    fno_incremental_n_modes : list[int] or None, default=None
        If provided, changes n_modes layer by layer in the FNO.
    fno_block_precision : {'full','half','amp'}, default='full'
        Controls precision in the FNO blocks.
    fno_use_channel_mlp : bool, default=True
        Whether to insert a channel MLP after each FNO layer.
    fno_channel_mlp_dropout : float, default=0
        Dropout probability in the channel MLP.
    fno_channel_mlp_expansion : float, default=0.5
        Expansion ratio in the channel MLP.
    fno_non_linearity : callable, default=F.gelu
        Activation function in the FNO layers.
    fno_stabilizer : callable or None, default=None
        If not None, an activation to apply before FFT in the FNO.
    fno_norm : str or None, default=None
        Type of normalization to use inside the FNO blocks. E.g., 'instance', 'ada_in'.
    fno_ada_in_features : int or None, default=4
        If using 'ada_in' normalization, how many features to embed for AdaIN.
    fno_ada_in_dim : int, default=1
        The dimension used for AdaIN features if fno_ada_in_features is not None.
    use_fgn_noise : bool, default=False
        If True and fno_norm=='ada_in', use FGN-style noise: z ~ N(0,I) encoded
        by a learned linear map (no sinusoidal embed). For probabilistic ensembles.
    fgn_noise_dim : int, default=32
        Dimension of the FGN noise vector z when use_fgn_noise=True.
        Canonical runtime shape is [B, fgn_noise_dim] (with [fgn_noise_dim] also accepted).
    output_distribution : {'deterministic','gaussian'}, default='deterministic'
        Output head type. 'deterministic' returns out_channels values.
        'gaussian' returns packed [mu, logvar] with size 2*out_channels.
    fno_preactivation : bool, default=False
        If True, use a pre-act style in the FNO blocks (like a pre-activation ResNet).
    fno_skip : str, default='linear'
        The skip connection style in each FNO layer.
    fno_channel_mlp_skip : str, default='soft-gating'
        Type of skip connection in the channel MLP inside the FNO blocks.
    fno_separable : bool, default=False
        If True, use depthwise separable spectral convolutions in the FNO.
    fno_factorization : {'tucker','tt','cp'} or None, default=None
        If not None, factorize the spectral convolution weight.
    fno_rank : float, default=1.0
        Rank ratio if factorizing the spectral convolution.
    fno_joint_factorization : bool, default=False
        If True, share one factorization among all FNO layers.
    fno_fixed_rank_modes : bool, default=False
        If True, do not factorize certain low-frequency modes.
    fno_implementation : {'factorized','reconstructed'} or None, default='factorized'
        If factorization is used, how to apply it at forward time.
    fno_decomposition_kwargs : dict, default={}
        Additional parameters for the decomposition.
    fno_conv_module : nn.Module, default=SpectralConv
        Which spectral convolution module to use in the FNO.

    autoregressive : bool, default=False
        If True, the final output is added to x[..., :out_channels] to form
        the final prediction (residual skip). This is useful for PDE time-stepping
        where x is the previous state and out is the predicted delta.

    Returns
    -------
    out : torch.Tensor, shape (B, n_out, out_channels)
        The predicted output field on `output_queries`.
        If `autoregressive=True`, it becomes x[..., :out_channels] + model_prediction.

    Example
    -------
    >>> model = GINO(in_channels=3, out_channels=3, gno_coord_dim=2, autoregressive=True)
    >>> # Suppose:
    >>> #   input_geom => (1, N_in, 2)
    >>> #   latent_queries => (1, H, W, 2)
    >>> #   output_queries => (1, N_out, 2)
    >>> #   x => (batch_size, N_in, 3)    # WD, VX, VY at previous time
    >>> y_next = model(input_geom, latent_queries, output_queries, x)
    >>> # shape => (batch_size, N_out, 3)
    >>> #  y_next = x[..., :3] + predicted_delta if autoregressive=True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        latent_feature_channels=None,
        projection_channels=256,
        gno_coord_dim=3,
        gno_radius=0.033,
        in_gno_transform_type='linear',
        out_gno_transform_type='linear',
        gno_pos_embed_type='transformer',
        fno_in_channels=3,
        fno_n_modes=(16, 16, 16),
        fno_hidden_channels=64,
        fno_lifting_channel_ratio=2,
        fno_n_layers=4,
        gno_embed_channels=32,
        gno_embed_max_positions=10000,
        in_gno_channel_mlp_hidden_layers=[80, 80, 80],
        out_gno_channel_mlp_hidden_layers=[512, 256],
        gno_channel_mlp_non_linearity=F.gelu,
        gno_use_open3d=True,
        gno_use_torch_scatter=True,
        out_gno_tanh=None,
        # NEW weighting parameters for GNO
        gno_in_weighting_fn=None,
        gno_in_wt_fn_scale=1.0,
        gno_out_weighting_fn=None,
        gno_out_wt_fn_scale=1.0,
        # FNO extras
        fno_resolution_scaling_factor=None,
        fno_incremental_n_modes=None,
        fno_block_precision='full',
        fno_use_channel_mlp=True,
        fno_channel_mlp_dropout=0,
        fno_channel_mlp_expansion=0.5,
        fno_non_linearity=F.gelu,
        fno_stabilizer=None,
        fno_norm=None,
        fno_ada_in_features=4,
        fno_ada_in_dim=1,
        use_fgn_noise=False,
        fgn_noise_dim=32,
        fgn_latent_temporal_mode="stepwise",
        fno_preactivation=False,
        fno_skip='linear',
        fno_channel_mlp_skip='soft-gating',
        fno_separable=False,
        fno_factorization=None,
        fno_rank=1.0,
        fno_joint_factorization=False,
        fno_fixed_rank_modes=False,
        fno_implementation='factorized',
        fno_decomposition_kwargs=dict(),
        fno_conv_module=None,  # e.g. SpectralConv
        # NEW ARG
        autoregressive=False,
        alpha: float = 1.0,
        beta:  float = 1.0,
        output_distribution: str = "deterministic",
        anchored_low_rank=None,
        **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_feature_channels = latent_feature_channels
        self.gno_coord_dim = gno_coord_dim
        self.fno_hidden_channels = fno_hidden_channels
        self.lifting_channels = fno_lifting_channel_ratio * fno_hidden_channels

        self.autoregressive = autoregressive  # controls residual skip
        self.alpha = alpha
        self.beta  = beta
        self.output_distribution = str(output_distribution).strip().lower()
        if self.output_distribution not in {"deterministic", "gaussian"}:
            raise ValueError(
                "output_distribution must be one of {'deterministic', 'gaussian'}, "
                f"got {output_distribution!r}."
            )

        # if in_gno_transform_type is 'linear' or 'nonlinear',
        # we assume out_channels = in_channels
        if in_gno_transform_type in ["linear", "nonlinear"]:
            in_gno_out_channels = self.in_channels
        else:
            in_gno_out_channels = fno_in_channels

        self.fno_in_channels = in_gno_out_channels
        if latent_feature_channels is not None:
            self.fno_in_channels += latent_feature_channels

        if self.gno_coord_dim != 3 and gno_use_open3d:
            print(
                f'Warning: GNO expects {self.gno_coord_dim}-d data but Open3D expects 3-d data'
            )
            gno_use_open3d = False

        self.in_coord_dim = len(fno_n_modes)
        self.gno_out_coord_dim = len(fno_n_modes)
        if self.in_coord_dim != self.gno_coord_dim:
            print(
                f'Warning: FNO expects {self.in_coord_dim}-d data while input GNO expects {self.gno_coord_dim}-d data'
            )

        self.in_coord_dim_forward_order = list(range(self.in_coord_dim))
        self.in_coord_dim_reverse_order = [j + 2 for j in self.in_coord_dim_forward_order]

        self.fno_norm = fno_norm
        self.use_fgn_noise = use_fgn_noise
        self.fgn_noise_dim = fgn_noise_dim
        self.fgn_latent_temporal_mode = str(fgn_latent_temporal_mode).strip().lower()

        if use_fgn_noise and fno_norm != "ada_in":
            raise ValueError(
                "FGN (use_fgn_noise=True) requires fno_norm='ada_in'. "
                f"Got fno_norm={fno_norm!r}. Set fno_norm: 'ada_in' in config when using FGN."
            )
        if self.output_distribution == "gaussian" and use_fgn_noise:
            raise ValueError(
                "output_distribution='gaussian' is incompatible with use_fgn_noise=True. "
                "Use one probabilistic path at a time."
            )

        # If we have "ada_in" normalization in FNO
        if self.fno_norm == "ada_in":
            if use_fgn_noise:
                # FGN: low-dim noise z encoded by learned linear map (arXiv:2506.10772)
                self.fgn_noise_encoder = nn.Linear(fgn_noise_dim, fgn_noise_dim)
                self.adain_pos_embed = None
                self.ada_in_dim = fgn_noise_dim
            elif fno_ada_in_features is not None and gno_pos_embed_type is not None:
                # E.g. a sinusoidal embedding (e.g. diffusion noise level)
                self.fgn_noise_encoder = None
                self.adain_pos_embed = SinusoidalEmbedding(
                    in_channels=fno_ada_in_dim,
                    num_frequencies=fno_ada_in_features,
                    max_positions=10000,
                    embedding_type=gno_pos_embed_type
                )
                self.ada_in_dim = self.adain_pos_embed.out_channels
            else:
                self.fgn_noise_encoder = None
                self.adain_pos_embed = None
                self.ada_in_dim = fno_ada_in_dim
        else:
            self.fgn_noise_encoder = None
            self.adain_pos_embed = None
            self.ada_in_dim = None

        self.gno_radius = gno_radius
        self.out_gno_tanh = out_gno_tanh

        # 1) Input GNO
        self.gno_in = GNOBlock(
            in_channels=in_channels,
            out_channels=in_gno_out_channels,
            coord_dim=self.gno_coord_dim,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            radius=gno_radius,
            channel_mlp_layers=in_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=in_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
            # NEW weighting arguments
            gno_weighting_fn=gno_in_weighting_fn,
            gno_wt_fn_scale=gno_in_wt_fn_scale
        )

        # 2) Lifting MLP
        self.lifting = ChannelMLP(
            in_channels=self.fno_in_channels,
            hidden_channels=self.lifting_channels,
            out_channels=fno_hidden_channels,
            n_layers=3
        )

        # 3) FNO Blocks
        if fno_conv_module is None:
            # fallback to default spectral conv if none provided
            fno_conv_module = SpectralConv

        self.fno_blocks = FNOBlocks(
            n_modes=fno_n_modes,
            hidden_channels=fno_hidden_channels,
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            positional_embedding=None,
            n_layers=fno_n_layers,
            resolution_scaling_factor=fno_resolution_scaling_factor,
            incremental_n_modes=fno_incremental_n_modes,
            fno_block_precision=fno_block_precision,
            use_channel_mlp=fno_use_channel_mlp,
            channel_mlp_expansion=fno_channel_mlp_expansion,
            channel_mlp_dropout=fno_channel_mlp_dropout,
            non_linearity=fno_non_linearity,
            stabilizer=fno_stabilizer,
            norm=fno_norm,
            ada_in_features=self.ada_in_dim,
            preactivation=fno_preactivation,
            fno_skip=fno_skip,
            channel_mlp_skip=fno_channel_mlp_skip,
            separable=fno_separable,
            factorization=fno_factorization,
            rank=fno_rank,
            joint_factorization=fno_joint_factorization,
            fixed_rank_modes=fno_fixed_rank_modes,
            implementation=fno_implementation,
            decomposition_kwargs=fno_decomposition_kwargs,
            domain_padding=None,
            domain_padding_mode=None,
            conv_module=fno_conv_module,
            **kwargs
        )

        # 4) Output GNO
        self.gno_out = GNOBlock(
            in_channels=fno_hidden_channels,
            out_channels=fno_hidden_channels,
            coord_dim=self.gno_coord_dim,
            radius=self.gno_radius,
            pos_embedding_type=gno_pos_embed_type,
            pos_embedding_channels=gno_embed_channels,
            pos_embedding_max_positions=gno_embed_max_positions,
            channel_mlp_layers=out_gno_channel_mlp_hidden_layers,
            channel_mlp_non_linearity=gno_channel_mlp_non_linearity,
            transform_type=out_gno_transform_type,
            use_open3d_neighbor_search=gno_use_open3d,
            use_torch_scatter_reduce=gno_use_torch_scatter,
            # NEW weighting arguments
            gno_weighting_fn=gno_out_weighting_fn,
            gno_wt_fn_scale=gno_out_wt_fn_scale
        )

        # 5) Final projection => out_channels
        if self.output_distribution == "gaussian":
            self.projection = None
            self.mu_head = ChannelMLP(
                in_channels=fno_hidden_channels,
                out_channels=self.out_channels,
                hidden_channels=projection_channels,
                n_layers=2,
                n_dim=1,
                non_linearity=fno_non_linearity,
            )
            self.logvar_head = ChannelMLP(
                in_channels=fno_hidden_channels,
                out_channels=self.out_channels,
                hidden_channels=projection_channels,
                n_layers=2,
                n_dim=1,
                non_linearity=fno_non_linearity,
            )
        else:
            self.projection = ChannelMLP(
                in_channels=fno_hidden_channels,
                out_channels=self.out_channels,
                hidden_channels=projection_channels,
                n_layers=2,
                n_dim=1,
                non_linearity=fno_non_linearity
            )
            self.mu_head = None
            self.logvar_head = None

        self._configure_anchored_low_rank(anchored_low_rank)

    def _configure_anchored_low_rank(self, config):
        self.anchored_low_rank_enabled = bool(_config_value(config, "enabled", False))
        self.anchored_low_rank_active = self.anchored_low_rank_enabled
        self.anchored_low_rank_num_particles = int(
            _config_value(config, "num_particles", 4)
        )
        self.anchored_low_rank_rank = int(_config_value(config, "rank", 4))
        if not self.anchored_low_rank_enabled:
            return
        self.register_buffer(
            "anchored_low_rank_base_validation_rmse",
            torch.tensor(float("nan"), dtype=torch.float32),
        )
        if self.output_distribution != "deterministic":
            raise ValueError("ALR-FGNO pilot requires deterministic Stage-1 output heads.")
        if not self.use_fgn_noise:
            raise ValueError("ALR-FGNO requires the existing FGNO aleatory latent path.")
        if self.anchored_low_rank_num_particles < 2:
            raise ValueError("ALR-FGNO requires at least two epistemic particles.")
        if self.anchored_low_rank_rank <= 0:
            raise ValueError("ALR-FGNO adapter rank must be positive.")

        seed = int(_config_value(config, "anchor_seed", 20260724))
        relative_norm = float(_config_value(config, "anchor_relative_norm", 0.01))
        self.fno_blocks.enable_anchored_low_rank(
            last_n_blocks=int(_config_value(config, "fno_last_n_blocks", 2)),
            num_particles=self.anchored_low_rank_num_particles,
            rank=self.anchored_low_rank_rank,
            anchor_relative_norm=relative_norm,
            seed=seed,
            adapt_spectral=bool(_config_value(config, "adapt_spectral", True)),
            adapt_pointwise=bool(_config_value(config, "adapt_pointwise", True)),
        )
        if bool(_config_value(config, "adapt_output_gno", True)):
            self.gno_out.enable_anchored_low_rank(
                final_n_layers=2,
                num_particles=self.anchored_low_rank_num_particles,
                rank=self.anchored_low_rank_rank,
                anchor_relative_norm=relative_norm,
                seed=seed + 1_000_003,
            )
        if bool(_config_value(config, "adapt_output_projection", True)):
            self.projection.enable_anchored_low_rank(
                layer_indices=range(self.projection.n_layers),
                num_particles=self.anchored_low_rank_num_particles,
                rank=self.anchored_low_rank_rank,
                anchor_relative_norm=relative_norm,
                seed=seed + 2_000_003,
            )
        if bool(_config_value(config, "adapt_forcing_encoder", False)):
            raise ValueError("adapt_forcing_encoder is outside the ALR-FGNO pilot scope.")

    def anchored_low_rank_adapters(self):
        for module in self.modules():
            if isinstance(
                module,
                (AnchoredLowRankDenseAdapter, AnchoredLowRankSpectralAdapter),
            ):
                yield module

    def anchored_low_rank_parameter_counts(self):
        adapter_ids = {
            id(parameter)
            for adapter in self.anchored_low_rank_adapters()
            for parameter in adapter.parameters()
        }
        adapter_trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) in adapter_ids and parameter.requires_grad
        )
        shared = sum(
            parameter.numel()
            for parameter in self.parameters()
            if id(parameter) not in adapter_ids
        )
        anchor_buffers = sum(
            buffer.numel()
            for adapter in self.anchored_low_rank_adapters()
            for buffer in adapter.buffers()
        )
        return {
            "shared": int(shared),
            "adapter_trainable": int(adapter_trainable),
            "anchor_buffers": int(anchor_buffers),
            "adapter_trainable_fraction": float(adapter_trainable / max(1, shared)),
        }

    def anchored_low_rank_offset_penalty(self):
        penalties = [adapter.offset_penalty() for adapter in self.anchored_low_rank_adapters()]
        if penalties:
            return torch.stack(penalties).sum()
        return next(self.parameters()).new_zeros(())

    def set_anchored_low_rank_active(self, active):
        """Enable or bypass every particle adapter without changing the backbone."""
        self.anchored_low_rank_active = bool(active) and self.anchored_low_rank_enabled
        for adapter in self.anchored_low_rank_adapters():
            adapter.active = self.anchored_low_rank_active

    def set_anchored_low_rank_training_phase(self, *, adapters_only):
        adapter_parameter_ids = {
            id(parameter)
            for adapter in self.anchored_low_rank_adapters()
            for parameter in adapter.parameters()
        }
        for parameter in self.parameters():
            if id(parameter) in adapter_parameter_ids:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(not bool(adapters_only))

    def latent_embedding(self, in_p, ada_in=None, particle_ids=None):
        """
        Internal helper: pass through self.lifting -> self.fno_blocks.

        in_p shape: (batch, d_1, d_2,..., d_k, channels)
        Returns: (batch, fno_hidden_channels, d_1, d_2,..., d_k)
        """
        # Permute channel to second dimension => (b,c,d_1,d_2,...)
        in_p = in_p.permute(0, in_p.ndim-1, *range(1, in_p.ndim-1))
        batch_size = in_p.shape[0]

        # AdaIN embedding: canonical contract is [B, D], while [D] and [1, D] are allowed.
        ada_in_embed = None
        raw_ada_in = None
        if ada_in is not None:
            if ada_in.ndim == 1:
                raw_ada_in = ada_in.unsqueeze(0)
            elif ada_in.ndim == 2:
                raw_ada_in = ada_in
            else:
                raise ValueError(
                    f"ada_in must have shape [D], [1, D], or [B, D], got shape {tuple(ada_in.shape)}."
                )

            if raw_ada_in.shape[0] == 1 and batch_size > 1:
                raw_ada_in = raw_ada_in.expand(batch_size, -1)
            elif raw_ada_in.shape[0] != batch_size:
                raise ValueError(
                    f"ada_in batch ({raw_ada_in.shape[0]}) must be 1 or match model batch ({batch_size})."
                )
        elif self.use_fgn_noise and self.fgn_noise_encoder is not None and self.fno_norm == "ada_in":
            # Deterministic fallback for callers that omit ada_in.
            raw_ada_in = torch.zeros(
                (batch_size, self.fgn_noise_dim), device=in_p.device, dtype=in_p.dtype
            )

        if raw_ada_in is not None:
            if self.use_fgn_noise and self.fgn_noise_encoder is not None:
                if raw_ada_in.shape[1] != self.fgn_noise_dim:
                    raise ValueError(
                        f"FGN expected ada_in dim {self.fgn_noise_dim}, got {raw_ada_in.shape[1]}."
                    )
                ada_in_embed = self.fgn_noise_encoder(raw_ada_in)
            elif self.adain_pos_embed is not None:
                if raw_ada_in.shape[1] != self.adain_pos_embed.in_channels:
                    raise ValueError(
                        f"AdaIN positional embedding expected dim {self.adain_pos_embed.in_channels}, "
                        f"got {raw_ada_in.shape[1]}."
                    )
                ada_in_embed = self.adain_pos_embed(raw_ada_in)
            else:
                if self.ada_in_dim is not None and raw_ada_in.shape[1] != self.ada_in_dim:
                    raise ValueError(
                        f"AdaIN expected embedding dim {self.ada_in_dim}, got {raw_ada_in.shape[1]}."
                    )
                ada_in_embed = raw_ada_in

        # Lifting
        in_p = self.lifting(in_p)
        for idx in range(self.fno_blocks.n_layers):
            in_p = self.fno_blocks(
                in_p,
                idx,
                ada_in_embed=ada_in_embed,
                particle_ids=particle_ids,
            )

        return in_p

    def forward(
        self,
        input_geom,
        latent_queries,
        output_queries,
        x=None,
        latent_features=None,
        ada_in=None,
        particle_ids=None,
        return_features=False,
        feature_source="decoder_pre_projection",
        **kwargs
    ):
        """
        Forward pass of GINO. If autoregressive=True, a skip connection is applied
        so the final output = x[..., :out_channels] + predicted_delta.

        Parameters
        ----------
        input_geom : torch.Tensor, shape (1, n_in, gno_coord_dim)
            Coordinates for the input domain.
        latent_queries : torch.Tensor, shape (1, d_1, ..., d_k, gno_coord_dim)
            The intermediate "latent domain" grid for the FNO.
        output_queries : torch.Tensor, shape (1, n_out, gno_coord_dim)
            Coordinates at which we want final predictions.
        x : torch.Tensor, shape (batch, n_in, in_channels), optional
            Input field(s) on input_geom. If None, batch=1 is assumed.
            If autoregressive=True, x[..., :out_channels] will be used for skip.
        latent_features : torch.Tensor, optional
            Additional channels for the latent domain, shape (b, d_1,..., d_k, C).
        ada_in : torch.Tensor, optional
            If using AdaIN in the FNO, pass conditioning with canonical shape [B, D].
            Backward-compatible shapes [D] and [1, D] are also accepted.

        Returns
        -------
        out : torch.Tensor
            If output_distribution == 'deterministic': shape (batch, n_out, out_channels).
            If output_distribution == 'gaussian': shape (batch, n_out, 2*out_channels),
            packed as [mu, logvar] along the last dimension.
            If self.autoregressive, mu is updated by skip connection.
        """
        if x is None:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        if self.anchored_low_rank_enabled and particle_ids is None:
            raise ValueError("particle_ids are required when ALR-FGNO is enabled.")

        # Possibly broadcast latent_features if batch=1
        if latent_features is not None:
            assert self.latent_feature_channels is not None, (
                "Must set latent_feature_channels if passing latent_features."
            )
            if latent_features.shape[0] != batch_size:
                if latent_features.shape[0] == 1:
                    latent_features = latent_features.repeat(
                        batch_size, *([1]*(latent_features.ndim-1))
                    )

        # Squeeze leading dimension from geometry & queries => remove the (1, ...)
        input_geom = input_geom.squeeze(0)         # (n_in, gno_coord_dim)
        latent_queries = latent_queries.squeeze(0) # (d_1,...,d_k, gno_coord_dim)
        output_queries = output_queries.squeeze(0) # (n_out, gno_coord_dim)
        # 1) Input GNO: merges geometry + x
        # Flatten latent_queries => (d_1*d_2*..., gno_coord_dim)
        in_p = self.gno_in(
            y=input_geom,
            x=latent_queries.reshape(-1, latent_queries.shape[-1]),
            f_y=x
        )
        # => shape (batch, d_1*d_2*..., in_gno_out_channels)
        # => reshape => (batch, d_1, d_2,..., in_gno_out_channels)
        grid_shape = latent_queries.shape[:-1]
        in_p = in_p.view(batch_size, *grid_shape, -1)

        # 2) Concatenate latent_features if present
        if latent_features is not None:
            in_p = torch.cat((in_p, latent_features), dim=-1)

        # 3) FNO embedding
        latent_embed = self.latent_embedding(
            in_p,
            ada_in=ada_in,
            particle_ids=particle_ids,
        )

        # Possibly apply tanh if out_gno_tanh in ['latent_embed','both']
        if self.out_gno_tanh in ['latent_embed', 'both']:
            latent_embed = torch.tanh(latent_embed)

        # latent_embed => (b, c, d_1, d_2,...)
        # permute => (b, d_1*d_2..., c)
        latent_embed = latent_embed.permute(
            0, *self.in_coord_dim_reverse_order, 1
        ).reshape(batch_size, -1, self.fno_hidden_channels)

        # 4) Output GNO => merges latent_queries & output_queries
        out = self.gno_out(
            y=latent_queries.reshape(-1, latent_queries.shape[-1]),
            x=output_queries,
            f_y=latent_embed,
            particle_ids=particle_ids,
        )
        # => shape (b, c, n_out) => permute => (b, n_out, c)
        out = out.permute(0, 2, 1)
        decoder_pre_projection = out

        # 5) final projection(s)
        if self.output_distribution == "gaussian":
            mu = self.mu_head(out).permute(0, 2, 1)
            logvar = self.logvar_head(out).permute(0, 2, 1)

            # Possibly apply tanh to mean branch when requested.
            if self.out_gno_tanh == "both":
                mu = torch.tanh(mu)

            # 6) Autoregressive skip on mean branch
            if self.autoregressive and (x is not None):
                if mu.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"Autoregressive skip requires out.shape[1] == x.shape[1], "
                        f"got {mu.shape[1]} vs {x.shape[1]}."
                    )
                if self.out_channels > x.shape[2]:
                    raise ValueError(
                        f"Cannot skip-add: out_channels {self.out_channels} > in_channels {x.shape[2]}."
                    )
                prev_step = x[..., -self.out_channels:]
                mu = self.alpha * prev_step + self.beta * mu
                # If mu is scaled by beta, variance scales by beta^2.
                beta_abs = abs(float(self.beta))
                if beta_abs > 0.0:
                    logvar = logvar + (2.0 * torch.log(torch.tensor(beta_abs, device=logvar.device, dtype=logvar.dtype)))
                else:
                    logvar = torch.full_like(logvar, -30.0)

            logvar = torch.nan_to_num(logvar, nan=0.0, posinf=20.0, neginf=-30.0)
            out = torch.cat([mu, logvar], dim=-1)
        else:
            out = self.projection(out, particle_ids=particle_ids).permute(0, 2, 1)

            # Possibly apply tanh if out_gno_tanh == 'both'
            if self.out_gno_tanh == 'both':
                out = torch.tanh(out)

            # 6) Autoregressive skip: out = previous + delta
            if self.autoregressive and (x is not None):
                if out.shape[1] != x.shape[1]:
                    raise ValueError(
                        f"Autoregressive skip requires out.shape[1] == x.shape[1], "
                        f"got {out.shape[1]} vs {x.shape[1]}."
                    )
                if self.out_channels > x.shape[2]:
                    raise ValueError(
                        f"Cannot skip-add: out_channels {self.out_channels} > in_channels {x.shape[2]}."
                    )
                # skip with the final out_channels in x
                prev_step = x[..., -self.out_channels:]  # shape (b, n_in, out_channels)
                delta = out
                out   = self.alpha * prev_step + self.beta * delta

        if not return_features:
            return out

        feature_source = str(feature_source).strip().lower()
        features = {}
        if feature_source in {"decoder_pre_projection", "all"}:
            features["decoder_pre_projection"] = decoder_pre_projection
        else:
            raise ValueError(
                "Unsupported GINO feature_source "
                f"{feature_source!r}. Currently supported: 'decoder_pre_projection' or 'all'."
            )
        return {
            "prediction": out,
            "features": features,
            "feature_source": feature_source,
        }

    def reset_verifications(self):
        """
        Reset the internal verification flags (or states) of the GNO blocks.
        """
        self.gno_in.reset_verification()
        self.gno_out.reset_verification()
