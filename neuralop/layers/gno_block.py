from typing import List, Optional, Callable, Dict
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from .channel_mlp import LinearChannelMLP
from .integral_transform import IntegralTransform
from .neighbor_search import NeighborSearch
from .embeddings import SinusoidalEmbedding

# -------- Optional: define or import your weighting cutoff functions -------- #
def linear_cutoff(x, radius=1., scale=1.):
    x = (radius - x).clip(0., radius)
    return x * scale / radius

def bump_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = x.clip(0., radius) / radius
    out = -1.0 / ((1.0 - out ** 2) + eps)
    return out.exp() * torch.e * scale

def tanh_cutoff(x, radius=1., scale=1., slope=2.0, eps=1e-6):
    out = x.clip(0., radius) / radius
    out = slope * (2 * out - 1) / (2 * torch.sqrt((1 - out) * out) + eps)
    out = -0.5 * torch.tanh(out) + 0.5
    return out * scale

def cubic_cutoff(x, radius=1., scale=1.):
    b = 3 * scale / (radius ** 2)
    a = 2 * b / (3 * radius)
    out = a * x ** 3 - b * x ** 2 + scale
    return out

def quadr_cutoff(x, radius=1., scale=1.):
    x = x / radius
    left = 1 - 2 * x ** 2
    right = 2 * (1 - x) ** 2
    return scale * torch.where(x < 0.5, left, right)

def cos_cutoff(x, radius=1., scale=1.):
    x = x / radius
    return scale * (0.5 * torch.cos(torch.pi * x) + 0.5)

def bump_sqrt_cutoff(x, radius=1., scale=1., eps=1e-7):
    out = -1.0 / (1.0 - x / radius + eps)
    return out.exp() * torch.e * scale

def edam_weight(distance, radius=1.0, scale=1.0, eps=1e-7):
    """
    Compute a smooth, differentiable weight based on the Euclidean distance.
    For a pair (x,y) with distance d, returns:
       w = exp( - r^2 / (r^2 - d^2) )   for d < r, and 0 for d >= r,
    scaled by 'scale'.
    """
    # distance is the Euclidean distance (not squared)
    weight = torch.where(
        distance < radius,
        torch.exp(- (radius ** 2) / (radius ** 2 - distance ** 2 + eps)),
        torch.zeros_like(distance)
    )
    return weight * scale


# --------------------------------------------------------------------------- #

class GNOBlock(nn.Module):
    """
    GNOBlock implements a Graph Neural Operator layer.

    It precomputes static components (such as neighbor indices) when the input coordinates are fixed.
    The positional embeddings are computed on the fly every forward pass so that they remain part of the
    computation graph (allowing for differentiation with respect to spatial coordinates).

    Parameters
    ----------
    in_channels : int
        Number of channels in the input function.
    out_channels : int
        Number of channels in the output function.
    coord_dim : int
        Dimension of the coordinates.
    radius : float
        Neighborhood search radius.
    transform_type : str, optional
        Type of transform; defaults to "linear".
    pos_embedding_type : Optional[str], default='transformer'
        Positional embedding style.
    pos_embedding_channels : int, default=32
    pos_embedding_max_positions : int, default=10000
    channel_mlp_layers : List[int], optional
        Layer widths for the MLP.
    channel_mlp_non_linearity : Callable, optional
        Nonlinearity for the MLP.
    channel_mlp : Optional[nn.Module], default=None
        Custom MLP module.
    use_open3d_neighbor_search : bool, default=True
        Whether to use Open3D for neighbor search (requires 3D data).
    use_torch_scatter_reduce : bool, default=True
        Whether to use torch_scatter.
    gno_weighting_fn : Optional[str], default=None
        Name of the weighting function.
    gno_wt_fn_scale : float, default=1.0
        Scale factor for the weighting function.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            coord_dim: int,
            radius: float,
            transform_type: str = "linear",
            pos_embedding_type: Optional[str] = 'transformer',
            pos_embedding_channels: int = 32,
            pos_embedding_max_positions: int = 10000,
            channel_mlp_layers: Optional[List[int]] = [128, 256, 128],
            channel_mlp_non_linearity=F.gelu,
            channel_mlp: Optional[nn.Module] = None,
            use_open3d_neighbor_search: bool = True,
            use_torch_scatter_reduce: bool = True,
            gno_weighting_fn: Optional[str] = None,
            gno_wt_fn_scale: float = 1.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coord_dim = coord_dim
        self.radius = radius
        self.transform_type = transform_type

        # Positional embedding is defined here but will be computed fresh on each forward pass.
        self.pos_embedding_type = pos_embedding_type
        self.pos_embedding: Optional[nn.Module] = None
        if self.pos_embedding_type is not None:
            self.pos_embedding = SinusoidalEmbedding(
                in_channels=coord_dim,
                num_frequencies=pos_embedding_channels,
                embedding_type=pos_embedding_type,
                max_positions=pos_embedding_max_positions
            )

        # Neighbor search (if using Open3D, ensure coord_dim==3)
        if use_open3d_neighbor_search and self.coord_dim != 3:
            raise ValueError(
                f"Open3D neighbor search requires 3D data, but coord_dim={self.coord_dim} was provided."
            )
        self.neighbor_search = NeighborSearch(use_open3d=use_open3d_neighbor_search)

        # Determine kernel input dimension (see original implementation)
        embed_dim = self.pos_embedding.out_channels if self.pos_embedding else self.coord_dim
        kernel_in_dim = 2 * embed_dim
        if self.transform_type in {"nonlinear", "nonlinear_kernelonly"}:
            kernel_in_dim += self.in_channels

        # Channel MLP
        if channel_mlp is not None:
            self.channel_mlp = channel_mlp
        else:
            if channel_mlp_layers is None:
                raise ValueError("Either channel_mlp or channel_mlp_layers must be provided.")
            mlp_layers = [kernel_in_dim] + channel_mlp_layers
            if mlp_layers[-1] != self.out_channels:
                mlp_layers.append(self.out_channels)
            self.channel_mlp = LinearChannelMLP(layers=mlp_layers, non_linearity=channel_mlp_non_linearity)

        # Integral Transform
        self.integral_transform = IntegralTransform(
            channel_mlp=self.channel_mlp,
            transform_type=self.transform_type,
            use_torch_scatter=use_torch_scatter_reduce
        )

        # Weighting function logic
        self.gno_weighting_fn = None
        if gno_weighting_fn is not None:
            sq_radius = self.radius ** 2
            if gno_weighting_fn == "linear":
                self.gno_weighting_fn = partial(linear_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "bump":
                self.gno_weighting_fn = partial(bump_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "tanh":
                self.gno_weighting_fn = partial(tanh_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "cubic":
                self.gno_weighting_fn = partial(cubic_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "cos":
                self.gno_weighting_fn = partial(cos_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "quadr":
                self.gno_weighting_fn = partial(quadr_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "bump_sqrt":
                self.gno_weighting_fn = partial(bump_sqrt_cutoff, radius=sq_radius, scale=gno_wt_fn_scale)
            elif gno_weighting_fn == "edam":
                self.gno_weighting_fn = partial(edam_weight, radius=radius, scale=gno_wt_fn_scale)
            else:
                raise NotImplementedError(f"Unknown weighting function '{gno_weighting_fn}'")

        # Caching placeholders: only cache the neighbor dictionary (and the original inputs for verification)
        self.register_buffer('cached_y_original', None, persistent=False)
        self.register_buffer('cached_x_original', None, persistent=False)
        self.register_buffer('cached_neighbors', None, persistent=False)
        self._is_cached = False
        self._is_verified = False

    def precompute_static_components(self, y: torch.Tensor, x: torch.Tensor):
        """
        Precompute and cache neighbor indices for static y and x.
        We do NOT cache the positional embeddings so that they remain in the computation graph.
        """
        if not (y.requires_grad or x.requires_grad):
            with torch.no_grad():
                neighbors_dict = self.neighbor_search(
                    data=y,
                    queries=x,
                    radius=self.radius,
                    compute_norm=False
                )
                neighbors_dict = {k: v.to(y.device) for k, v in neighbors_dict.items()}
                self.cached_neighbors = neighbors_dict
                self.cached_y_original = y.clone()
                self.cached_x_original = x.clone()
            self._is_cached = True
            self._is_verified = True
        else:
            # Do not cache if inputs require grad.
            self._is_cached = False

    def _verify_cached_components(self, y: torch.Tensor, x: torch.Tensor):
        if y.shape != self.cached_y_original.shape:
            raise ValueError("Input tensor y shape differs from cached shape.")
        if x.shape != self.cached_x_original.shape:
            raise ValueError("Input tensor x shape differs from cached shape.")
        if y.device != self.cached_y_original.device:
            raise ValueError("Input tensor y is on a different device than cached.")
        if x.device != self.cached_x_original.device:
            raise ValueError("Input tensor x is on a different device than cached.")
        if not torch.equal(y, self.cached_y_original):
            raise ValueError("Input tensor y has changed since precomputation. Re-run precompute_static_components.")
        if not torch.equal(x, self.cached_x_original):
            raise ValueError("Input tensor x has changed since precomputation. Re-run precompute_static_components.")
        self._is_verified = True

    def forward(self, y: torch.Tensor, x: torch.Tensor, f_y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the output function evaluated at x by performing the kernel integral transform.
        Always compute the positional embeddings fresh to ensure they are part of the computation graph.
        If caching is available, only the neighbor dictionary is reused.
        """
        if y.requires_grad or x.requires_grad or (not self._is_cached):
            if self.pos_embedding is not None:
                y_embed = self.pos_embedding(y)
                x_embed = self.pos_embedding(x)
            else:
                y_embed = y
                x_embed = x
            neighbors_dict = self.neighbor_search(
                data=y,
                queries=x,
                radius=self.radius,
                compute_norm=(self.gno_weighting_fn is not None)
            )
            neighbors_dict = {k: v.to(y.device) for k, v in neighbors_dict.items()}
        else:
            if not self._is_cached:
                self.precompute_static_components(y, x)
                self._verify_cached_components(y, x)
            # Always compute embeddings fresh so that gradients are tracked.
            if self.pos_embedding is not None:
                y_embed = self.pos_embedding(y)
                x_embed = self.pos_embedding(x)
            else:
                y_embed = y
                x_embed = x
            neighbors_dict = self.cached_neighbors

        out_features = self.integral_transform(
            y=y_embed,
            neighbors=neighbors_dict,
            x=x_embed,
            f_y=f_y,
            weighting_fn=self.gno_weighting_fn
        )
        return out_features

    def reset_verification(self):
        """
        Reset the internal verification flags (or states) of the GNO block.
        """
        self._is_verified = False
        self._is_cached = False

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self.cached_neighbors is not None:
            for key in self.cached_neighbors:
                self.cached_neighbors[key] = self.cached_neighbors[key].to(*args, **kwargs)
        return self
