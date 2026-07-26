import torch
from torch import nn
import torch.nn.functional as F

from .channel_mlp import LinearChannelMLP
from .segment_csr import segment_csr


class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined

    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on\
        each x) over which to integrate

    k : A kernel parametrized as a MLP (LinearChannelMLP)

    f : Input function to integrate against given\
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    channel_mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f.
        MLP should not be pointwise and should only operate across
        channels to preserve the discretization-invariance of the
        kernel integral.
    channel_mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the LinearChannelMLP class
    channel_mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        LinearChannelMLP class. Only used if channel_mlp_layers is
        given and channel_mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    use_torch_scatter : bool, default 'True'
        Whether to use torch_scatter's implementation of
        segment_csr or our native PyTorch version. torch_scatter
        should be installed by default, but there are known versioning
        issues on some linux builds of CPU-only PyTorch. Try setting
        to False if you experience an error from torch_scatter.
    """

    def __init__(
            self,
            channel_mlp=None,
            channel_mlp_layers=None,
            channel_mlp_non_linearity=F.gelu,
            transform_type="linear",
            use_torch_scatter=True,
    ):
        super().__init__()

        assert channel_mlp is not None or channel_mlp_layers is not None

        self.transform_type = transform_type
        self.use_torch_scatter = use_torch_scatter

        if (
                self.transform_type not in [
            "linear_kernelonly",
            "linear",
            "nonlinear_kernelonly",
            "nonlinear",
        ]
        ):
            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if channel_mlp is None:
            self.channel_mlp = LinearChannelMLP(
                layers=channel_mlp_layers,
                non_linearity=channel_mlp_non_linearity
            )
        else:
            self.channel_mlp = channel_mlp

    """
    Assumes x=y if not specified.
    Integral is taken w.r.t. the neighbors.
    If no weights are given, a Monte-Carlo approximation is made.
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f.
    """

    def forward(
            self,
            y,
            neighbors,
            x=None,
            f_y=None,
            weights=None,
            weighting_fn=None,
            particle_ids=None,
    ):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [batch, n, d3] or [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proportional to the
            volume around f(y) being integrated.
        weighting_fn : callable, default None
            If provided, used to compute pointwise weights
            from additional neighbor data (e.g., norms).
            Must return a tensor broadcastable to
            `rep_features.shape[0]`.

        Output
        ----------
        out_features : torch.Tensor of shape [batch, m, d4] or [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y

        # For each row in x, we gather its neighbors in y
        rep_features = y[neighbors["neighbors_index"]]

        # Determine if we have a batch dimension in f_y
        batched = False
        if f_y is not None:
            if f_y.ndim == 3:
                batched = True
                batch_size = f_y.shape[0]
                in_features = f_y[:, neighbors["neighbors_index"], :]
            elif f_y.ndim == 2:
                in_features = f_y[neighbors["neighbors_index"]]

        # number of neighbors for each row in x
        num_reps = (
                neighbors["neighbors_row_splits"][1:]
                - neighbors["neighbors_row_splits"][:-1]
        )

        # Repeat each x[i, :] for however many neighbors that row has
        self_features = torch.repeat_interleave(x, num_reps, dim=0)

        # Concatenate x and y to form MLP input
        agg_features = torch.cat([rep_features, self_features], dim=-1)

        # If we have a nonlinear kernel dependency on f(y)
        if f_y is not None and (
                self.transform_type == "nonlinear_kernelonly"
                or self.transform_type == "nonlinear"
        ):
            if batched:
                # replicate the (x,y) features for each example in the batch
                agg_features = agg_features.repeat(
                    [batch_size] + [1] * agg_features.ndim
                )
            # Add the f(y) features
            agg_features = torch.cat([agg_features, in_features], dim=-1)

        # A particle-adapted linear kernel must be evaluated per batch item.
        # The legacy unadapted path remains unchanged and shares one kernel field.
        adapters = getattr(self.channel_mlp, "anchored_low_rank", None)
        has_particle_adapters = bool(adapters)
        if has_particle_adapters:
            if not batched:
                raise ValueError(
                    "Particle-adapted integral transforms require batched f_y features."
                )
            if self.transform_type not in {"linear", "linear_kernelonly"}:
                raise NotImplementedError(
                    "ALR output-GNO adapters currently support linear integral transforms only."
                )
            kernel_features = agg_features.unsqueeze(0).expand(batch_size, -1, -1)
            rep_features = self.channel_mlp(
                kernel_features,
                particle_ids=particle_ids,
            )
        else:
            rep_features = self.channel_mlp(agg_features)

        # If transform type is not 'nonlinear_kernelonly',
        # multiply by input features (to get (b) or (d))
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features

        # Apply weighting
        if weights is not None:
            # Use provided explicit weights
            assert weights.ndim == 1, "Weights must be a 1D tensor."
            nbr_weights = weights[neighbors["neighbors_index"]]
            if batched:
                nbr_weights = nbr_weights.repeat([batch_size] + [1] * nbr_weights.ndim)
            rep_features = nbr_weights * rep_features
            reduction = "sum"


        elif weighting_fn is not None:

            # Instead of using 'neighbors["norm"]', recompute distances:

            # Determine number of neighbors for each query:

            num_reps = neighbors["neighbors_row_splits"][1:] - neighbors["neighbors_row_splits"][:-1]

            # Expand x (the query coordinates) so that each neighbor gets its corresponding x:

            if x is None:
                raise ValueError("x must be provided for differentiable weighting.")

            self_features = torch.repeat_interleave(x, num_reps, dim=0)

            # Get neighbor positions from y (using the cached indices):

            rep_data = y[neighbors["neighbors_index"]]

            # Compute the Euclidean distances (this operation is differentiable):

            d = torch.norm(self_features - rep_data, dim=-1)

            # Apply the (differentiable) weighting function:

            rep_weights = weighting_fn(d).unsqueeze(-1)

            if batched:
                rep_weights = rep_weights.repeat([batch_size] + [1] * rep_weights.ndim)

            rep_features = rep_features * rep_weights

            reduction = "sum"

        else:
            # Default to mean
            reduction = "mean"

        # If batching, replicate the row_splits across the batch
        splits = neighbors["neighbors_row_splits"]
        if batched:
            splits = splits.repeat([batch_size] + [1] * splits.ndim)

        # Finally, segment_csr to aggregate results
        out_features = segment_csr(
            rep_features,
            splits,
            reduce=reduction,
            use_scatter=self.use_torch_scatter
        )


        return out_features
