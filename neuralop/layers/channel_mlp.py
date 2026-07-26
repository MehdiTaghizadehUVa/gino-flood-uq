import torch
from torch import nn
import torch.nn.functional as F

from .anchored_low_rank import AnchoredLowRankDenseAdapter


class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
    out_channels : int, default is None
        if None, same is in_channels
    hidden_channels : int, default is None
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : default is F.gelu
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        hidden_channels=None,
        n_layers=2,
        n_dim=2,
        non_linearity=F.gelu,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

        self.anchored_low_rank = nn.ModuleDict()

    def enable_anchored_low_rank(
        self,
        *,
        layer_indices,
        num_particles,
        rank,
        anchor_relative_norm,
        seed,
    ):
        """Attach particle-indexed adapters without replacing shared weights."""
        indices = sorted({int(i) for i in layer_indices})
        if any(i < 0 or i >= self.n_layers for i in indices):
            raise ValueError(
                f"ChannelMLP adapter indices must be in [0, {self.n_layers - 1}], got {indices}."
            )
        for i in indices:
            fc = self.fcs[i]
            self.anchored_low_rank[str(i)] = AnchoredLowRankDenseAdapter(
                in_features=fc.in_channels,
                out_features=fc.out_channels,
                num_particles=num_particles,
                rank=rank,
                reference_weight=fc.weight.detach().squeeze(-1),
                anchor_relative_norm=anchor_relative_norm,
                seed=int(seed) + 1009 * i,
            )

    def forward(self, x, particle_ids=None):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True

        for i, fc in enumerate(self.fcs):
            layer_input = x
            x = fc(layer_input)
            adapter = self.anchored_low_rank[str(i)] if str(i) in self.anchored_low_rank else None
            if adapter is not None:
                x = x + adapter(layer_input, particle_ids, channels_last=False)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


# Reimplementation of the ChannelMLP class using Linear instead of Conv
class LinearChannelMLP(torch.nn.Module):
    def __init__(self, layers, non_linearity=F.gelu, dropout=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

        self.anchored_low_rank = nn.ModuleDict()

    def enable_anchored_low_rank(
        self,
        *,
        layer_indices,
        num_particles,
        rank,
        anchor_relative_norm,
        seed,
    ):
        indices = sorted({int(i) for i in layer_indices})
        if any(i < 0 or i >= self.n_layers for i in indices):
            raise ValueError(
                f"LinearChannelMLP adapter indices must be in [0, {self.n_layers - 1}], got {indices}."
            )
        for i in indices:
            fc = self.fcs[i]
            self.anchored_low_rank[str(i)] = AnchoredLowRankDenseAdapter(
                in_features=fc.in_features,
                out_features=fc.out_features,
                num_particles=num_particles,
                rank=rank,
                reference_weight=fc.weight.detach(),
                anchor_relative_norm=anchor_relative_norm,
                seed=int(seed) + 1009 * i,
            )

    def forward(self, x, particle_ids=None):
        for i, fc in enumerate(self.fcs):
            layer_input = x
            x = fc(layer_input)
            adapter = self.anchored_low_rank[str(i)] if str(i) in self.anchored_low_rank else None
            if adapter is not None:
                x = x + adapter(layer_input, particle_ids, channels_last=True)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x
