import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.GELU(),
                nn.Linear(512, 2*in_channels)
            )
        self.mlp = mlp

        self.embedding = None

    def _canonicalize_embedding(self, embedding):
        """Normalize embedding shape to [B, embed_dim] while accepting [embed_dim] too."""
        if embedding.ndim == 1:
            if embedding.numel() != self.embed_dim:
                raise ValueError(
                    f"AdaIN expected embedding dim {self.embed_dim}, got {embedding.numel()}."
                )
            return embedding.reshape(1, self.embed_dim)
        if embedding.ndim == 2:
            if embedding.shape[1] != self.embed_dim:
                raise ValueError(
                    f"AdaIN expected embedding shape [B, {self.embed_dim}], got {tuple(embedding.shape)}."
                )
            return embedding
        raise ValueError(
            f"AdaIN expected embedding with rank 1 or 2, got rank {embedding.ndim}."
        )

    def set_embedding(self, x):
        if x.ndim == 1:
            self.embedding = self._canonicalize_embedding(x).squeeze(0)
        elif x.ndim == 2:
            self.embedding = self._canonicalize_embedding(x)
        else:
            raise ValueError(
                f"AdaIN expected embedding with rank 1 or 2, got rank {x.ndim}."
            )

    def forward(self, x, embedding=None):
        """Forward with optional embedding.

        Accepted embedding shapes:
          - [embed_dim]
          - [1, embed_dim]
          - [B, embed_dim]
        """
        if embedding is not None:
            e = embedding
        else:
            e = self.embedding
            if e is None:
                raise RuntimeError("AdaIN: pass embedding or call set_embedding before forward")

        e = self._canonicalize_embedding(e)
        batch_size = x.shape[0]
        if e.shape[0] == 1 and batch_size > 1:
            e = e.expand(batch_size, -1)
        elif e.shape[0] != batch_size:
            raise ValueError(
                f"AdaIN embedding batch ({e.shape[0]}) must be 1 or match x batch ({batch_size})."
            )

        # Normalize per sample/channel group first, then apply sample-wise affine parameters.
        x_norm = nn.functional.group_norm(
            x, self.in_channels, weight=None, bias=None, eps=self.eps
        )
        affine = self.mlp(e)
        weight, bias = torch.split(affine, self.in_channels, dim=-1)
        view_shape = (batch_size, self.in_channels) + (1,) * (x.ndim - 2)
        weight = weight.reshape(view_shape)
        bias = bias.reshape(view_shape)
        return x_norm * weight + bias

class InstanceNorm(nn.Module):
    def __init__(self, **kwargs):
        """InstanceNorm applies dim-agnostic instance normalization
        to data as an nn.Module. 

        kwargs: additional parameters to pass to instance_norm() for use as a module
        e.g. eps, affine
        """
        super().__init__()
        self.kwargs = kwargs
    
    def forward(self, x):
        size = x.shape
        x = torch.nn.functional.instance_norm(x, **self.kwargs)
        assert x.shape == size
        return x
