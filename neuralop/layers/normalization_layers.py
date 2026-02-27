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
    
    def set_embedding(self, x):
        self.embedding = x.reshape(self.embed_dim,)

    def forward(self, x, embedding=None):
        """Forward with optional embedding. If embedding is passed, it is used (and kept in the
        computation graph). Otherwise use self.embedding (set via set_embedding)."""
        if embedding is not None:
            e = embedding.reshape(self.embed_dim,)
        else:
            e = self.embedding
            if e is None:
                raise RuntimeError("AdaIN: pass embedding or call set_embedding before forward")
            e = e.reshape(self.embed_dim,)
        weight, bias = torch.split(self.mlp(e), self.in_channels, dim=0)
        out = nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)
        return out

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