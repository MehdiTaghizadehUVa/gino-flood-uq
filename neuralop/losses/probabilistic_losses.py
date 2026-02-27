"""
Probabilistic losses for ensemble / distributional predictions.

Implements the fair Continuous Ranked Probability Score (fCRPS) for training
models that output predictive distributions via sampling (e.g. FGN-style
noise-conditioned forwards).

**Univariate fCRPS (Eq. 4, FGN paper arXiv:2506.10772)**

For ensemble :math:`x^{1:N} = (x_1, \\ldots, x_N)` and target :math:`y`:

.. math::

   \\mathrm{fCRPS}(x^{1:N}, y)
   = \\frac{1}{N} \\sum_{n=1}^{N} |x_n - y|
   - \\frac{1}{2N(N-1)} \\sum_{n \\neq n'} |x_n - x_{n'}|

**Aggregate loss (Eq. 5)** — mean over batch, locations, and channels (optionally
weighted by :math:`a_i` per channel and/or spatial weights):

.. math::

   L = \\frac{1}{|D|} \\sum_{d} \\frac{1}{G} \\sum_{i} a_i \\,
   \\mathrm{fCRPS}(x^{1:N}_{i,d}, y_{i,d})
"""

from typing import Optional, Union

import torch


def fair_crps_univariate(samples: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Fair CRPS estimator for univariate predictions (one location, one variable).

    **Equation (FGN paper arXiv:2506.10772 Eq. 4)**

    .. math::

       \\mathrm{fCRPS}(x^{1:N}, y)
       = \\underbrace{\\frac{1}{N} \\sum_{n=1}^{N} |x_n - y|}_{\\mathrm{term}_1}
       - \\underbrace{\\frac{1}{2N(N-1)} \\sum_{n \\neq n'} |x_n - x_{n'}|}_{\\mathrm{term}_2}

    - **term1**: mean absolute error of the ensemble to the target (sharpness).
    - **term2**: mean pairwise spread of the ensemble (rewards useful uncertainty;
      zero if all :math:`x_n` are identical).

    Parameters
    ----------
    samples : torch.Tensor, shape (N, ...)
        N ensemble members (e.g. from N forward passes with different noise z).
    target : torch.Tensor, shape (...)
        Ground truth, same trailing shape as samples.

    Returns
    -------
    torch.Tensor
        fCRPS per (location, channel), same trailing shape as ``target``.

    Notes
    -----
    If pairwise differences (samples[i] - samples[j]) are all zeros, the N rows
    of ``samples`` are identical. Then term2 = 0 and fCRPS = term1 only. That
    usually means the caller passed the same prediction twice, or the model
    output does not depend on the noise (e.g. noise not passed through forward).

    References
    ----------
    Zamo & Naveau (2018), "Estimation of the continuous ranked probability score
    with limited information and applications to ensemble weather forecasts."
    """
    N = samples.shape[0]
    if N < 2:
        raise ValueError("fair_crps requires at least 2 samples (N>=2).")
    term1 = (samples - target).abs().mean(dim=0)
    # Pairwise differences: diff[i,j,...] = samples[i,...] - samples[j,...]
    # (1,N,...) - (N,1,...) broadcasts to (N,N,...)
    diff = samples.unsqueeze(0) - samples.unsqueeze(1)  # (N, N, ...)
    sum_pairs = diff.abs().sum(dim=(0, 1))  # each pair (n,n') and (n',n) counted
    term2 = sum_pairs / (2 * N * (N - 1))
    return term1 - term2


class CRPSLoss:
    """
    Continuous Ranked Probability Score (fair estimator) for marginal forecast distributions.

    **Per-location fCRPS (Eq. 4)**

    For each (batch index :math:`d`, location, channel) :math:`i`, with ensemble
    :math:`x^{1:N}_{i,d}` and target :math:`y_{i,d}`:

    .. math::

       \\mathrm{fCRPS}(x^{1:N}_{i,d}, y_{i,d})
       = \\frac{1}{N} \\sum_{n=1}^{N} |x_{n,i,d} - y_{i,d}|
       - \\frac{1}{2N(N-1)} \\sum_{n \\neq n'} |x_{n,i,d} - x_{n',i,d}|

    **Aggregate loss (Eq. 5)**

    .. math::

       L = \\frac{1}{|D|} \\sum_{d \\in D} \\frac{1}{G} \\sum_{i} a_i \\,
       \\mathrm{fCRPS}(x^{1:N}_{i,d}, y_{i,d})

    where :math:`d` indexes the batch, :math:`i` indexes (location, channel),
    :math:`G` is the number of such tuples, and :math:`a_i` are channel weights.
    With ``reduction='mean'`` and no spatial weights: :math:`L` is the mean over
    batch, locations, and channels (optionally weighted by :math:`a_c` per channel).
    With ``spatial_weights``: :math:`L = \\sum w_{i,d} \\mathrm{fCRPS}_{i,d} / (\\sum w_{i,d} + \\epsilon)`.

    Expects model to be evaluated N times (e.g. N=2) with different noise samples,
    yielding N predictions. fCRPS (Eq. 4) is computed per (batch, location, channel),
    then reduced. Use reduction='mean' to match the paper.

    Parameters
    ----------
    n_samples : int
        Number of predictive samples per batch (N). Must be >= 2. Paper uses N=2.
    channel_weights : optional tensor or list
        Weights :math:`a_i` per output channel (variable). Shape (n_channels,) or list.
        Applied as in Eq. 5: weighted sum then divide by G. If None, uniform (a_i=1).
    reduction : str
        'mean' (paper Eq. 5: average over batch, locations, channels) or 'sum'.

    Call signature
    -------------
    loss(pred_samples, y, spatial_weights=None, **kwargs) where:
      pred_samples : torch.Tensor
          Shape (N, B, n_points, n_channels) — N samples from the predictive distribution.
      y : torch.Tensor
          Shape (B, n_points, n_channels) — ground truth.
      spatial_weights : torch.Tensor, optional
          Shape (B, n_points, n_channels). When provided, loss is a weighted mean over
          (batch, location, channel): sum(weights * crps) / (sum(weights) + eps). Use for
          area-weighted and wet/dry-aware flood objectives (e.g. area, wet-mask for velocity).
    """

    def __init__(
        self,
        n_samples: int = 2,
        channel_weights: Optional[Union[torch.Tensor, list]] = None,
        reduction: str = "mean",
    ):
        if n_samples < 2:
            raise ValueError("CRPSLoss requires n_samples >= 2 for the fair estimator.")
        self.n_samples = n_samples
        self.reduction = reduction
        if channel_weights is not None:
            if isinstance(channel_weights, (list, tuple)):
                channel_weights = torch.tensor(channel_weights, dtype=torch.float32)
            self.channel_weights = channel_weights
        else:
            self.channel_weights = None

    @property
    def name(self) -> str:
        return "CRPSLoss"

    def __call__(
        self,
        pred_samples: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute mean (or sum) fair CRPS over batch, locations, and channels.

        For each (batch, location, channel) computes
        :math:`\\mathrm{fCRPS}(x^{1:N}, y)` then:

        - ``reduction='mean'``: :math:`L = (1 / (B \\cdot P \\cdot C)) \\sum_{d,p,c} a_c \\, \\mathrm{fCRPS}_{d,p,c}`
        - ``reduction='sum'``: :math:`L = \\sum_{d,p,c} a_c \\, \\mathrm{fCRPS}_{d,p,c}`

        If ``spatial_weights`` is provided, a weighted mean is used instead.

        Parameters
        ----------
        pred_samples : torch.Tensor
            Shape (N, B, n_points, n_channels).
        y : torch.Tensor
            Shape (B, n_points, n_channels).

        Returns
        -------
        loss : torch.Tensor
            Scalar.
        """
        N, B, n_points, n_channels = pred_samples.shape
        if N != self.n_samples:
            raise ValueError(
                f"CRPSLoss expects pred_samples with first dim {self.n_samples}, got {N}."
            )
        if y.shape != (B, n_points, n_channels):
            raise ValueError(
                f"y shape must be (B, n_points, n_channels) = ({B}, {n_points}, {n_channels}), got {y.shape}."
            )
        # Flatten batch and space: (N, B * n_points, n_channels)
        pred_flat = pred_samples.permute(0, 2, 3, 1).reshape(N, -1, n_channels)
        y_flat = y.permute(2, 1, 0).reshape(-1, n_channels)

        # (B*n_points, n_channels): fCRPS per (batch, location, channel) — Eq. 4
        crps_per_loc = fair_crps_univariate(pred_flat, y_flat)

        # Eq. 5 loss weighting: a_i fCRPS for each i (channel = variable)
        if self.channel_weights is not None:
            w = self.channel_weights.to(crps_per_loc.device)
            if w.shape[0] != n_channels:
                raise ValueError(
                    f"channel_weights length {w.shape[0]} != n_channels {n_channels}."
                )
            crps_per_loc = crps_per_loc * w.unsqueeze(0)

        # Optional spatial (and channel) weights: (B, n_points, n_channels) -> weighted mean
        spatial_weights = kwargs.get("spatial_weights")
        if spatial_weights is not None:
            if spatial_weights.shape != (B, n_points, n_channels):
                raise ValueError(
                    f"spatial_weights shape must be (B, n_points, n_channels) = ({B}, {n_points}, {n_channels}), got {spatial_weights.shape}."
                )
            # Flatten to (B*n_points, n_channels) to match crps_per_loc layout
            weights_flat = spatial_weights.permute(2, 1, 0).reshape(-1, n_channels)
            weights_flat = weights_flat.to(crps_per_loc.device)
            crps_weighted = crps_per_loc * weights_flat
            w_sum = weights_flat.sum()
            eps = 1e-10
            if self.reduction == "mean":
                return crps_weighted.sum() / (w_sum + eps)
            return crps_weighted.sum()

        if self.reduction == "mean":
            return crps_per_loc.mean()  # (1/(B*P*C)) sum_d sum_{p,c} a_c fCRPS
        return crps_per_loc.sum()
