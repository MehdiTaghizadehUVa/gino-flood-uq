"""Nested rollout helpers for anchored low-rank FGNO ensembles."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ALRMemberLayout:
    """Particle-major flattening contract for an ``M x K`` ensemble."""

    num_particles: int
    aleatory_samples: int

    def __post_init__(self) -> None:
        if int(self.num_particles) < 2:
            raise ValueError("ALR evaluation requires at least two particles.")
        if int(self.aleatory_samples) < 2:
            raise ValueError("ALR evaluation requires at least two aleatory samples.")

    @property
    def n_members(self) -> int:
        return int(self.num_particles) * int(self.aleatory_samples)

    @property
    def member_epistemic_id(self) -> torch.Tensor:
        return torch.arange(int(self.num_particles), dtype=torch.long).repeat_interleave(
            int(self.aleatory_samples)
        )

    @property
    def member_aleatory_id(self) -> torch.Tensor:
        return torch.arange(int(self.aleatory_samples), dtype=torch.long).repeat(
            int(self.num_particles)
        )

    def unflatten_members(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[0] != self.n_members:
            raise ValueError(
                f"Expected {self.n_members} flattened members, got {values.shape[0]}."
            )
        return values.reshape(
            int(self.num_particles), int(self.aleatory_samples), *values.shape[1:]
        )


def _expand_common_latents(
    latent_bank: torch.Tensor,
    *,
    layout: ALRMemberLayout,
) -> torch.Tensor:
    """Repeat one ordered ``K``-draw bank identically over all particles."""

    if latent_bank.ndim != 3:
        raise ValueError("latent_bank must have shape [K, B, latent_dim].")
    if latent_bank.shape[0] != layout.aleatory_samples or latent_bank.shape[1] != 1:
        raise ValueError(
            "Hydrograph ALR rollout expects latent_bank shape "
            f"[{layout.aleatory_samples}, 1, latent_dim], got {tuple(latent_bank.shape)}."
        )
    return latent_bank.unsqueeze(0).expand(
        layout.num_particles, layout.aleatory_samples, 1, latent_bank.shape[-1]
    ).reshape(layout.n_members, latent_bank.shape[-1])


def forward_alr_rollout_step(
    model,
    *,
    histories: torch.Tensor,
    static: torch.Tensor,
    boundary: torch.Tensor,
    input_geom: torch.Tensor,
    latent_queries: torch.Tensor,
    output_queries: torch.Tensor,
    latent_bank: torch.Tensor,
) -> torch.Tensor:
    """Advance all ALR particles and aleatory members in one model call.

    Parameters
    ----------
    histories
        Normalized member histories with shape ``[M, K, H, N, C]``.
    static
        Static cell features with shape ``[1, N, C_static]``.
    boundary
        Shared boundary history with shape ``[H, N, C_boundary]``.

    Returns
    -------
    torch.Tensor
        Nested one-step predictions with shape ``[M, K, 1, N, C_target]``.
    """

    if histories.ndim != 5:
        raise ValueError("histories must have shape [M, K, H, N, C].")
    particles, aleatory, history_steps, n_cells, _ = histories.shape
    layout = ALRMemberLayout(particles, aleatory)
    if static.ndim != 3 or static.shape[:2] != (1, n_cells):
        raise ValueError("static must have shape [1, N, C_static].")
    if boundary.ndim != 3 or boundary.shape[:2] != (history_steps, n_cells):
        raise ValueError("boundary must have shape [H, N, C_boundary].")

    dynamic_flat = histories.permute(0, 1, 3, 2, 4).reshape(
        layout.n_members, n_cells, -1
    )
    boundary_flat = boundary.permute(1, 0, 2).reshape(1, n_cells, -1).expand(
        layout.n_members, n_cells, -1
    )
    static_flat = static.expand(layout.n_members, n_cells, static.shape[-1])
    x = torch.cat([static_flat, boundary_flat, dynamic_flat], dim=-1)
    latents = _expand_common_latents(latent_bank, layout=layout)
    particle_ids = layout.member_epistemic_id.to(device=x.device)
    prediction = model(
        input_geom=input_geom,
        latent_queries=latent_queries,
        output_queries=output_queries,
        x=x,
        ada_in=latents,
        particle_ids=particle_ids,
    )
    if prediction.shape[0] != layout.n_members:
        raise ValueError(
            "ALR model output member count does not match the requested nested ensemble."
        )
    return prediction.reshape(particles, aleatory, 1, *prediction.shape[1:])


def alr_nested_variance_components(prediction: torch.Tensor) -> dict[str, torch.Tensor]:
    """Decompose a crossed ``[M,K,...]`` ensemble using sample variances.

    The same aleatory bank must be ordered identically for every particle. The
    epistemic component is the sample variance over particle-wise aleatory
    means; the aleatory component is the mean within-particle sample variance.
    """

    if prediction.ndim < 3:
        raise ValueError("prediction must have shape [M, K, ...].")
    if prediction.shape[0] < 2 or prediction.shape[1] < 2:
        raise ValueError("variance decomposition requires M >= 2 and K >= 2.")
    aleatory = prediction.var(dim=1, unbiased=True).mean(dim=0)
    epistemic = prediction.mean(dim=1).var(dim=0, unbiased=True)
    return {
        "variance_aleatory": aleatory,
        "variance_epistemic": epistemic,
        "variance_total": aleatory + epistemic,
    }
