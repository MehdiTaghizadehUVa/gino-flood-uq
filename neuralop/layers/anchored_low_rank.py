"""Anchored low-rank residual adapters for dense and spectral operators."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _validate_particle_ids(
    particle_ids: torch.Tensor | None,
    *,
    batch_size: int,
    num_particles: int,
    device: torch.device,
) -> torch.Tensor:
    if particle_ids is None:
        raise ValueError("particle_ids are required when anchored low-rank adapters are enabled.")
    ids = torch.as_tensor(particle_ids, dtype=torch.long, device=device).reshape(-1)
    if ids.numel() != int(batch_size):
        raise ValueError(
            f"particle_ids must have one entry per batch item ({batch_size}), got {ids.numel()}."
        )
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= int(num_particles)):
        raise ValueError(
            f"particle_ids must be in [0, {num_particles - 1}], got "
            f"min={int(ids.min())}, max={int(ids.max())}."
        )
    return ids


def _complex_randn(shape: Sequence[int], *, generator: torch.Generator) -> torch.Tensor:
    real = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    imag = torch.randn(tuple(shape), generator=generator, dtype=torch.float32)
    return torch.complex(real, imag) / (2.0**0.5)


class AnchoredLowRankDenseAdapter(nn.Module):
    """Particle-indexed low-rank update for linear or pointwise channel maps.

    The fixed anchors are buffers. Only offsets from those anchors are trainable.
    At initialization, each particle's effective update has the requested norm
    relative to the shared reference weight.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_particles: int,
        rank: int,
        reference_weight: torch.Tensor,
        anchor_relative_norm: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_particles = int(num_particles)
        self.rank = int(rank)
        if min(self.in_features, self.out_features, self.num_particles, self.rank) <= 0:
            raise ValueError("Adapter dimensions, particle count, and rank must be positive.")
        if float(anchor_relative_norm) < 0:
            raise ValueError("anchor_relative_norm must be nonnegative.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        anchor_a = torch.randn(
            self.num_particles,
            self.out_features,
            self.rank,
            generator=generator,
            dtype=torch.float32,
        )
        anchor_b = torch.randn(
            self.num_particles,
            self.rank,
            self.in_features,
            generator=generator,
            dtype=torch.float32,
        )
        anchor_a, anchor_b = self._normalize_anchors(
            anchor_a,
            anchor_b,
            reference_weight=reference_weight,
            relative_norm=float(anchor_relative_norm),
        )
        self.register_buffer("anchor_a", anchor_a)
        self.register_buffer("anchor_b", anchor_b)
        self.offset_a = nn.Parameter(torch.zeros_like(anchor_a))
        self.offset_b = nn.Parameter(torch.zeros_like(anchor_b))

    def _normalize_anchors(self, a, b, *, reference_weight, relative_norm):
        delta = torch.einsum("mor,mri->moi", a, b) / float(self.rank)
        current = torch.linalg.vector_norm(delta.reshape(self.num_particles, -1), dim=1)
        target = float(relative_norm) * float(torch.linalg.vector_norm(reference_weight.detach()).cpu())
        if target == 0.0:
            return torch.zeros_like(a), torch.zeros_like(b)
        scale = target / current.clamp_min(torch.finfo(current.dtype).eps)
        return a * scale[:, None, None], b

    def explicit_delta_weight(self, particle_ids: torch.Tensor) -> torch.Tensor:
        ids = _validate_particle_ids(
            particle_ids,
            batch_size=torch.as_tensor(particle_ids).numel(),
            num_particles=self.num_particles,
            device=self.offset_a.device,
        )
        a = self.anchor_a.index_select(0, ids) + self.offset_a.index_select(0, ids)
        b = self.anchor_b.index_select(0, ids) + self.offset_b.index_select(0, ids)
        return torch.einsum("bor,bri->boi", a, b) / float(self.rank)

    def forward(
        self,
        x: torch.Tensor,
        particle_ids: torch.Tensor | None,
        *,
        channels_last: bool,
    ) -> torch.Tensor:
        ids = _validate_particle_ids(
            particle_ids,
            batch_size=x.shape[0],
            num_particles=self.num_particles,
            device=x.device,
        )
        delta = self.explicit_delta_weight(ids).to(dtype=x.dtype)
        if channels_last:
            if x.shape[-1] != self.in_features:
                raise ValueError(
                    f"Expected {self.in_features} input features on the last axis, got {x.shape[-1]}."
                )
            return torch.einsum("b...i,boi->b...o", x, delta)
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features on axis 1, got {x.shape[1]}."
            )
        return torch.einsum("bi...,boi->bo...", x, delta)

    def offset_penalty(self) -> torch.Tensor:
        return self.offset_a.square().sum() + self.offset_b.square().sum()


class AnchoredLowRankSpectralAdapter(nn.Module):
    """Particle-indexed complex low-rank update for a dense spectral kernel."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: Sequence[int],
        num_particles: int,
        rank: int,
        reference_weight: torch.Tensor,
        anchor_relative_norm: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.n_modes = tuple(int(v) for v in n_modes)
        self.num_particles = int(num_particles)
        self.rank = int(rank)
        if min(self.in_channels, self.out_channels, self.num_particles, self.rank, *self.n_modes) <= 0:
            raise ValueError("Spectral adapter dimensions, modes, particle count, and rank must be positive.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        anchor_a = _complex_randn(
            (self.num_particles, self.out_channels, self.rank), generator=generator
        )
        anchor_b = _complex_randn(
            (self.num_particles, self.rank, self.in_channels, *self.n_modes),
            generator=generator,
        )
        delta = torch.einsum("mor,mri...->mio...", anchor_a, anchor_b) / float(self.rank)
        current = torch.linalg.vector_norm(delta.reshape(self.num_particles, -1), dim=1)
        target = float(anchor_relative_norm) * float(
            torch.linalg.vector_norm(reference_weight.detach()).cpu()
        )
        if target == 0.0:
            anchor_a.zero_()
            anchor_b.zero_()
        else:
            scale = target / current.clamp_min(torch.finfo(current.dtype).eps)
            anchor_a.mul_(scale[:, None, None])

        self.register_buffer("anchor_a", anchor_a)
        self.register_buffer("anchor_b", anchor_b)
        self.offset_a = nn.Parameter(torch.zeros_like(anchor_a))
        self.offset_b = nn.Parameter(torch.zeros_like(anchor_b))

    def explicit_delta_weight(
        self,
        particle_ids: torch.Tensor,
        *,
        mode_shape: Sequence[int] | None = None,
        mode_slices: Sequence[slice] | None = None,
    ) -> torch.Tensor:
        ids = _validate_particle_ids(
            particle_ids,
            batch_size=torch.as_tensor(particle_ids).numel(),
            num_particles=self.num_particles,
            device=self.offset_a.device,
        )
        a = self.anchor_a.index_select(0, ids) + self.offset_a.index_select(0, ids)
        b = self.anchor_b.index_select(0, ids) + self.offset_b.index_select(0, ids)
        if mode_slices is not None:
            if len(mode_slices) != len(self.n_modes):
                raise ValueError(
                    f"Expected {len(self.n_modes)} mode slices, got {len(mode_slices)}."
                )
            b = b[(slice(None), slice(None), slice(None), *tuple(mode_slices))]
        if mode_shape is not None:
            mode_shape = tuple(int(v) for v in mode_shape)
            if len(mode_shape) != len(self.n_modes):
                raise ValueError(
                    f"Expected {len(self.n_modes)} spectral dimensions, got {len(mode_shape)}."
                )
            if any(got > have for got, have in zip(mode_shape, self.n_modes)):
                raise ValueError(
                    f"Requested mode shape {mode_shape} exceeds configured modes {self.n_modes}."
                )
            slices = (slice(None), slice(None), slice(None)) + tuple(
                slice(0, size) for size in mode_shape
            )
            b = b[slices]
        return torch.einsum("bor,bri...->bio...", a, b) / float(self.rank)

    def forward(
        self,
        x: torch.Tensor,
        particle_ids: torch.Tensor | None,
        *,
        mode_slices: Sequence[slice] | None = None,
    ) -> torch.Tensor:
        ids = _validate_particle_ids(
            particle_ids,
            batch_size=x.shape[0],
            num_particles=self.num_particles,
            device=x.device,
        )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} spectral channels, got {x.shape[1]}."
            )
        delta = self.explicit_delta_weight(
            ids,
            mode_shape=None if mode_slices is not None else x.shape[2:],
            mode_slices=mode_slices,
        ).to(dtype=x.dtype)
        if tuple(delta.shape[3:]) != tuple(x.shape[2:]):
            raise ValueError(
                f"Adapter modes {tuple(delta.shape[3:])} do not match input modes {tuple(x.shape[2:])}."
            )
        return torch.einsum("bi...,bio...->bo...", x, delta)

    def offset_penalty(self) -> torch.Tensor:
        return self.offset_a.abs().square().sum() + self.offset_b.abs().square().sum()
