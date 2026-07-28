#!/usr/bin/env python3
"""Compare anchored and trained ALR updates in effective-weight space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _real_inner(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.vdot(left.reshape(-1), right.reshape(-1)).real


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return float("nan")
    return float((_real_inner(left, right) / denominator).cpu())


def _effective_update(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    rank = int(a.shape[-1])
    return torch.einsum("mor,mri...->moi...", a, b) / float(rank)


def _pairwise_mean(values: torch.Tensor) -> float:
    distances = []
    for left in range(values.shape[0]):
        for right in range(left + 1, values.shape[0]):
            distances.append((values[left] - values[right]).norm())
    return float(torch.stack(distances).mean().cpu())


def _summarize_update(anchor: torch.Tensor, trained: torch.Tensor) -> dict:
    anchor_centered = anchor - anchor.mean(dim=0, keepdim=True)
    trained_centered = trained - trained.mean(dim=0, keepdim=True)
    anchor_spread = anchor_centered.square().abs().mean().sqrt()
    trained_spread = trained_centered.square().abs().mean().sqrt()
    change = trained - anchor
    return {
        "anchor_rms": float(anchor.square().abs().mean().sqrt().cpu()),
        "trained_rms": float(trained.square().abs().mean().sqrt().cpu()),
        "effective_displacement_rms": float(change.square().abs().mean().sqrt().cpu()),
        "anchor_centered_spread_rms": float(anchor_spread.cpu()),
        "trained_centered_spread_rms": float(trained_spread.cpu()),
        "centered_spread_retention": float(
            (trained_spread / anchor_spread.clamp_min(torch.finfo(anchor_spread.dtype).eps)).cpu()
        ),
        "anchor_pairwise_frobenius_mean": _pairwise_mean(anchor),
        "trained_pairwise_frobenius_mean": _pairwise_mean(trained),
        "anchor_trained_cosine_by_particle": [
            _cosine(anchor[index], trained[index]) for index in range(anchor.shape[0])
        ],
    }


def checkpoint_diagnostics(path: Path) -> dict:
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}")
    prefixes = sorted(
        key[: -len(".anchor_a")] for key in state if key.endswith(".anchor_a")
    )
    if not prefixes:
        raise ValueError(f"No ALR anchors found in {path}")

    layers = {}
    anchor_flat = []
    trained_flat = []
    factor_offset_squared = torch.zeros((), dtype=torch.float64)
    for prefix in prefixes:
        anchor_a = state[prefix + ".anchor_a"]
        anchor_b = state[prefix + ".anchor_b"]
        offset_a = state[prefix + ".offset_a"]
        offset_b = state[prefix + ".offset_b"]
        anchor = _effective_update(anchor_a, anchor_b)
        trained = _effective_update(anchor_a + offset_a, anchor_b + offset_b)
        layers[prefix] = _summarize_update(anchor, trained)
        anchor_flat.append(anchor.reshape(anchor.shape[0], -1).to(torch.complex128))
        trained_flat.append(trained.reshape(trained.shape[0], -1).to(torch.complex128))
        factor_offset_squared += offset_a.abs().square().sum().double()
        factor_offset_squared += offset_b.abs().square().sum().double()

    anchor_all = torch.cat(anchor_flat, dim=1)
    trained_all = torch.cat(trained_flat, dim=1)
    return {
        "checkpoint": str(path),
        "num_adapters": len(prefixes),
        "num_particles": int(anchor_all.shape[0]),
        "factor_offset_norm": float(factor_offset_squared.sqrt()),
        "effective_weight_aggregate": _summarize_update(anchor_all, trained_all),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat for each checkpoint to compare.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = {}
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("--checkpoint must use LABEL=PATH")
        label, raw_path = value.split("=", 1)
        reports[label] = checkpoint_diagnostics(Path(raw_path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
