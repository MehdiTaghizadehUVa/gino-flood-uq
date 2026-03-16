"""Utilities for strict reproducibility in training and evaluation."""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from hashlib import blake2b
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


_UINT32_MOD = 2**32 - 1


def _normalize_numpy_seed(seed: int) -> int:
    seed_int = int(seed) % _UINT32_MOD
    if seed_int == 0:
        seed_int = 1
    return seed_int


def stable_seed_from_parts(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def configure_global_determinism(deterministic: bool = True) -> None:
    deterministic = bool(deterministic)

    if deterministic:
        os.environ.setdefault("PYTHONHASHSEED", "0")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not deterministic

    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = not deterministic


def seed_all(seed: int, *, deterministic: bool = True) -> None:
    configure_global_determinism(deterministic)
    random.seed(int(seed))
    np.random.seed(_normalize_numpy_seed(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def collect_rng_state_across_ranks() -> dict[str, Any]:
    local_state = capture_rng_state()
    if not (dist.is_available() and dist.is_initialized()):
        return {
            "format_version": 1,
            "world_size": 1,
            "rank_states": [local_state],
        }

    world_size = dist.get_world_size()
    gathered: list[Any] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_state)
    return {
        "format_version": 1,
        "world_size": int(world_size),
        "rank_states": gathered,
    }


def _select_rank_state(state_bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    if state_bundle is None:
        return None
    if "rank_states" not in state_bundle:
        return state_bundle

    rank_states = state_bundle.get("rank_states") or []
    if not rank_states:
        return None

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank < len(rank_states) and rank_states[rank] is not None:
        return rank_states[rank]
    return rank_states[0]


def restore_rng_state(state_bundle: dict[str, Any] | None) -> bool:
    state = _select_rank_state(state_bundle)
    if state is None:
        return False

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_cpu_state = state.get("torch_cpu")
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state)

    torch_cuda_state = state.get("torch_cuda")
    if torch_cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(torch_cuda_state)

    return True


@contextmanager
def deterministic_seed_context(seed: int | None):
    if seed is None:
        yield
        return

    previous_state = capture_rng_state()
    seed_all(int(seed), deterministic=True)
    try:
        yield
    finally:
        restore_rng_state(previous_state)


def seed_sampler_for_epoch(sampler: Any, *, base_seed: int | None, epoch: int) -> bool:
    """Seed sampler-local RNG from an epoch-stable seed for reproducible resumes."""
    if sampler is None or base_seed is None:
        return False

    if not hasattr(sampler, "generator"):
        return False

    generator = getattr(sampler, "generator", None)
    if generator is None:
        generator = torch.Generator()
        sampler.generator = generator

    generator.manual_seed(stable_seed_from_parts("train_sampler", int(base_seed), int(epoch)))
    return True


def seed_dataloader_for_epoch(loader: Any, *, base_seed: int | None, epoch: int) -> bool:
    if loader is None or base_seed is None:
        return False

    sampler = getattr(loader, "sampler", None)
    if seed_sampler_for_epoch(sampler, base_seed=base_seed, epoch=epoch):
        return True

    batch_sampler = getattr(loader, "batch_sampler", None)
    nested_sampler = getattr(batch_sampler, "sampler", None)
    return seed_sampler_for_epoch(nested_sampler, base_seed=base_seed, epoch=epoch)
