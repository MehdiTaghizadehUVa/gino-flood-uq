"""
Snippet to load all artifacts of training state as Modules
without constraining to use inside a default Trainer
"""
from typing import Union
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
from neuralop.mpu.comm import get_local_rank
from .determinism import collect_rng_state_across_ranks, restore_rng_state


def _normalize_map_location(map_location):
    """Make torch.load map_location robust to torch.device values."""
    if isinstance(map_location, torch.device):
        return str(map_location)
    if isinstance(map_location, dict):
        normalized = {}
        for src, dst in map_location.items():
            norm_src = str(src) if isinstance(src, torch.device) else src
            norm_dst = str(dst) if isinstance(dst, torch.device) else dst
            normalized[norm_src] = norm_dst
        return normalized
    return map_location


def load_training_state(save_dir: Union[str, Path], 
                        save_name: str,
                        model: nn.Module,
                        optimizer: nn.Module=None,
                        scheduler: nn.Module=None,
                        regularizer: nn.Module=None,
                        map_location: dict=None,
                        restore_rng_state_on_load: bool=False) -> dict:
    
    """load_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    model : nn.Module
        model to save
    optimizer : nn.Module, optional
        optimizer object to save, by default None
    scheduler : nn.Module, optional
        scheduler object to save, by default None
    regularizer : nn.Module, optional
        regularizer object to save, by default None
    map_location : dict, optional
        mapping dictionary keyed `{device_from: device_to}`, by default None
        dictionary instructs torch to load a model from a checkpoint on rank `device_from`
        and send it to `device_to`

    Returns
    -------
    tuple of training state
        ``model, optimizer, scheduler, regularizer, epoch``
        
    """
    if not map_location:
        if dist.is_initialized() and torch.cuda.is_available():
            map_location = {"cuda:0": f"cuda:{get_local_rank()}"}
        else:
            map_location = "cpu"
    map_location = _normalize_map_location(map_location)

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    # optionally load epoch 
    epoch = None
    manifest = None
    manifest_pth = save_dir / "manifest.pt"
    if manifest_pth.exists():
        manifest = torch.load(manifest_pth, weights_only=False)
        epoch = manifest.get('epoch')
    
    save_filename = f"{save_name}_state_dict.pt"
    if save_name == "model" and manifest is not None:
        manifest_model = manifest.get("model")
        if manifest_model:
            manifest_model_pth = save_dir / str(manifest_model)
            if manifest_model_pth.exists():
                save_filename = str(manifest_model)

    save_pth = save_dir / save_filename
    if dist.is_initialized() and torch.cuda.is_available():
        # To minimize CUDA memory overhead during checkpoint loading,
        # load the model to CPU first, then load to GPU instead of mapping from
        # CUDA:0 to CUDA:DEVICE_ID
        device_id = get_local_rank()
        model.load_state_dict(torch.load(save_pth.absolute().as_posix(), map_location="cpu"))
        model = model.to(device=f"cuda:{device_id}")
        torch.cuda.empty_cache()
    else:
        model.load_state_dict(torch.load(save_pth.absolute().as_posix(), map_location=map_location))

    # load optimizer if state exists
    if optimizer is not None:
        optimizer_pth = save_dir / "optimizer.pt"
        if optimizer_pth.exists():
            optimizer.load_state_dict(torch.load(optimizer_pth.absolute().as_posix(), map_location=map_location))
            target_device = None
            if dist.is_initialized():
                target_device = f"cuda:{get_local_rank()}" if torch.cuda.is_available() else "cpu"
            elif torch.cuda.is_available():
                target_device = "cuda:0"
            else:
                target_device = "cpu"
            for s in optimizer.state.values():
                for k, v in s.items():
                    if torch.is_tensor(v):
                        s[k] = v.to(target_device)

        else:
            print(f"Warning: requested to load optimizer state, but no saved optimizer state exists in {save_dir}.")
    
    if scheduler is not None:
        scheduler_pth = save_dir / "scheduler.pt"
        if scheduler_pth.exists():
            scheduler.load_state_dict(torch.load(scheduler_pth.absolute().as_posix(), map_location=map_location))
        else:
            print(f"Warning: requested to load scheduler state, but no saved scheduler state exists in {save_dir}.")
    
    if regularizer is not None:
        regularizer_pth = save_dir / "regularizer.pt"
        if regularizer_pth.exists():
            regularizer.load_state_dict(torch.load(regularizer_pth.absolute().as_posix(), map_location=map_location))
        else:
            print(f"Warning: requested to load regularizer state, but no saved regularizer state exists in {save_dir}.")

    if restore_rng_state_on_load:
        rng_state_pth = save_dir / "rng_state.pt"
        if rng_state_pth.exists():
            rng_state = torch.load(rng_state_pth.absolute().as_posix(), map_location="cpu", weights_only=False)
            restore_rng_state(rng_state)
        else:
            print(
                f"Warning: requested to restore RNG state, but no saved rng_state.pt exists in {save_dir}."
            )
    
    return model, optimizer, scheduler, regularizer, epoch


def save_training_state(save_dir: Union[str, Path], save_name: str,
                        model: nn.Module,
                        optimizer: nn.Module=None,
                        scheduler: nn.Module=None,
                        regularizer: nn.Module=None,
                        epoch: int=None) -> None:
    """save_training_state returns model and optional other training modules
    saved from prior training for downstream use

    Parameters
    ----------
    save_dir : Union[str, Path]
        directory from which to load training state (model, optional optimizer, scheduler, regularizer)
    save_name : str
        name of model to load
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
    rng_state_bundle = collect_rng_state_across_ranks()

    manifest = {}

    if is_rank0:
        # Just save the model.module if model is in DDP mode
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            save_dir.mkdir(exist_ok=True, parents=True)
            model_pth = save_dir / f"{save_name}_state_dict.pt"
            torch.save(model.module.state_dict(), model_pth.as_posix())
        else:
            # otherwise save the model checkpoint
            model.save_checkpoint(save_dir, save_name)
        manifest['model'] = f"{save_name}_state_dict.pt"

        # save optimizer if state exists
        if optimizer is not None:
            optimizer_pth = save_dir / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_pth)
            manifest['optimizer'] = "optimizer.pt"
        
        if scheduler is not None:
            scheduler_pth = save_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_pth)
            manifest['scheduler'] = "scheduler.pt"
        
        if regularizer is not None:
            regularizer_pth = save_dir / "regularizer.pt"
            torch.save(regularizer.state_dict(), regularizer_pth)
            manifest['regularizer'] = "regularizer.pt"

        if epoch is not None:
            manifest["epoch"] = epoch
        manifest["rng_state"] = "rng_state.pt"

        torch.save(rng_state_bundle, save_dir / "rng_state.pt")
        torch.save(manifest, save_dir / "manifest.pt")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    
