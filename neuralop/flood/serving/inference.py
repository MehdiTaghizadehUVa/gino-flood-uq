"""FGN inference seam for the web worker.

The production adapter owns CUDA/model lifetime and exposes a small serving
interface. It reuses lower-level model/checkpoint/normalizer utilities, but it
never imports FastAPI/Celery/Postgres or calls the batch eval application.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Protocol, Sequence

import numpy as np

from neuralop.flood.serving.forcing import ForcingInput
from neuralop.flood.serving.model_bundle import FGNModelBundle
from neuralop.flood.serving.products import ForecastResult
from neuralop.flood.serving.run_spec import RunSpec

_LOG = logging.getLogger(__name__)


class FGNInferenceService(Protocol):
    def run(self, run_spec: RunSpec, forcing_input: ForcingInput) -> ForecastResult: ...


@dataclass(frozen=True)
class DomainAssets:
    """Fixed-domain tensors used by the coastal serving rollout.

    geometry/static are raw physical-space tensors. They are normalized once by
    ProductionFGNInferenceService with the train-fit normalizers from the bundle.
    """

    geometry: object
    static: object
    structural_dry_mask: object | None = None
    query_res: tuple[int, int] = (48, 48)


class FakeFGNInferenceService:
    """Deterministic adapter for API/orchestrator tests."""

    def __init__(self, bundle: FGNModelBundle, *, n_cells: int = 8) -> None:
        self.bundle = bundle
        self.n_cells = int(n_cells)

    def run(self, run_spec: RunSpec, forcing_input: ForcingInput) -> ForecastResult:
        rng = np.random.default_rng(int(run_spec.seed))
        n_members = int(self.bundle.total_members)
        n_time = int(run_spec.forecast_steps)
        n_cells = self.n_cells
        lead_hours = (np.arange(1, n_time + 1, dtype=np.float32) * self.bundle.dt_seconds) / 3600.0
        forcing = forcing_input.as_boundary_matrix()
        start = self.bundle.skip_before_timestep + self.bundle.n_history
        stage_signal = forcing[start:start + n_time, 0]
        precip_signal = forcing[start:start + n_time, 1]
        base = np.maximum(stage_signal[:, None] * 0.05 + precip_signal[:, None] * 0.001, 0.0)
        spatial = np.linspace(0.2, 1.0, n_cells, dtype=np.float32)[None, :]
        mean = base * spatial
        noise = rng.normal(0.0, 0.01, size=(n_members, n_time, n_cells)).astype(np.float32)
        members = np.clip(mean[None, :, :] + noise, 0.0, None)
        # Synthetic UTM-shaped geometry so map/animation products can render.
        side = max(2, int(np.ceil(np.sqrt(n_cells))))
        grid_x, grid_y = np.meshgrid(
            np.linspace(0.0, 1000.0, side, dtype=np.float32),
            np.linspace(0.0, 1000.0, side, dtype=np.float32),
            indexing="xy",
        )
        geometry_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)[:n_cells]
        return ForecastResult(
            members_wd=members.astype(np.float32),
            lead_time_hours=lead_hours,
            wettable_mask=np.ones(n_cells, dtype=bool),
            metadata={
                "adapter": "fake",
                "bundle_id": self.bundle.bundle_id,
                "geometry_xy": geometry_xy,
            },
        )


class ProductionFGNInferenceService:
    """Production fixed-domain coastal FGN rollout adapter.

    The adapter supports dependency injection for tests. In production it lazily
    loads checkpointed models, train-fit normalizers, and fixed-domain assets
    declared in the model bundle manifest.
    """

    def __init__(
        self,
        bundle: FGNModelBundle,
        *,
        device: str = "cuda:0",
        member_chunk_size: int = 4,
        preloaded_models: Sequence[object] | None = None,
        preloaded_normalizers: Mapping[str, object] | None = None,
        preloaded_domain_assets: DomainAssets | None = None,
    ) -> None:
        self.bundle = bundle
        self.device_name = str(device)
        self.member_chunk_size = max(1, int(member_chunk_size))
        self._models = list(preloaded_models) if preloaded_models is not None else None
        self._normalizers = dict(preloaded_normalizers) if preloaded_normalizers is not None else None
        self._domain_assets = preloaded_domain_assets
        self._prepared = None

    def _torch(self):
        import torch

        return torch

    def _device(self):
        torch = self._torch()
        if "cuda" in self.device_name and not torch.cuda.is_available():
            _LOG.warning("CUDA device %s requested but unavailable; falling back to CPU.", self.device_name)
            return torch.device("cpu")
        return torch.device(self.device_name)

    @staticmethod
    def _load_array(path: Path):
        torch = __import__("torch")
        suffix = path.suffix.lower()
        if suffix == ".npy":
            return torch.as_tensor(np.load(path), dtype=torch.float32)
        if suffix in {".pt", ".pth"}:
            value = torch.load(path, map_location="cpu")
            if isinstance(value, dict):
                for key in ("tensor", "data", "array", "geometry", "static", "dry_mask"):
                    if key in value:
                        value = value[key]
                        break
            return torch.as_tensor(value, dtype=torch.float32)
        raise ValueError(f"Unsupported domain asset format: {path}. Use .npy, .pt, or .pth.")

    def _load_models(self, device):
        if self._models is not None:
            models = list(self._models)
        else:  # pragma: no cover - exercised by GPU smoke/prod deployment
            import torch

            from neuralop import get_model
            from neuralop.flood.eval.runtime import clone_model_config_for_get_model
            from neuralop.models.base_model import BaseModel
            from neuralop.training.training_state import load_training_state

            def _metadata_candidates(run_dir: Path, alias: str):
                return [
                    run_dir / f"{alias}_metadata.pkl",
                    run_dir / "model_metadata.pkl",
                    run_dir / "best_model_metadata.pkl",
                ]

            def _instantiate_from_metadata(metadata: Dict[str, object]):
                arch_name = str(metadata.get("_name", "")).strip().lower()
                if not arch_name:
                    raise KeyError("Missing _name in checkpoint metadata.")
                model_cls = BaseModel._models.get(arch_name)
                if model_cls is None:
                    raise KeyError(f"Unknown model name {arch_name!r} in checkpoint metadata.")
                init_kwargs = dict(metadata)
                init_args = init_kwargs.pop("args", ())
                init_kwargs.pop("_version", None)
                init_kwargs.pop("_name", None)
                if not isinstance(init_args, (list, tuple)):
                    init_args = (init_args,)
                return model_cls(*init_args, **init_kwargs)

            alias = "best_model" if self.bundle.checkpoint_alias == "best_model" else "model"
            models = []
            for run_dir in self.bundle.checkpoint_dirs:
                model = None
                for meta_path in _metadata_candidates(Path(run_dir), alias):
                    if not meta_path.exists():
                        continue
                    metadata = torch.load(meta_path, map_location="cpu")
                    if isinstance(metadata, dict):
                        model = _instantiate_from_metadata(metadata)
                        break
                if model is None:
                    if self.bundle.model_config_path is None:
                        raise FileNotFoundError(
                            f"No checkpoint metadata found in {run_dir}; model_config_path is required as fallback."
                        )
                    import json

                    with Path(self.bundle.model_config_path).open("r", encoding="utf-8") as handle:
                        cfg = json.load(handle)
                    model = get_model(clone_model_config_for_get_model(cfg))
                load_training_state(save_dir=run_dir, save_name=alias, model=model)
                models.append(model)

        expected = int(self.bundle.n_checkpoints)
        if len(models) != expected:
            raise ValueError(f"Model bundle expected {expected} models, got {len(models)}.")
        loaded = []
        for model in models:
            if hasattr(model, "to"):
                model = model.to(device)
            if hasattr(model, "eval"):
                model = model.eval()
            loaded.append(model)
        return loaded

    def _load_normalizers(self, device):
        if self._normalizers is not None:
            normalizers = dict(self._normalizers)
        else:  # pragma: no cover - exercised by GPU smoke/prod deployment
            from neuralop.data.transforms.normalizers import load_normalizers

            normalizers = load_normalizers(self.bundle.normalizer_path, device=None)
        required = {"geometry", "static", "boundary", "dynamic", "target"}
        missing = sorted(k for k in required if k not in normalizers or normalizers[k] is None)
        if missing:
            raise ValueError(f"Serving normalizer bundle is missing required keys: {missing}.")
        for normalizer in normalizers.values():
            if hasattr(normalizer, "to"):
                normalizer.to(device)
        return normalizers

    def _load_domain_assets(self):
        if self._domain_assets is not None:
            return self._domain_assets
        if self.bundle.geometry_path is None or self.bundle.static_tensor_path is None:
            raise ValueError(
                "Production FGN serving requires geometry_path and static_tensor_path in the model bundle, "
                "or preloaded_domain_assets in tests."
            )
        geometry = self._load_array(Path(self.bundle.geometry_path))
        static = self._load_array(Path(self.bundle.static_tensor_path))
        dry_mask = None
        if self.bundle.structural_dry_mask_path is not None:
            dry_mask = self._load_array(Path(self.bundle.structural_dry_mask_path)).to(dtype=self._torch().bool)
        return DomainAssets(
            geometry=geometry,
            static=static,
            structural_dry_mask=dry_mask,
            query_res=tuple(int(x) for x in self.bundle.query_res),
        )

    @staticmethod
    def _normalize(normalizer, value):
        return normalizer.transform(value.unsqueeze(0)).squeeze(0)

    @staticmethod
    def _build_query_points(geometry_norm, query_res):
        torch = __import__("torch")
        x_vals = geometry_norm[:, 0]
        y_vals = geometry_norm[:, 1]
        tx = torch.linspace(torch.min(x_vals), torch.max(x_vals), int(query_res[0]), device=geometry_norm.device)
        ty = torch.linspace(torch.min(y_vals), torch.max(y_vals), int(query_res[1]), device=geometry_norm.device)
        grid_x, grid_y = torch.meshgrid(tx, ty, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=-1).to(dtype=geometry_norm.dtype)

    def _ensure_loaded(self):
        if self._prepared is not None:
            return self._prepared
        torch = self._torch()
        device = self._device()
        models = self._load_models(device)
        normalizers = self._load_normalizers(device)
        assets = self._load_domain_assets()
        geometry_raw = torch.as_tensor(assets.geometry, dtype=torch.float32, device=device)
        static_raw = torch.as_tensor(assets.static, dtype=torch.float32, device=device)
        if geometry_raw.ndim != 2 or geometry_raw.shape[1] != 2:
            raise ValueError(f"Domain geometry must have shape [n_cells,2], got {tuple(geometry_raw.shape)}.")
        if static_raw.ndim != 2 or static_raw.shape[0] != geometry_raw.shape[0]:
            raise ValueError(
                "Domain static tensor must have shape [n_cells,n_static] with the same n_cells as geometry. "
                f"Got static={tuple(static_raw.shape)}, geometry={tuple(geometry_raw.shape)}."
            )
        geometry_norm = self._normalize(normalizers["geometry"], geometry_raw)
        static_norm = self._normalize(normalizers["static"], static_raw)
        query_points = self._build_query_points(geometry_norm, tuple(int(x) for x in assets.query_res))
        dry_mask = None
        if assets.structural_dry_mask is not None:
            dry_mask = torch.as_tensor(assets.structural_dry_mask, dtype=torch.bool, device=device).reshape(-1)
            if dry_mask.numel() != geometry_raw.shape[0]:
                raise ValueError(
                    f"structural_dry_mask length {dry_mask.numel()} does not match n_cells={geometry_raw.shape[0]}."
                )
        self._prepared = {
            "device": device,
            "models": models,
            "normalizers": normalizers,
            "geometry_norm": geometry_norm,
            "geometry_raw_np": geometry_raw.detach().cpu().numpy().astype(np.float32, copy=False),
            "static_norm": static_norm,
            "query_points": query_points,
            "structural_dry_mask": dry_mask,
        }
        return self._prepared

    def _latent_bank(self, *, n_members: int, seed: int, model_idx: int, device, dtype, batch_size: int = 1):
        torch = self._torch()
        gen = torch.Generator(device=device if device.type == "cuda" else "cpu")
        gen.manual_seed(int(seed) + 1009 * int(model_idx))
        return torch.randn(
            int(n_members),
            int(batch_size),
            int(self.bundle.fgn_noise_dim),
            generator=gen,
            device=device,
            dtype=dtype,
        )

    def _boundary_tensor(self, forcing_input: ForcingInput, *, n_cells: int, device):
        torch = self._torch()
        boundary_matrix = torch.as_tensor(forcing_input.as_boundary_matrix(), dtype=torch.float32, device=device)
        if boundary_matrix.shape[0] < self.bundle.skip_before_timestep + self.bundle.n_history + forcing_input.forecast_steps:
            raise ValueError("Forcing input is shorter than the requested forecast horizon.")
        return boundary_matrix[:, None, :].expand(-1, n_cells, -1).contiguous()

    def run(self, run_spec: RunSpec, forcing_input: ForcingInput) -> ForecastResult:
        torch = self._torch()
        from neuralop.flood.data.structural_dry import (
            apply_structural_dry_zero_mask,
            clamp_structural_dry_normalized_values,
        )

        prepared = self._ensure_loaded()
        device = prepared["device"]
        normalizers = prepared["normalizers"]
        models = prepared["models"]
        geometry = prepared["geometry_norm"]
        static = prepared["static_norm"]
        query_points = prepared["query_points"]
        dry_mask = prepared["structural_dry_mask"]
        n_cells = int(geometry.shape[0])
        n_time = int(run_spec.forecast_steps)
        if n_time != int(forcing_input.forecast_steps):
            raise ValueError(
                f"RunSpec forecast_steps={n_time} does not match forcing forecast_steps={forcing_input.forecast_steps}."
            )

        boundary_raw = self._boundary_tensor(forcing_input, n_cells=n_cells, device=device)
        boundary_norm = self._normalize(normalizers["boundary"], boundary_raw)
        zero_dyn_raw = torch.zeros((self.bundle.n_history, n_cells, 1), dtype=torch.float32, device=device)
        zero_dyn_norm = self._normalize(normalizers["dynamic"], zero_dyn_raw)
        start = int(self.bundle.skip_before_timestep) + int(self.bundle.n_history)
        initial_boundary = boundary_norm[self.bundle.skip_before_timestep:start].clone()
        boundary_future = boundary_norm[start:start + n_time]
        if initial_boundary.shape[0] != int(self.bundle.n_history):
            raise ValueError("Boundary history window is empty or malformed.")
        if boundary_future.shape[0] != n_time:
            raise ValueError("Boundary future window is shorter than forecast_steps.")

        outputs = []
        member_model_id: list[str] = []
        member_sample_id: list[int] = []
        dtype = geometry.dtype
        n_per_model = int(self.bundle.members_per_checkpoint)
        with torch.no_grad():
            for model_idx, model in enumerate(models):
                latent_bank = self._latent_bank(
                    n_members=n_per_model,
                    seed=int(run_spec.seed),
                    model_idx=model_idx,
                    device=device,
                    dtype=dtype,
                )
                current_dynamics = [zero_dyn_norm.clone() for _ in range(n_per_model)]
                current_boundary = initial_boundary.clone()
                model_members = []
                for t in range(n_time):
                    pred_chunks = []
                    for start_idx in range(0, n_per_model, self.member_chunk_size):
                        end_idx = min(n_per_model, start_idx + self.member_chunk_size)
                        chunk_size = end_idx - start_idx
                        dyn_flat = torch.stack(
                            [
                                current_dynamics[member_idx].permute(1, 0, 2).reshape(n_cells, -1)
                                for member_idx in range(start_idx, end_idx)
                            ],
                            dim=0,
                        )
                        bc_flat = current_boundary.permute(1, 0, 2).reshape(1, n_cells, -1).expand(
                            chunk_size, -1, -1
                        )
                        x = torch.cat([static.unsqueeze(0).expand(chunk_size, -1, -1), bc_flat, dyn_flat], dim=2)
                        z = latent_bank[start_idx:end_idx, 0, :]
                        pred = model(
                            input_geom=geometry.unsqueeze(0).expand(chunk_size, -1, -1),
                            latent_queries=query_points.unsqueeze(0).expand(chunk_size, -1, -1, -1),
                            output_queries=geometry.unsqueeze(0).expand(chunk_size, -1, -1),
                            x=x,
                            ada_in=z,
                        )
                        if pred.ndim == 4 and pred.shape[1] == 1:
                            pred = pred[:, 0]
                        if pred.ndim != 3:
                            raise ValueError(f"FGN model output must have shape [batch,n_cells,channels], got {tuple(pred.shape)}.")
                        pred_chunks.append(pred)
                    pred_stack = torch.cat(pred_chunks, dim=0).unsqueeze(1)  # [M,1,N,1]
                    inv_pred = normalizers["target"].inverse_transform(pred_stack.squeeze(1))
                    inv_pred = apply_structural_dry_zero_mask(inv_pred, structural_dry_mask=dry_mask)
                    inv_pred = torch.clamp(inv_pred, min=0.0)
                    model_members.append(inv_pred[..., 0].detach().cpu())
                    update_stack = clamp_structural_dry_normalized_values(
                        pred_stack,
                        structural_dry_mask=dry_mask,
                        normalizer=normalizers["target"],
                    )
                    for member_idx in range(n_per_model):
                        current_dynamics[member_idx] = torch.cat(
                            [current_dynamics[member_idx][1:], update_stack[member_idx, 0].unsqueeze(0)],
                            dim=0,
                        )
                    current_boundary = torch.cat([current_boundary[1:], boundary_future[t].unsqueeze(0)], dim=0)
                outputs.append(torch.stack(model_members, dim=1))  # [M,T,N]
                member_model_id.extend([f"model_{model_idx}"] * n_per_model)
                member_sample_id.extend(list(range(n_per_model)))

        members = torch.cat(outputs, dim=0).numpy().astype(np.float32, copy=False)
        lead_hours = (np.arange(1, n_time + 1, dtype=np.float32) * float(self.bundle.dt_seconds)) / 3600.0
        wettable = None
        if dry_mask is not None:
            wettable = (~dry_mask).detach().cpu().numpy().astype(bool)
        else:
            wettable = np.ones(n_cells, dtype=bool)
        return ForecastResult(
            members_wd=members,
            lead_time_hours=lead_hours,
            wettable_mask=wettable,
            metadata={
                "adapter": "production_fgn",
                "bundle_id": self.bundle.bundle_id,
                "member_model_id": member_model_id,
                "member_sample_id": member_sample_id,
                "seed": int(run_spec.seed),
                "geometry_xy": prepared.get("geometry_raw_np"),
                "fgn_latent_temporal_mode": "persistent",
                "fgn_ar_state_update": "member_feedback",
            },
        ).validate()
