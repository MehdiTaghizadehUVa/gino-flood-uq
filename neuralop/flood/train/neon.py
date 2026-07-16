"""Training helpers for NEON-aligned Stage-2 FGNO experiments."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from neuralop.flood.neon import (
    base_rmse_from_reference,
    NEONEpistemicCorrection,
    PersistentDirichletParticleControl,
    NEONStage2LossOutput,
    NEONStage2LossWeights,
    cancellation_diagnostics,
    epistemic_variance_diagnostics,
    compute_stage2_loss,
    epistemic_bootstrap_weights,
    epistemic_member_bootstrap_weights,
    freeze_stage1_model,
    crossed_fair_crps_members,
    fair_crps_members,
    fixed_support_fair_crps_members,
    per_epistemic_fair_crps,
    prior_psi_floor_diagnostic,
    sample_epistemic_indices,
    save_neon_stage2_checkpoint,
    stage2_fit_score,
)


@dataclass
class FrozenFGNOFeatureBatch:
    """Frozen Stage-1 outputs for one Stage-2 optimization batch."""

    base_prediction: torch.Tensor
    features: torch.Tensor
    aleatory_latents: torch.Tensor
    canonical_mean_features: torch.Tensor | None = None
    canonical_latent_hash: str | None = None


def _json_safe(value: Any) -> Any:
    """Convert common training values to JSON-safe scalars/containers."""

    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    return str(value)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_json_safe(payload), fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_json_safe(payload), sort_keys=True))
        fh.write("\n")


@dataclass
class NEONTrainingProgressReporter:
    """Live file/log reporter for long NEON Stage-2 training jobs.

    The reporter intentionally sits outside the optimizer and loss code. It
    records observable training progress without changing the mathematical
    objective, gradients, or checkpoint semantics.
    """

    output_dir: Any
    log_interval_effective_batches: int = 10
    logger: Optional[logging.Logger] = None
    events_path: Optional[Any] = None
    partial_history_path: Optional[Any] = None
    latest_status_path: Optional[Any] = None
    _start_time: float = field(default_factory=time.time, init=False)

    def __post_init__(self) -> None:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.output_dir = out
        self.log_interval_effective_batches = max(1, int(self.log_interval_effective_batches))
        self.events_path = Path(self.events_path) if self.events_path is not None else out / "progress_events.jsonl"
        self.partial_history_path = (
            Path(self.partial_history_path)
            if self.partial_history_path is not None
            else out / "history_partial.jsonl"
        )
        self.latest_status_path = (
            Path(self.latest_status_path)
            if self.latest_status_path is not None
            else out / "latest_status.json"
        )

    def should_report_batch(self, batch_idx: int, total_batches: int) -> bool:
        return (
            int(batch_idx) == 1
            or int(batch_idx) == int(total_batches)
            or int(batch_idx) % self.log_interval_effective_batches == 0
        )

    def _record(
        self,
        event: str,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        log_message: Optional[str] = None,
    ) -> None:
        now = time.time()
        record: dict[str, Any] = {
            "event": str(event),
            "timestamp_unix": now,
            "elapsed_sec": now - self._start_time,
        }
        if payload:
            record.update(dict(payload))
        _append_jsonl(Path(self.events_path), record)
        _atomic_write_json(Path(self.latest_status_path), record)
        if self.logger is not None and log_message:
            self.logger.info(log_message)

    def training_start(
        self,
        *,
        n_epochs: int,
        n_train_families: int,
        n_val_families: int,
        family_batch_size: int,
        effective_batch_size: int,
        m_train: int,
        k_train: int,
        d_e: int,
        objective: str,
        latent_bank_count: int,
    ) -> None:
        self._record(
            "training_start",
            {
                "n_epochs": int(n_epochs),
                "n_train_families": int(n_train_families),
                "n_val_families": int(n_val_families),
                "family_batch_size": int(family_batch_size),
                "effective_batch_size": int(effective_batch_size),
                "m_train": int(m_train),
                "k_train": int(k_train),
                "d_e": int(d_e),
                "objective": str(objective),
                "latent_bank_count": int(latent_bank_count),
                "output_dir": self.output_dir,
            },
            log_message=(
                "NEON training start: epochs=%d train_families=%d val_families=%d "
                "effective_batch=%d micro_batch=%d M=%d K=%d d_e=%d objective=%s"
                % (
                    int(n_epochs),
                    int(n_train_families),
                    int(n_val_families),
                    int(effective_batch_size),
                    int(family_batch_size),
                    int(m_train),
                    int(k_train),
                    int(d_e),
                    str(objective),
                )
            ),
        )

    def epoch_start(self, *, epoch: int, n_epochs: int, n_train_families: int) -> None:
        self._record(
            "epoch_start",
            {
                "epoch": int(epoch),
                "epoch_display": int(epoch) + 1,
                "n_epochs": int(n_epochs),
                "n_train_families": int(n_train_families),
            },
            log_message=f"NEON epoch {int(epoch) + 1}/{int(n_epochs)} started",
        )

    def train_progress(
        self,
        *,
        epoch: int,
        n_epochs: int,
        effective_batch_idx: int,
        total_effective_batches: int,
        train_families_done: int,
        n_train_families: int,
        running_train_fit: float,
        running_train_total: float,
        epoch_elapsed_sec: float,
    ) -> None:
        rate = float(train_families_done) / max(float(epoch_elapsed_sec), 1.0e-9)
        remaining = max(int(n_train_families) - int(train_families_done), 0)
        eta_epoch_sec = remaining / rate if rate > 0.0 else None
        payload = {
            "epoch": int(epoch),
            "epoch_display": int(epoch) + 1,
            "n_epochs": int(n_epochs),
            "effective_batch_idx": int(effective_batch_idx),
            "total_effective_batches": int(total_effective_batches),
            "train_families_done": int(train_families_done),
            "n_train_families": int(n_train_families),
            "running_train_fit": float(running_train_fit),
            "running_train_total": float(running_train_total),
            "epoch_elapsed_sec": float(epoch_elapsed_sec),
            "families_per_sec": float(rate),
            "eta_epoch_sec": eta_epoch_sec,
        }
        log = (
            "NEON epoch %d/%d progress: batch %d/%d families %d/%d "
            "train_fit=%.6f train_total=%.6f epoch_elapsed=%.1fs"
            % (
                int(epoch) + 1,
                int(n_epochs),
                int(effective_batch_idx),
                int(total_effective_batches),
                int(train_families_done),
                int(n_train_families),
                float(running_train_fit),
                float(running_train_total),
                float(epoch_elapsed_sec),
            )
        )
        if eta_epoch_sec is not None:
            log += " eta_epoch=%.1fs" % float(eta_epoch_sec)
        self._record("train_progress", payload, log_message=log)

    def validation_start(self, *, epoch: int, n_epochs: int, n_val_families: int) -> None:
        self._record(
            "validation_start",
            {
                "epoch": int(epoch),
                "epoch_display": int(epoch) + 1,
                "n_epochs": int(n_epochs),
                "n_val_families": int(n_val_families),
            },
            log_message=(
                f"NEON epoch {int(epoch) + 1}/{int(n_epochs)} validation started "
                f"(families={int(n_val_families)})"
            ),
        )

    def checkpoint_saved(self, *, epoch: int, checkpoint_path: Any, val_fit: float) -> None:
        self._record(
            "checkpoint_saved",
            {
                "epoch": int(epoch),
                "epoch_display": int(epoch) + 1,
                "checkpoint_path": checkpoint_path,
                "val_fit": float(val_fit),
            },
            log_message=(
                "NEON checkpoint saved: epoch=%d val_fit=%.6f path=%s"
                % (int(epoch) + 1, float(val_fit), str(checkpoint_path))
            ),
        )

    def epoch_end(
        self,
        *,
        row: Mapping[str, Any],
        best_epoch: int,
        best_val_fit: float,
        improved: bool,
        epoch_elapsed_sec: float,
    ) -> None:
        payload = dict(row)
        payload.update(
            {
                "best_epoch": int(best_epoch),
                "best_epoch_display": int(best_epoch) + 1 if int(best_epoch) >= 0 else None,
                "best_val_fit": float(best_val_fit),
                "improved": bool(improved),
                "epoch_elapsed_sec": float(epoch_elapsed_sec),
            }
        )
        _append_jsonl(Path(self.partial_history_path), payload)
        self._record(
            "epoch_end",
            payload,
            log_message=(
                "NEON epoch %d done: train_fit=%.6f train_total=%.6f val_fit=%.6f "
                "best_epoch=%s best_val_fit=%.6f elapsed=%.1fs"
                % (
                    int(row["epoch"]) + 1,
                    float(row["train_fit"]),
                    float(row["train_total"]),
                    float(row["val_fit"]),
                    str(int(best_epoch) + 1 if int(best_epoch) >= 0 else None),
                    float(best_val_fit),
                    float(epoch_elapsed_sec),
                )
            ),
        )

    def training_end(self, *, best_epoch: int, best_val_fit: float, n_epochs: int) -> None:
        self._record(
            "training_end",
            {
                "best_epoch": int(best_epoch),
                "best_epoch_display": int(best_epoch) + 1 if int(best_epoch) >= 0 else None,
                "best_val_fit": float(best_val_fit),
                "n_epochs": int(n_epochs),
            },
            log_message=(
                "NEON training finished: best_epoch=%s best_val_fit=%.6f"
                % (str(int(best_epoch) + 1 if int(best_epoch) >= 0 else None), float(best_val_fit))
            ),
        )


def assert_optimizer_excludes_stage1(
    *,
    stage1_model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Fail early if a Stage-2 optimizer accidentally owns Stage-1 parameters."""

    stage1_ids = {id(param) for param in stage1_model.parameters()}
    for group_idx, group in enumerate(optimizer.param_groups):
        for param_idx, param in enumerate(group.get("params", [])):
            if id(param) in stage1_ids:
                raise ValueError(
                    "Stage-2 optimizer includes frozen Stage-1 parameter "
                    f"(group={group_idx}, param={param_idx})."
                )


def _latent_for_member(aleatory_latents: torch.Tensor, member_idx: int, batch_size: int) -> torch.Tensor:
    latent = aleatory_latents[int(member_idx)]
    if latent.ndim == 1:
        latent = latent.unsqueeze(0).expand(batch_size, -1)
    elif latent.ndim == 2 and latent.shape[0] == 1 and batch_size > 1:
        latent = latent.expand(batch_size, -1)
    elif latent.ndim != 2 or latent.shape[0] != batch_size:
        raise ValueError(
            "aleatory_latents must have shape [K, d_a] or [K, B, d_a], "
            f"got {tuple(aleatory_latents.shape)} for batch_size={batch_size}."
        )
    return latent


def _ensure_single_rollout_time_dim(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """Normalize one-step FGNO tensors to the Stage-2 rollout contract.

    GINO one-step outputs are usually ``[B, Nv, C]`` before stacking over
    aleatory members. After stacking they become ``[B, K, Nv, C]``. Stage 2
    consistently consumes rollout tensors ``[B, K, T, Nv, C]``; for one-step
    feature extraction the implicit rollout length is ``T=1``.
    """

    if tensor.ndim == 4:
        return tensor.unsqueeze(2)
    if tensor.ndim == 5:
        return tensor
    raise ValueError(
        f"{name} must have shape [B, K, Nv, C] or [B, K, T, Nv, C], "
        f"got {tuple(tensor.shape)}."
    )


def collect_frozen_fgno_features(
    *,
    stage1_model: nn.Module,
    model_kwargs: dict[str, Any],
    aleatory_latents: torch.Tensor,
    feature_source: str = "decoder_pre_projection",
) -> FrozenFGNOFeatureBatch:
    """Run frozen FGNO over ``K`` aleatory latents and collect decoder features.

    ``model_kwargs`` should contain the normal GINO forward inputs except
    ``ada_in``, ``return_features``, and ``feature_source``.
    """

    freeze_stage1_model(stage1_model)
    x = model_kwargs.get("x")
    if x is None:
        raise ValueError("model_kwargs must contain x so batch size can be inferred.")
    batch_size = int(x.shape[0])
    if aleatory_latents.ndim not in {2, 3}:
        raise ValueError(
            "aleatory_latents must have shape [K, d_a] or [K, B, d_a], "
            f"got {tuple(aleatory_latents.shape)}."
        )
    base_members = []
    feature_members = []
    with torch.no_grad():
        for member_idx in range(int(aleatory_latents.shape[0])):
            latent = _latent_for_member(aleatory_latents, member_idx, batch_size)
            out = stage1_model(
                **model_kwargs,
                ada_in=latent,
                return_features=True,
                feature_source=feature_source,
            )
            if not isinstance(out, dict) or "prediction" not in out or "features" not in out:
                raise TypeError(
                    "Stage-1 model must return a feature dictionary when return_features=True."
                )
            features = out["features"].get(feature_source)
            if features is None and str(feature_source).strip().lower() == "all":
                # GINO returns a dictionary of all requested features; Stage-2 v1
                # consumes the decoder-pre-projection tensor from that bundle.
                features = out["features"].get("decoder_pre_projection")
            if features is None:
                raise KeyError(f"Stage-1 output did not include feature_source={feature_source!r}.")
            base_members.append(out["prediction"].detach())
            feature_members.append(features.detach())
    base_prediction = _ensure_single_rollout_time_dim(
        torch.stack(base_members, dim=1),
        name="base_prediction",
    )
    features = _ensure_single_rollout_time_dim(
        torch.stack(feature_members, dim=1),
        name="features",
    )
    return FrozenFGNOFeatureBatch(
        base_prediction=base_prediction,
        features=features,
        aleatory_latents=aleatory_latents.detach(),
    )


@dataclass
class ARRolloutOutput:
    """Autoregressive rollout base predictions and features for one hydrograph.

    Shapes follow ``[K, T, Nv, C]`` (predictions) and ``[K, T, Nv, C_phi]``
    (features), where ``K`` indexes aleatory latents and ``T`` is the rollout
    horizon.
    """

    base_prediction: torch.Tensor
    features: torch.Tensor


# Type of the injected per-step function: (history[n_hist,Nv,C], t, latent[d_a])
# -> (prediction[Nv,C], feature[Nv,C_phi]).
ARStepFn = Callable[[torch.Tensor, int, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def autoregressive_feature_rollout(
    *,
    step_fn: ARStepFn,
    initial_histories: torch.Tensor,
    aleatory_latents: torch.Tensor,
    rollout_length: int,
    detach: bool = True,
) -> ARRolloutOutput:
    """Run a member-feedback autoregressive rollout, collecting predictions + features.

    This is the model-agnostic core of NEON Stage-2 feature collection. It owns
    the aleatory-member loop, the dynamic-history sliding window, member
    feedback (each member's own prediction feeds back into its own history),
    and stacking over time. The model-specific tensor assembly and forward pass
    live entirely inside the injected ``step_fn``.

    Parameters
    ----------
    step_fn
        ``(history, t, latent) -> (prediction, feature)`` where ``history`` is
        ``[n_history, Nv, C]``, ``t`` is the 0-based rollout step, ``latent`` is
        ``[d_a]``, and the returned prediction/feature are ``[Nv, C]`` /
        ``[Nv, C_phi]``.
    initial_histories
        ``[K, n_history, Nv, C]`` warm-start histories, one per aleatory member.
    aleatory_latents
        ``[K, d_a]`` persistent aleatory latents (one per member, held fixed
        across the rollout).
    rollout_length
        Number of autoregressive steps ``T`` (>= 1).
    detach
        Detach predictions/features each step (default True). Stage-2 consumes
        frozen FGNO outputs, so features must not carry Stage-1 gradients.
    """
    T = int(rollout_length)
    if T < 1:
        raise ValueError(f"rollout_length must be >= 1, got {rollout_length}.")
    if initial_histories.ndim != 4:
        raise ValueError(
            "initial_histories must have shape [K, n_history, Nv, C], "
            f"got {tuple(initial_histories.shape)}."
        )
    K = int(initial_histories.shape[0])
    if aleatory_latents.ndim != 2 or int(aleatory_latents.shape[0]) != K:
        raise ValueError(
            "aleatory_latents must have shape [K, d_a] matching initial_histories K="
            f"{K}, got {tuple(aleatory_latents.shape)}."
        )

    preds_per_member: list[torch.Tensor] = []
    feats_per_member: list[torch.Tensor] = []
    for member_idx in range(K):
        history = initial_histories[member_idx].clone()
        latent = aleatory_latents[member_idx]
        step_preds: list[torch.Tensor] = []
        step_feats: list[torch.Tensor] = []
        for t in range(T):
            pred_t, feat_t = step_fn(history, t, latent)
            if detach:
                pred_t = pred_t.detach()
                feat_t = feat_t.detach()
            step_preds.append(pred_t)
            step_feats.append(feat_t)
            # Member feedback: slide the history window forward with this
            # member's own prediction.
            history = torch.cat([history[1:], pred_t.unsqueeze(0)], dim=0)
        preds_per_member.append(torch.stack(step_preds, dim=0))
        feats_per_member.append(torch.stack(step_feats, dim=0))

    return ARRolloutOutput(
        base_prediction=torch.stack(preds_per_member, dim=0),
        features=torch.stack(feats_per_member, dim=0),
    )


def _build_frozen_gino_step_fn(
    *,
    stage1_model: nn.Module,
    static: torch.Tensor,
    geometry: torch.Tensor,
    query_points: torch.Tensor,
    boundary_sequence: torch.Tensor,
    n_history: int,
    feature_source: str,
    structural_dry_mask: torch.Tensor | None = None,
    target_normalizer=None,
) -> ARStepFn:
    """Build the per-step closure that runs one frozen GINO forward.

    Mirrors the flood rollout tensor assembly: the GINO input ``x`` is
    ``concat([static, flattened boundary window, flattened dynamic history])``
    on the native mesh. The boundary window for step ``t`` is
    ``boundary_sequence[t : t + n_history]``.
    """

    def step_fn(history: torch.Tensor, t: int, latent: torch.Tensor):
        # history: [n_history, Nv, C]
        nv = history.shape[1]
        dyn_flat = history.permute(1, 0, 2).reshape(1, nv, -1)  # [1, Nv, n_history*C]
        bwin = boundary_sequence[t : t + n_history]             # [n_history, Nv, Cb]
        if bwin.shape[0] != n_history:
            raise ValueError(
                "boundary_sequence too short for rollout: needs "
                f"{n_history} frames at step {t}, has {bwin.shape[0]}."
            )
        bc_flat = bwin.permute(1, 0, 2).reshape(1, nv, -1)      # [1, Nv, n_history*Cb]
        x = torch.cat([static, bc_flat, dyn_flat], dim=2)
        out = stage1_model(
            x=x,
            input_geom=geometry,
            latent_queries=query_points,
            output_queries=geometry,
            ada_in=latent.unsqueeze(0),
            return_features=True,
            feature_source=feature_source,
        )
        if not isinstance(out, dict) or "prediction" not in out or "features" not in out:
            raise TypeError(
                "Stage-1 model must return a feature dictionary when return_features=True."
            )
        feat = out["features"].get(feature_source)
        if feat is None:
            feat = out["features"].get("decoder_pre_projection")
        if feat is None:
            raise KeyError(f"Stage-1 output did not include feature_source={feature_source!r}.")
        pred = out["prediction"][0]   # [Nv, C]
        feat = feat[0]                # [Nv, C_phi] or [C_phi, Nv]
        # GINO's decoder pre-projection feature is returned channels-first
        # ([C_phi, Nv]); orient it to node-first [Nv, C_phi] to match the
        # prediction and the Stage-2 [.., Nv, C_phi] contract.
        n_cells = int(pred.shape[0])
        if feat.shape[0] != n_cells and feat.shape[-1] == n_cells:
            feat = feat.transpose(0, 1).contiguous()
        if structural_dry_mask is not None:
            from neuralop.flood.data.structural_dry import (
                clamp_structural_dry_normalized_values,
            )

            pred = clamp_structural_dry_normalized_values(
                pred,
                structural_dry_mask=structural_dry_mask.to(pred.device),
                normalizer=target_normalizer,
            )
        return pred, feat

    return step_fn


def collect_frozen_fgno_rollout_features(
    *,
    stage1_model: nn.Module,
    static: torch.Tensor,
    geometry: torch.Tensor,
    query_points: torch.Tensor,
    boundary_sequence: torch.Tensor,
    initial_histories: torch.Tensor,
    aleatory_latents: torch.Tensor,
    rollout_length: int,
    n_history: int = 3,
    feature_source: str = "decoder_pre_projection",
    structural_dry_mask: torch.Tensor | None = None,
    target_normalizer=None,
) -> "FrozenFGNOFeatureBatch":
    """Autoregressively roll out a frozen FGNO, collecting base predictions + features.

    Unlike :func:`collect_frozen_fgno_features` (single-step, ``T=1``), this
    produces a real rollout of horizon ``rollout_length`` with member feedback,
    matching the plan's Stage-2 training protocol. Returns a
    :class:`FrozenFGNOFeatureBatch` with ``B=1`` (one hydrograph):
    ``base_prediction`` ``[1, K, T, Nv, C]`` and ``features`` ``[1, K, T, Nv, C_phi]``.
    """
    freeze_stage1_model(stage1_model)
    step_fn = _build_frozen_gino_step_fn(
        stage1_model=stage1_model,
        static=static,
        geometry=geometry,
        query_points=query_points,
        boundary_sequence=boundary_sequence,
        n_history=int(n_history),
        feature_source=feature_source,
        structural_dry_mask=structural_dry_mask,
        target_normalizer=target_normalizer,
    )
    with torch.no_grad():
        rollout = autoregressive_feature_rollout(
            step_fn=step_fn,
            initial_histories=initial_histories,
            aleatory_latents=aleatory_latents,
            rollout_length=rollout_length,
            detach=True,
        )
    return FrozenFGNOFeatureBatch(
        base_prediction=rollout.base_prediction.unsqueeze(0),
        features=rollout.features.unsqueeze(0),
        aleatory_latents=aleatory_latents.detach(),
    )


def neon_stage2_training_step(
    *,
    module: NEONEpistemicCorrection,
    optimizer: torch.optim.Optimizer,
    base_prediction: torch.Tensor,
    features: torch.Tensor,
    reference: torch.Tensor,
    z_e: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    node_coords: torch.Tensor | None = None,
    edge_index: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    zero_threshold: float | torch.Tensor = 0.0,
    loss_weights: NEONStage2LossWeights | None = None,
    grad_clip_norm: float | None = None,
    objective: str = "per_epistemic_fcrps",
    epistemic_chunk_size: int | None = None,
    sample_weights: torch.Tensor | None = None,
    member_weights: torch.Tensor | None = None,
    canonical_mean_features: torch.Tensor | None = None,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
) -> NEONStage2LossOutput:
    """Perform one Stage-2 optimization step from frozen FGNO tensors."""

    module.train()
    if z_e is None:
        z_e = sample_epistemic_indices(
            4,
            module.epistemic_dim,
            device=features.device,
            dtype=features.dtype,
        )
    if zero_grad:
        optimizer.zero_grad(set_to_none=True)
    total_m = int(z_e.shape[0])
    chunk = int(epistemic_chunk_size) if epistemic_chunk_size else total_m
    chunk = max(1, min(chunk, total_m))
    if chunk >= total_m:
        out = module(
            base_prediction,
            features,
            z_e,
            node_coords=node_coords,
            canonical_mean_features=canonical_mean_features,
        )
        losses = compute_stage2_loss(
            prediction=out.prediction,
            reference=reference,
            correction=out.correction,
            trainable_correction=out.trainable_correction,
            module=module,
            weights=weights,
            sample_weights=sample_weights,
            member_weights=member_weights,
            edge_index=edge_index,
            edge_weights=edge_weights,
            zero_threshold=zero_threshold,
            loss_weights=loss_weights,
            objective=objective,
        )
        losses.diagnostics = cancellation_diagnostics(
            trainable_correction=out.trainable_correction,
            prior_correction=out.prior_correction,
            alpha=module.alpha,
        )
        _mbar_prior = (float(module.alpha) * out.prior_correction.detach().float()).mean(dim=2)
        _mbar_total = out.trainable_correction.detach().float().mean(dim=2) + _mbar_prior
        losses.diagnostics.update(
            epistemic_variance_diagnostics(
                mbar_total=_mbar_total,
                mbar_prior_scaled=_mbar_prior,
            )
        )
        losses.diagnostics.update(
            prior_psi_floor_diagnostic(
                module=module,
                features=features,
                z_e=z_e,
                node_coords=node_coords,
            )
        )
        (losses.total * float(loss_scale)).backward()
    else:
        # Chunk the epistemic (M) axis with gradient accumulation to cap
        # activation memory. Per-epistemic fair CRPS and the correction
        # penalties are computed independently per z_e, so the mean over M
        # equals the size-weighted sum of per-chunk means (exact gradients).
        agg = {n: 0.0 for n in ("total", "fit", "rpf", "graph", "time", "pos", "mag")}
        diag_agg: dict[str, float] = {}
        mbar_train_chunks: list[torch.Tensor] = []
        mbar_prior_chunks: list[torch.Tensor] = []
        for start in range(0, total_m, chunk):
            z_chunk = z_e[start : start + chunk]
            weight_chunk = None
            if sample_weights is not None:
                weight_chunk = sample_weights[:, start : start + chunk]
            member_weight_chunk = None
            if member_weights is not None:
                member_weight_chunk = member_weights[:, start : start + chunk, :]
            scale = float(z_chunk.shape[0]) / float(total_m)
            out = module(
                base_prediction,
                features,
                z_chunk,
                node_coords=node_coords,
                canonical_mean_features=canonical_mean_features,
            )
            losses_c = compute_stage2_loss(
                prediction=out.prediction,
                reference=reference,
                correction=out.correction,
                trainable_correction=out.trainable_correction,
                module=module,
                weights=weights,
                sample_weights=weight_chunk,
                member_weights=member_weight_chunk,
                edge_index=edge_index,
                edge_weights=edge_weights,
                zero_threshold=zero_threshold,
                loss_weights=loss_weights,
                objective=objective,
            )
            (losses_c.total * scale * float(loss_scale)).backward()
            for n in agg:
                agg[n] += float(getattr(losses_c, n).item()) * scale
            diag_c = cancellation_diagnostics(
                trainable_correction=out.trainable_correction,
                prior_correction=out.prior_correction,
                alpha=module.alpha,
            )
            for key, value in diag_c.items():
                diag_agg[key] = diag_agg.get(key, 0.0) + float(value) * scale
            mbar_prior_chunks.append(
                (float(module.alpha) * out.prior_correction.detach().float()).mean(dim=2)
            )
            mbar_train_chunks.append(out.trainable_correction.detach().float().mean(dim=2))
        _mbar_prior = torch.cat(mbar_prior_chunks, dim=1)
        _mbar_total = torch.cat(mbar_train_chunks, dim=1) + _mbar_prior
        diag_agg.update(
            epistemic_variance_diagnostics(
                mbar_total=_mbar_total,
                mbar_prior_scaled=_mbar_prior,
            )
        )
        diag_agg.update(
            prior_psi_floor_diagnostic(
                module=module,
                features=features,
                z_e=z_e,
                node_coords=node_coords,
            )
        )
        losses = NEONStage2LossOutput(
            total=torch.tensor(agg["total"]),
            fit=torch.tensor(agg["fit"]),
            rpf=torch.tensor(agg["rpf"]),
            graph=torch.tensor(agg["graph"]),
            time=torch.tensor(agg["time"]),
            pos=torch.tensor(agg["pos"]),
            mag=torch.tensor(agg["mag"]),
            diagnostics=diag_agg,
        )
    if optimizer_step and grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            float(grad_clip_norm),
        )
    if optimizer_step:
        optimizer.step()
    return losses


def neon_stage2_eval_forward(
    *,
    module: NEONEpistemicCorrection,
    base_prediction: torch.Tensor,
    features: torch.Tensor,
    z_e: torch.Tensor,
    node_coords: torch.Tensor | None = None,
    canonical_mean_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inference helper returning nested corrected predictions only."""

    module.eval()
    with torch.no_grad():
        return module(
            base_prediction,
            features,
            z_e,
            node_coords=node_coords,
            canonical_mean_features=canonical_mean_features,
        ).prediction


def neon_stage2_eval_forward_chunked(
    *,
    module: nn.Module,
    base_prediction: torch.Tensor,
    features: torch.Tensor,
    z_e: torch.Tensor,
    k_chunk: int,
    epistemic_chunk_size: int | None = None,
    node_coords: torch.Tensor | None = None,
    canonical_mean_features: torch.Tensor | None = None,
    output_device: str | torch.device | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Evaluate nested predictions with independently bounded M and K chunks."""

    if base_prediction.ndim != 5 or features.ndim != 5:
        raise ValueError("base_prediction and features must have shape [B,K,T,Nv,C]")
    if z_e.ndim != 2:
        raise ValueError("z_e must have shape [M,d_e]")
    if int(base_prediction.shape[1]) != int(features.shape[1]):
        raise ValueError("base_prediction and features must use the same K")
    k_total = int(features.shape[1])
    m_total = int(z_e.shape[0])
    k_step = max(1, min(int(k_chunk), k_total))
    m_requested = m_total if epistemic_chunk_size is None else int(epistemic_chunk_size)
    m_step = max(1, min(m_requested, m_total))
    target_device = None if output_device is None else torch.device(output_device)

    module.eval()
    m_parts = []
    with torch.no_grad():
        for ms in range(0, m_total, m_step):
            k_parts = []
            z_chunk = z_e[ms : ms + m_step]
            for ks in range(0, k_total, k_step):
                prediction = module(
                    base_prediction[:, ks : ks + k_step],
                    features[:, ks : ks + k_step],
                    z_chunk,
                    node_coords=node_coords,
                    canonical_mean_features=canonical_mean_features,
                ).prediction.detach()
                if target_device is not None or output_dtype is not None:
                    prediction = prediction.to(
                        device=prediction.device if target_device is None else target_device,
                        dtype=prediction.dtype if output_dtype is None else output_dtype,
                    )
                k_parts.append(prediction)
            m_parts.append(torch.cat(k_parts, dim=2))
    return torch.cat(m_parts, dim=1)


def build_neon_stage2_optimizer(
    module: NEONEpistemicCorrection,
    *,
    learning_rate: float = 1.0e-4,
    weight_decay: float = 1.0e-4,
) -> torch.optim.Optimizer:
    """Build the default optimizer for Stage-2 trainable EpiNet parameters."""

    return torch.optim.AdamW(
        [param for param in module.parameters() if param.requires_grad],
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


# ---------------------------------------------------------------------------
# Family-level Stage-2 training loop (Gap 3)
# ---------------------------------------------------------------------------


@dataclass
class NEONFamilySample:
    """One grouped-hydrograph event family for Stage-2 training/validation.

    ``reference`` is the HEC-RAS reference ensemble ``[R, T, Nv, C]`` (R>1).
    The optional domain-input fields are consumed by a production feature
    collector that runs the frozen FGNO rollout; tests inject a fake collector
    and only need ``reference`` (plus optional scoring weights / graph edges).
    """

    family_id: str
    reference: torch.Tensor
    weights: Optional[torch.Tensor] = None
    edge_index: Optional[torch.Tensor] = None
    edge_weights: Optional[torch.Tensor] = None
    # Optional fixed-domain inputs for the production frozen-FGNO rollout.
    static: Optional[torch.Tensor] = None
    geometry: Optional[torch.Tensor] = None
    query_points: Optional[torch.Tensor] = None
    boundary_sequence: Optional[torch.Tensor] = None
    initial_histories: Optional[torch.Tensor] = None
    structural_dry_mask: Optional[torch.Tensor] = None


@dataclass
class NEONTrainingResult:
    """Outcome of a Stage-2 training run."""

    history: list[dict[str, float]]
    best_epoch: int
    best_val_fit: float


def save_neon_stage2_training_state(
    path: Any,
    *,
    module: NEONEpistemicCorrection,
    optimizer: torch.optim.Optimizer,
    metadata: Optional[Mapping[str, Any]] = None,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    best_epoch: int = -1,
    best_val_fit: float = math.inf,
    next_epoch: int = 0,
    generator: Optional[torch.Generator] = None,
    particle_training_step: int = 0,
    n_ineligible_epochs: int = 0,
) -> None:
    """Save resumable Stage-2 training state after a completed epoch."""

    state_path = Path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metadata": dict(metadata or {}),
        "history": [dict(row) for row in (history or [])],
        "best_epoch": int(best_epoch),
        "best_val_fit": float(best_val_fit),
        "next_epoch": int(next_epoch),
        "generator_state": (
            None if generator is None else generator.get_state().detach().cpu().clone()
        ),
        "particle_training_step": int(particle_training_step),
        "n_ineligible_epochs": int(n_ineligible_epochs),
    }
    tmp = state_path.with_name(f"{state_path.name}.tmp{os.getpid()}")
    torch.save(payload, tmp)
    os.replace(tmp, state_path)


def load_neon_stage2_training_state(
    path: Any,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a resumable Stage-2 training state checkpoint."""

    payload = torch.load(Path(path), map_location=map_location)
    if int(payload.get("format_version", 0)) != 1:
        raise ValueError(f"unsupported NEON Stage-2 training state format in {path}.")
    required = {
        "state_dict",
        "optimizer_state_dict",
        "history",
        "best_epoch",
        "best_val_fit",
        "next_epoch",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"training state {path} is missing keys: {missing}.")
    return dict(payload)


# feature_collector(family, *, num_aleatory, generator) -> FrozenFGNOFeatureBatch
FeatureCollector = Callable[..., "FrozenFGNOFeatureBatch"]


def _reference_with_batch_dim(reference: torch.Tensor) -> torch.Tensor:
    """Return reference as ``[B, R, T, Nv, C]`` (add B=1 if given ``[R, T, Nv, C]``)."""
    if reference.ndim == 5:
        return reference
    if reference.ndim == 4:
        return reference.unsqueeze(0)
    raise ValueError(
        "family reference must have shape [R, T, Nv, C] or [B, R, T, Nv, C], "
        f"got {tuple(reference.shape)}."
    )


def _call_feature_collector(
    feature_collector: FeatureCollector,
    family: NEONFamilySample,
    *,
    num_aleatory: int,
    generator: Optional[torch.Generator],
    latent_bank_id: int,
) -> FrozenFGNOFeatureBatch:
    """Call a feature collector with latent-bank support and legacy fallback."""

    try:
        return feature_collector(
            family,
            num_aleatory=int(num_aleatory),
            generator=generator,
            latent_bank_id=int(latent_bank_id),
        )
    except TypeError as exc:
        # Older tests and third-party callers may still provide collectors with
        # the v1 signature. Preserve compatibility unless the TypeError came
        # from inside the collector body.
        if "latent_bank_id" not in str(exc):
            raise
        return feature_collector(family, num_aleatory=int(num_aleatory), generator=generator)


def _collate_frozen_batches(batches: Sequence[FrozenFGNOFeatureBatch]) -> FrozenFGNOFeatureBatch:
    if not batches:
        raise ValueError("cannot collate an empty feature batch list.")
    base_prediction = torch.cat([b.base_prediction for b in batches], dim=0)
    features = torch.cat([b.features for b in batches], dim=0)
    latents = [b.aleatory_latents for b in batches]
    if all(latent.ndim == 2 for latent in latents):
        aleatory_latents = torch.stack(latents, dim=1)  # [K, B, d_a]
    else:
        aleatory_latents = latents[0]
    canonical = [batch.canonical_mean_features for batch in batches]
    if all(value is None for value in canonical):
        canonical_mean_features = None
    elif any(value is None for value in canonical):
        raise ValueError("canonical mean features must be present for every collated family or none")
    else:
        canonical_mean_features = torch.cat([value for value in canonical if value is not None], dim=0)
    canonical_hashes = [batch.canonical_latent_hash for batch in batches]
    if all(value is None for value in canonical_hashes):
        canonical_latent_hash = None
    elif any(value is None for value in canonical_hashes):
        raise ValueError("canonical latent hashes must be present for every collated family or none")
    else:
        canonical_latent_hash = "|".join(str(value) for value in canonical_hashes)
    return FrozenFGNOFeatureBatch(
        base_prediction=base_prediction,
        features=features,
        aleatory_latents=aleatory_latents,
        canonical_mean_features=canonical_mean_features,
        canonical_latent_hash=canonical_latent_hash,
    )


def _collate_references(families: Sequence[NEONFamilySample]) -> torch.Tensor:
    refs = [_reference_with_batch_dim(family.reference) for family in families]
    return torch.cat(refs, dim=0)


def _collate_optional_score_weights(families: Sequence[NEONFamilySample]) -> Optional[torch.Tensor]:
    weights = [family.weights for family in families]
    if all(weight is None for weight in weights):
        return None
    if any(weight is None for weight in weights):
        raise ValueError("either all families in a mini-batch must provide weights or none.")
    stacked = []
    for weight in weights:
        assert weight is not None
        if weight.ndim in {3, 4}:
            stacked.append(weight.unsqueeze(0) if weight.ndim == 3 else weight)
        else:
            raise ValueError(
                "family weights must have shape [T, Nv, C] or [B, T, Nv, C], "
                f"got {tuple(weight.shape)}."
            )
    return torch.cat(stacked, dim=0)


def _collate_optional_geometry(families: Sequence[NEONFamilySample]) -> Optional[torch.Tensor]:
    geometries = [family.geometry for family in families]
    if all(geometry is None for geometry in geometries):
        return None
    if any(geometry is None for geometry in geometries):
        raise ValueError("either all families in a mini-batch must provide geometry or none.")
    stacked = []
    for geometry in geometries:
        assert geometry is not None
        if geometry.ndim == 2:
            stacked.append(geometry.unsqueeze(0))
        elif geometry.ndim == 3:
            stacked.append(geometry)
        else:
            raise ValueError(
                "family geometry must have shape [Nv, 2] or [B, Nv, 2], "
                f"got {tuple(geometry.shape)}."
            )
    return torch.cat(stacked, dim=0)


def _maybe_subsample_reference_members(
    reference: torch.Tensor,
    *,
    max_members: Optional[int],
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Randomly subsample HEC-RAS reference members along R for stochastic training."""

    if max_members is None:
        return reference, None
    n_keep = int(max_members)
    if n_keep < 1:
        raise ValueError(f"reference_member_subsample must be >= 1, got {max_members}.")
    R = int(reference.shape[1])
    if n_keep >= R:
        return reference, None
    perm = torch.randperm(R, generator=generator)
    idx = perm[:n_keep].to(device=reference.device)
    return reference.index_select(1, idx), idx


def _shuffled_family_indices(
    n_families: int,
    *,
    shuffle: bool,
    generator: Optional[torch.Generator],
) -> list[int]:
    if n_families < 1:
        return []
    if not shuffle:
        return list(range(n_families))
    return torch.randperm(int(n_families), generator=generator).tolist()


def _chunk_indices(indices: Sequence[int], chunk_size: int) -> Iterable[list[int]]:
    chunk = max(1, int(chunk_size))
    for start in range(0, len(indices), chunk):
        yield list(indices[start : start + chunk])


def _sample_latent_bank_id(
    *,
    latent_bank_count: int,
    generator: Optional[torch.Generator],
) -> int:
    count = max(1, int(latent_bank_count))
    if count == 1:
        return 0
    return int(torch.randint(count, (1,), generator=generator).item())


def _epoch_latent_bank_schedule(
    n_families: int,
    *,
    latent_bank_count: int,
    epoch: int,
    generator: Optional[torch.Generator],
) -> list[int]:
    """Build a reproducible bank schedule without consuming the training RNG.

    Isolating bank selection makes the next disk-cache key knowable for
    background prefetch while leaving epistemic-index and reference-member
    sampling unchanged by I/O timing.
    """

    count = max(1, int(latent_bank_count))
    if int(n_families) < 1:
        return []
    if count == 1:
        return [0] * int(n_families)
    base_seed = 0 if generator is None else int(generator.initial_seed())
    schedule_seed = (
        base_seed + 1_000_003 * (int(epoch) + 1) + 97_409 * count
    ) % (2**63 - 1)
    bank_generator = torch.Generator(device="cpu").manual_seed(schedule_seed)
    return torch.randint(
        count,
        (int(n_families),),
        generator=bank_generator,
    ).tolist()


def _feature_prefetch_depth(feature_collector: FeatureCollector) -> int:
    if not callable(getattr(feature_collector, "prefetch", None)):
        return 0
    return max(0, int(getattr(feature_collector, "prefetch_depth", 0)))


def _prefetch_feature(
    feature_collector: FeatureCollector,
    family: NEONFamilySample,
    *,
    num_aleatory: int,
    latent_bank_id: int,
) -> bool:
    prefetch = getattr(feature_collector, "prefetch", None)
    if not callable(prefetch):
        return False
    return bool(
        prefetch(
            family,
            num_aleatory=int(num_aleatory),
            latent_bank_id=int(latent_bank_id),
        )
    )


def _evaluate_neon_validation(
    *,
    module: NEONEpistemicCorrection,
    families: Sequence[NEONFamilySample],
    feature_collector: FeatureCollector,
    m: int,
    k: int,
    d_e: int,
    generator: Optional[torch.Generator],
    objective: str = "per_epistemic_fcrps",
    epistemic_chunk_size: int | None = None,
    physical_scale: float = 1.0,
    paired_rmse_rows: list[dict[str, float | str]] | None = None,
    fixed_epistemic_support: torch.Tensor | None = None,
    target_normalizer=None,
    reference_normalizer=None,
) -> tuple[float, dict[str, float]]:
    """Mean validation fit score and diagnostics (no gradients).

    Validation is intentionally unweighted by family/member bootstrap weights:
    it measures fit to the actual empirical reference law used for selection.
    """
    module.eval()
    total = 0.0
    count = 0
    diag_total: dict[str, float] = {}
    prefetch_depth = _feature_prefetch_depth(feature_collector)
    for prefetch_idx in range(min(prefetch_depth, len(families))):
        _prefetch_feature(
            feature_collector,
            families[prefetch_idx],
            num_aleatory=int(k),
            latent_bank_id=0,
        )
    with torch.no_grad():
        for family_idx, family in enumerate(families):
            batch = _call_feature_collector(
                feature_collector,
                family,
                num_aleatory=int(k),
                generator=generator,
                latent_bank_id=0,
            )
            next_prefetch_idx = family_idx + prefetch_depth
            if prefetch_depth and next_prefetch_idx < len(families):
                _prefetch_feature(
                    feature_collector,
                    families[next_prefetch_idx],
                    num_aleatory=int(k),
                    latent_bank_id=0,
                )
            z_e = (
                fixed_epistemic_support.to(
                    device=batch.features.device, dtype=batch.features.dtype
                )
                if fixed_epistemic_support is not None
                else sample_epistemic_indices(
                    int(m), int(d_e),
                    device=batch.features.device,
                    dtype=batch.features.dtype,
                    generator=generator,
                )
            )
            ref = _reference_with_batch_dim(family.reference)
            total_m = int(z_e.shape[0])
            chunk = int(epistemic_chunk_size) if epistemic_chunk_size else total_m
            chunk = max(1, min(chunk, total_m))
            fit_val = 0.0
            diag_val: dict[str, float] = {}
            mbar_train_chunks: list[torch.Tensor] = []
            mbar_prior_chunks: list[torch.Tensor] = []
            prediction_chunks: list[torch.Tensor] = []
            deterministic_prediction: torch.Tensor | None = None
            for start in range(0, total_m, chunk):
                z_chunk = z_e[start : start + chunk]
                scale = float(z_chunk.shape[0]) / float(total_m)
                out = module(
                    batch.base_prediction,
                    batch.features,
                    z_chunk,
                    node_coords=family.geometry,
                    canonical_mean_features=batch.canonical_mean_features,
                )
                fit = stage2_fit_score(
                    out.prediction,
                    ref.to(device=out.prediction.device, dtype=out.prediction.dtype),
                    weights=family.weights,
                    objective=objective,
                )
                fit_val += float(fit.item()) * scale
                prediction_chunks.append(out.prediction)
                if deterministic_prediction is None:
                    deterministic_prediction = (
                        batch.base_prediction.unsqueeze(1)
                        + out.deterministic_correction
                    )[:, 0]
                diag_c = cancellation_diagnostics(
                    trainable_correction=out.trainable_correction,
                    prior_correction=out.prior_correction,
                    alpha=module.alpha,
                )
                for key, value in diag_c.items():
                    diag_val[key] = diag_val.get(key, 0.0) + float(value) * scale
                mbar_prior_chunks.append(
                    (float(module.alpha) * out.prior_correction.detach().float()).mean(dim=2)
                )
                mbar_train_chunks.append(out.trainable_correction.detach().float().mean(dim=2))
            _mbar_prior = torch.cat(mbar_prior_chunks, dim=1)
            _mbar_total = torch.cat(mbar_train_chunks, dim=1) + _mbar_prior
            nested_prediction = torch.cat(prediction_chunks, dim=1)
            flat_prediction = nested_prediction.reshape(
                nested_prediction.shape[0],
                nested_prediction.shape[1] * nested_prediction.shape[2],
                *nested_prediction.shape[3:],
            )
            if deterministic_prediction is None:
                raise RuntimeError("validation produced no epistemic chunks.")
            reference_on_device = ref.to(
                device=flat_prediction.device, dtype=flat_prediction.dtype
            )
            if (target_normalizer is None) != (reference_normalizer is None):
                raise ValueError(
                    "physical validation requires both target and reference normalizers."
                )
            if target_normalizer is not None:
                target_normalizer.to(flat_prediction.device)
                reference_normalizer.to(flat_prediction.device)
                flat_metric = target_normalizer.inverse_transform(flat_prediction)
                base_metric = target_normalizer.inverse_transform(batch.base_prediction)
                deterministic_metric = target_normalizer.inverse_transform(
                    deterministic_prediction
                )
                reference_metric = reference_normalizer.inverse_transform(reference_on_device)
                metric_scale = 1.0
            else:
                flat_metric = flat_prediction
                base_metric = batch.base_prediction
                deterministic_metric = deterministic_prediction
                reference_metric = reference_on_device
                metric_scale = float(physical_scale)
            nested_metric = flat_metric.reshape_as(nested_prediction)
            mixture_score = (
                fixed_support_fair_crps_members
                if isinstance(module, PersistentDirichletParticleControl)
                else crossed_fair_crps_members
            )
            mixture_crps = mixture_score(
                nested_metric,
                reference_metric,
                weights=family.weights,
                reduction="mean",
            )
            base_crps = fair_crps_members(
                base_metric,
                reference_metric,
                weights=family.weights,
                reduction="mean",
            )
            deterministic_crps = fair_crps_members(
                deterministic_metric,
                reference_metric,
                weights=family.weights,
                reduction="mean",
            )
            base_rmse = base_rmse_from_reference(
                base_metric,
                reference_metric,
                weights=family.weights,
            )
            deterministic_rmse = base_rmse_from_reference(
                deterministic_metric,
                reference_metric,
                weights=family.weights,
            )
            stage2_rmse = base_rmse_from_reference(
                flat_metric,
                reference_metric,
                weights=family.weights,
            )
            diag_val["mixture_fair_crps_physical"] = float(mixture_crps.item()) * metric_scale
            diag_val["base_fair_crps_physical"] = float(base_crps.item()) * metric_scale
            diag_val["deterministic_head_fair_crps_physical"] = (
                float(deterministic_crps.item()) * metric_scale
            )
            diag_val["base_rmse_physical"] = float(base_rmse) * metric_scale
            diag_val["deterministic_head_rmse_physical"] = (
                float(deterministic_rmse) * metric_scale
            )
            diag_val["stage2_rmse_physical"] = float(stage2_rmse) * metric_scale
            diag_val["stage2_minus_base_rmse_physical"] = (
                float(stage2_rmse) - float(base_rmse)
            ) * metric_scale
            if paired_rmse_rows is not None:
                paired_rmse_rows.append(
                    {
                        "family_id": str(family.family_id),
                        "base_fair_crps_physical": float(base_crps.item())
                        * metric_scale,
                        "deterministic_head_fair_crps_physical": float(
                            deterministic_crps.item()
                        )
                        * metric_scale,
                        "mixture_fair_crps_physical": float(mixture_crps.item())
                        * metric_scale,
                        "base_rmse_physical": float(base_rmse) * metric_scale,
                        "deterministic_head_rmse_physical": float(
                            deterministic_rmse
                        )
                        * metric_scale,
                        "stage2_rmse_physical": float(stage2_rmse)
                        * metric_scale,
                        "stage2_minus_base_rmse_physical": (
                            float(stage2_rmse) - float(base_rmse)
                        )
                        * metric_scale,
                    }
                )
            diag_val.update(
                epistemic_variance_diagnostics(
                    mbar_total=_mbar_total,
                    mbar_prior_scaled=_mbar_prior,
                )
            )
            # Corrections are represented in normalized target units even when
            # forecast skill is evaluated after inverse transformation.  The
            # standard-deviation diagnostics therefore need the affine target
            # scale explicitly to remain comparable across normalizers.
            diag_val["total_epistemic_std_physical"] = (
                math.sqrt(max(diag_val["total_epistemic_variance"], 0.0))
                * float(physical_scale)
            )
            diag_val["prior_epistemic_std_physical"] = (
                math.sqrt(max(diag_val["prior_epistemic_variance"], 0.0))
                * float(physical_scale)
            )
            diag_val.update(
                prior_psi_floor_diagnostic(
                    module=module,
                    features=batch.features,
                    z_e=z_e,
                    node_coords=family.geometry,
                )
            )
            total += fit_val
            for key, value in diag_val.items():
                diag_total[key] = diag_total.get(key, 0.0) + value
            count += 1
    denom = max(count, 1)
    return total / denom, {key: value / denom for key, value in diag_total.items()}


def build_neon_stage2_metadata(
    *,
    stage1_checkpoint_path: str,
    stage1_checkpoint_alias: str,
    normalizer_fingerprint: Mapping[str, Any],
    structural_dry_policy: Any,
    feature_source: str,
    dependency: str,
    d_a: int,
    d_e: int,
    k_train: int,
    m_train: int,
    k_eval: int,
    m_eval: int,
    alpha: Optional[float],
    prior_seed: Optional[int],
    loss_weights: Mapping[str, float],
    optimizer_settings: Mapping[str, Any],
    stage1_config_snapshot: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the plan-mandated structured Stage-2 checkpoint metadata.

    ``best_epoch`` and ``val_metrics`` are filled in by the training loop at
    save time; everything else is pinned here so a checkpoint is fully
    reproducible and auditable.
    """
    metadata: dict[str, Any] = {
        "stage1_checkpoint_path": str(stage1_checkpoint_path),
        "stage1_checkpoint_alias": str(stage1_checkpoint_alias),
        "stage1_config_snapshot": dict(stage1_config_snapshot) if stage1_config_snapshot else None,
        "normalizer_fingerprint": dict(normalizer_fingerprint),
        "structural_dry_policy": structural_dry_policy,
        "feature_source": str(feature_source),
        "dependency": str(dependency),
        "d_a": int(d_a),
        "d_e": int(d_e),
        "k_train": int(k_train),
        "m_train": int(m_train),
        "k_eval": int(k_eval),
        "m_eval": int(m_eval),
        "alpha": None if alpha is None else float(alpha),
        "prior_seed": None if prior_seed is None else int(prior_seed),
        "loss_weights": dict(loss_weights),
        "optimizer_settings": dict(optimizer_settings),
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def train_neon_stage2_epochs(
    *,
    module: NEONEpistemicCorrection,
    optimizer: torch.optim.Optimizer,
    train_families: Sequence[NEONFamilySample],
    val_families: Sequence[NEONFamilySample],
    feature_collector: FeatureCollector,
    n_epochs: int,
    m_train: int,
    k_train: int,
    d_e: int,
    loss_weights: Optional[NEONStage2LossWeights] = None,
    generator: Optional[torch.Generator] = None,
    grad_clip_norm: Optional[float] = None,
    zero_threshold: float = 0.0,
    checkpoint_path: Optional[Any] = None,
    checkpoint_metadata: Optional[Mapping[str, Any]] = None,
    stage1_model: Optional[nn.Module] = None,
    objective: str = "per_epistemic_fcrps",
    epistemic_chunk_size: Optional[int] = None,
    val_seed: Optional[int] = None,
    validation_interval: int = 1,
    bootstrap_config: Optional[Mapping[str, Any]] = None,
    member_bootstrap_config: Optional[Mapping[str, Any]] = None,
    cancellation_config: Optional[Mapping[str, Any]] = None,
    family_batch_size: int = 1,
    effective_batch_size: int = 1,
    shuffle_families: bool = True,
    epistemic_resample: str = "epoch",
    latent_bank_count: int = 1,
    reference_member_subsample: Optional[int] = None,
    selection_min_retention: float = 0.0,
    selection_rmse_margin_m: float = 0.001,
    selection_metric: str = "mixture_crps",
    selection_enforce_rmse: bool = True,
    validation_physical_scale: float = 1.0,
    dirichlet_particle_control: PersistentDirichletParticleControl | None = None,
    validation_target_normalizer=None,
    validation_reference_normalizer=None,
    progress_reporter: Optional[NEONTrainingProgressReporter] = None,
    latest_checkpoint_path: Optional[Any] = None,
    start_epoch: int = 0,
    initial_history: Optional[Sequence[Mapping[str, Any]]] = None,
    initial_best_epoch: int = -1,
    initial_best_val_fit: float = math.inf,
    initial_particle_training_step: int = 0,
    initial_n_ineligible_epochs: int = 0,
) -> NEONTrainingResult:
    """Run the family-level Stage-2 epoch loop.

    For each epoch and training family: collect frozen FGNO base predictions +
    features (``feature_collector`` samples ``k_train`` aleatory members and
    runs the AR rollout), sample ``m_train`` epistemic indices, forward the
    EpiNet, compute per-epistemic fair CRPS plus penalties, and backprop only
    the trainable EpiNet parameters. After each epoch, evaluate mean validation
    fair CRPS and, if it improved, save the best checkpoint (with structured
    metadata + best-epoch bookkeeping).

    If ``stage1_model`` is provided, the optimizer is asserted not to own any
    Stage-1 parameter before training starts.
    """
    if stage1_model is not None:
        assert_optimizer_excludes_stage1(stage1_model=stage1_model, optimizer=optimizer)
    family_batch_size = max(1, int(family_batch_size))
    effective_batch_size = max(family_batch_size, int(effective_batch_size))
    latent_bank_count = max(1, int(latent_bank_count))
    validation_interval = int(validation_interval)
    if validation_interval < 1:
        raise ValueError(
            f"validation_interval must be >= 1, got {validation_interval}."
        )
    epistemic_resample = str(epistemic_resample).strip().lower()
    if epistemic_resample not in {"epoch", "effective_batch"}:
        raise ValueError(
            "epistemic_resample must be one of {'epoch', 'effective_batch'}, "
            f"got {epistemic_resample!r}."
        )

    history: list[dict[str, float]] = [dict(row) for row in (initial_history or [])]
    best_val = float(initial_best_val_fit)
    best_epoch = int(initial_best_epoch)
    n_ineligible_epochs = int(initial_n_ineligible_epochs)
    particle_training_step = int(initial_particle_training_step)
    first_epoch = max(0, int(start_epoch))
    total_epochs = int(n_epochs)
    if first_epoch > total_epochs:
        first_epoch = total_epochs

    if progress_reporter is not None:
        progress_reporter.training_start(
            n_epochs=total_epochs,
            n_train_families=len(train_families),
            n_val_families=len(val_families),
            family_batch_size=family_batch_size,
            effective_batch_size=effective_batch_size,
            m_train=int(m_train),
            k_train=int(k_train),
            d_e=int(d_e),
            objective=objective,
            latent_bank_count=latent_bank_count,
        )

    for epoch in range(first_epoch, total_epochs):
        epoch_start_time = time.time()
        module.train()
        fit_sum = 0.0
        total_sum = 0.0
        train_diag_sum: dict[str, float] = {}
        n_train_families = 0
        epoch_z_e = None
        epoch_bootstrap_weights = None
        ordered_indices = _shuffled_family_indices(
            len(train_families),
            shuffle=bool(shuffle_families),
            generator=generator,
        )
        bank_schedule = _epoch_latent_bank_schedule(
            len(ordered_indices),
            latent_bank_count=latent_bank_count,
            epoch=epoch,
            generator=generator,
        )
        prefetch_depth = _feature_prefetch_depth(feature_collector)
        for prefetch_position in range(min(prefetch_depth, len(ordered_indices))):
            _prefetch_feature(
                feature_collector,
                train_families[ordered_indices[prefetch_position]],
                num_aleatory=int(k_train),
                latent_bank_id=bank_schedule[prefetch_position],
            )
        train_family_position = 0
        all_family_ids = [fam.family_id for fam in train_families]
        total_effective_batches = max(
            1,
            math.ceil(float(len(ordered_indices)) / float(effective_batch_size)),
        )
        if progress_reporter is not None:
            progress_reporter.epoch_start(
                epoch=epoch,
                n_epochs=total_epochs,
                n_train_families=len(train_families),
            )
        for effective_batch_idx, effective_indices in enumerate(
            _chunk_indices(ordered_indices, effective_batch_size),
            start=1,
        ):
            optimizer.zero_grad(set_to_none=True)
            group_z_e = epoch_z_e
            group_bootstrap_weights = epoch_bootstrap_weights
            if epistemic_resample == "effective_batch":
                group_z_e = None
                group_bootstrap_weights = None
            for micro_indices in _chunk_indices(effective_indices, family_batch_size):
                micro_families = [train_families[idx] for idx in micro_indices]
                micro_batches = []
                for family in micro_families:
                    bank_id = bank_schedule[train_family_position]
                    micro_batches.append(
                        _call_feature_collector(
                            feature_collector,
                            family,
                            num_aleatory=int(k_train),
                            generator=generator,
                            latent_bank_id=bank_id,
                        )
                    )
                    next_prefetch_position = train_family_position + prefetch_depth
                    if (
                        prefetch_depth
                        and next_prefetch_position < len(ordered_indices)
                    ):
                        _prefetch_feature(
                            feature_collector,
                            train_families[ordered_indices[next_prefetch_position]],
                            num_aleatory=int(k_train),
                            latent_bank_id=bank_schedule[next_prefetch_position],
                        )
                    train_family_position += 1
                batch = _collate_frozen_batches(micro_batches)
                if group_z_e is None:
                    if dirichlet_particle_control is not None:
                        particle_indices = dirichlet_particle_control.training_indices(
                            particle_training_step, int(m_train)
                        )
                        particle_training_step += 1
                        group_z_e = dirichlet_particle_control.indices_to_epistemic(
                            particle_indices,
                            device=batch.features.device,
                            dtype=batch.features.dtype,
                        )
                        group_bootstrap_weights = (
                            dirichlet_particle_control.family_particle_weights(
                                all_family_ids,
                                particle_indices,
                                device=batch.features.device,
                                dtype=batch.features.dtype,
                            )
                        )
                    else:
                        group_z_e = sample_epistemic_indices(
                            int(m_train), int(d_e),
                            device=batch.features.device,
                            dtype=batch.features.dtype,
                            generator=generator,
                        )
                        boot = dict(bootstrap_config or {})
                        if bool(boot.get("enabled", False)) and objective == "per_epistemic_fcrps":
                            group_bootstrap_weights = epistemic_bootstrap_weights(
                                all_family_ids,
                                group_z_e,
                                seed=int(boot.get("seed", 0)),
                                distribution=str(boot.get("distribution", "tempered_exponential")),
                                temperature=float(boot.get("temperature", 0.5)),
                                normalize=str(boot.get("normalize", "per_epistemic_batch")),
                                min_weight=float(boot.get("min_weight", 0.05)),
                                max_weight=float(boot.get("max_weight", 5.0)),
                            )
                    if epistemic_resample == "epoch":
                        epoch_z_e = group_z_e
                        epoch_bootstrap_weights = group_bootstrap_weights
                z_e = group_z_e
                sample_weights = None
                if group_bootstrap_weights is not None:
                    sample_weights = group_bootstrap_weights[micro_indices]
                reference = _collate_references(micro_families).to(
                    device=batch.features.device, dtype=batch.features.dtype
                )
                reference, reference_member_indices = _maybe_subsample_reference_members(
                    reference,
                    max_members=reference_member_subsample,
                    generator=generator,
                )
                node_coords = _collate_optional_geometry(micro_families)
                member_weights = None
                member_boot = dict(member_bootstrap_config or {})
                if bool(member_boot.get("enabled", False)) and objective == "per_epistemic_fcrps":
                    member_weights = epistemic_member_bootstrap_weights(
                        [family.family_id for family in micro_families],
                        z_e,
                        int(reference.shape[1]),
                        member_indices=reference_member_indices,
                        seed=int(member_boot.get("seed", 1)),
                        temperature=float(member_boot.get("temperature", 1.0)),
                    )
                losses = neon_stage2_training_step(
                    module=module,
                    optimizer=optimizer,
                    base_prediction=batch.base_prediction,
                    features=batch.features,
                    reference=reference,
                    z_e=z_e,
                    weights=_collate_optional_score_weights(micro_families),
                    node_coords=node_coords,
                    edge_index=micro_families[0].edge_index,
                    edge_weights=micro_families[0].edge_weights,
                    zero_threshold=zero_threshold,
                    loss_weights=loss_weights,
                    grad_clip_norm=None,
                    objective=objective,
                    epistemic_chunk_size=epistemic_chunk_size,
                    sample_weights=sample_weights,
                    member_weights=member_weights,
                    canonical_mean_features=batch.canonical_mean_features,
                    zero_grad=False,
                    optimizer_step=False,
                    loss_scale=float(len(micro_indices)) / float(len(effective_indices)),
                )
                fit_sum += float(losses.fit.item()) * float(len(micro_indices))
                total_sum += float(losses.total.item()) * float(len(micro_indices))
                for key, value in (losses.diagnostics or {}).items():
                    train_diag_sum[key] = train_diag_sum.get(key, 0.0) + (
                        float(value) * float(len(micro_indices))
                    )
                n_train_families += len(micro_indices)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in module.parameters() if parameter.requires_grad],
                    float(grad_clip_norm),
                )
            optimizer.step()
            if progress_reporter is not None and progress_reporter.should_report_batch(
                effective_batch_idx,
                total_effective_batches,
            ):
                progress_reporter.train_progress(
                    epoch=epoch,
                    n_epochs=total_epochs,
                    effective_batch_idx=effective_batch_idx,
                    total_effective_batches=total_effective_batches,
                    train_families_done=n_train_families,
                    n_train_families=len(train_families),
                    running_train_fit=fit_sum / max(n_train_families, 1),
                    running_train_total=total_sum / max(n_train_families, 1),
                    epoch_elapsed_sec=time.time() - epoch_start_time,
                )

        should_validate = (
            (int(epoch) + 1) % validation_interval == 0
            or int(epoch) == total_epochs - 1
        )
        if not should_validate:
            row = {
                "epoch": int(epoch),
                "train_fit": fit_sum / max(n_train_families, 1),
                "train_total": total_sum / max(n_train_families, 1),
                "val_fit": float("nan"),
                "validation_ran": 0.0,
            }
            for key, value in train_diag_sum.items():
                row[f"train_{key}"] = value / max(n_train_families, 1)
            row["epoch_seconds"] = float(time.time() - epoch_start_time)
            history.append(row)
            if progress_reporter is not None:
                progress_reporter.epoch_end(
                    row=row,
                    best_epoch=best_epoch,
                    best_val_fit=best_val,
                    improved=False,
                    epoch_elapsed_sec=time.time() - epoch_start_time,
                )
            if latest_checkpoint_path is not None:
                latest_metadata = dict(checkpoint_metadata or {})
                latest_metadata["last_completed_epoch"] = int(epoch)
                latest_metadata["next_epoch"] = int(epoch) + 1
                latest_metadata["best_epoch"] = int(best_epoch)
                latest_metadata["best_val_fit"] = float(best_val)
                latest_metadata["selection_min_retention"] = float(
                    selection_min_retention
                )
                latest_metadata["n_ineligible_epochs"] = int(n_ineligible_epochs)
                save_neon_stage2_training_state(
                    latest_checkpoint_path,
                    module=module,
                    optimizer=optimizer,
                    metadata=latest_metadata,
                    history=history,
                    best_epoch=best_epoch,
                    best_val_fit=best_val,
                    next_epoch=int(epoch) + 1,
                    generator=generator,
                    particle_training_step=particle_training_step,
                    n_ineligible_epochs=n_ineligible_epochs,
                )
            continue

        # A fixed val_seed redraws the SAME validation z_e (and any collector
        # sampling) every epoch, so best-epoch selection compares like with
        # like instead of riding sampling noise.
        if progress_reporter is not None:
            progress_reporter.validation_start(
                epoch=epoch,
                n_epochs=total_epochs,
                n_val_families=len(val_families),
            )
        val_generator = generator
        if val_seed is not None:
            val_generator = torch.Generator()
            val_generator.manual_seed(int(val_seed))
        paired_rmse_rows: list[dict[str, float | str]] = []
        val_fit, val_diag = _evaluate_neon_validation(
            module=module,
            families=val_families,
            feature_collector=feature_collector,
            m=int(m_train),
            k=int(k_train),
            d_e=int(d_e),
            generator=val_generator,
            objective=objective,
            epistemic_chunk_size=epistemic_chunk_size,
            physical_scale=float(validation_physical_scale),
            paired_rmse_rows=paired_rmse_rows,
            fixed_epistemic_support=(
                None
                if dirichlet_particle_control is None
                else dirichlet_particle_control.eval_epistemic_indices()
            ),
            target_normalizer=validation_target_normalizer,
            reference_normalizer=validation_reference_normalizer,
        )
        if progress_reporter is not None:
            _atomic_write_json(
                Path(progress_reporter.output_dir)
                / f"validation_rmse_pairs_epoch_{epoch + 1:04d}.json",
                {"epoch": int(epoch), "pairs": paired_rmse_rows},
            )
        row = {
            "epoch": int(epoch),
            "train_fit": fit_sum / max(n_train_families, 1),
            "train_total": total_sum / max(n_train_families, 1),
            "val_fit": float(val_fit),
            "validation_ran": 1.0,
        }
        for key, value in train_diag_sum.items():
            row[f"train_{key}"] = value / max(n_train_families, 1)
        for key, value in val_diag.items():
            row[f"val_{key}"] = float(value)
        history.append(row)

        diag_cfg = dict(cancellation_config or {})
        if bool(diag_cfg.get("enabled", False)):
            warn_cos = float(diag_cfg.get("warn_cosine_below", -0.90))
            warn_cancel = float(diag_cfg.get("warn_cancellation_above", 0.80))
            cosine = float(row.get("val_train_prior_cosine", row.get("train_train_prior_cosine", 0.0)))
            cancel = float(row.get("val_cancellation_fraction", row.get("train_cancellation_fraction", 0.0)))
            if cosine < warn_cos and cancel > warn_cancel:
                row["cancellation_warning"] = 1.0

        retention = float(row.get("val_prior_retention_ratio", 1.0))
        retention_floor = float(selection_min_retention)
        retention_warning = retention_floor > 0.0 and retention < retention_floor
        if retention_warning:
            row["retention_warning"] = 1.0
            n_ineligible_epochs += 1
        rmse_delta = float(row.get("val_stage2_minus_base_rmse_physical", 0.0))
        rmse_margin = float(selection_rmse_margin_m)
        eligible = (not bool(selection_enforce_rmse)) or rmse_delta <= rmse_margin
        row["selection_eligible"] = 1.0 if eligible else 0.0
        row["selection_min_retention"] = retention_floor
        row["selection_rmse_margin_m"] = rmse_margin
        metric_name = str(selection_metric).strip().lower()
        if metric_name == "mixture_crps":
            selection_score = float(row.get("val_mixture_fair_crps_physical", val_fit))
        elif metric_name == "per_epistemic_fit":
            selection_score = float(val_fit)
        else:
            raise ValueError(f"unsupported selection_metric {selection_metric!r}.")
        row["selection_metric"] = metric_name
        row["selection_enforce_rmse"] = 1.0 if selection_enforce_rmse else 0.0
        row["selection_score_mixture_fair_crps_physical"] = selection_score
        improved = bool(eligible and selection_score < best_val)
        if improved:
            best_val = selection_score
            best_epoch = int(epoch)
            if checkpoint_path is not None:
                save_metadata = dict(checkpoint_metadata or {})
                save_metadata["best_epoch"] = int(epoch)
                save_metadata["val_metrics"] = {
                    "val_fit": float(val_fit),
                    "mixture_fair_crps_physical": selection_score,
                    "base_fair_crps_physical": float(
                        row.get("val_base_fair_crps_physical", float("nan"))
                    ),
                    "deterministic_head_fair_crps_physical": float(
                        row.get(
                            "val_deterministic_head_fair_crps_physical",
                            float("nan"),
                        )
                    ),
                    "base_rmse_physical": float(
                        row.get("val_base_rmse_physical", float("nan"))
                    ),
                    "deterministic_head_rmse_physical": float(
                        row.get(
                            "val_deterministic_head_rmse_physical",
                            float("nan"),
                        )
                    ),
                    "stage2_minus_base_rmse_physical": rmse_delta,
                }
                save_metadata["selection_min_retention"] = retention_floor
                save_metadata["selection_eligible"] = bool(eligible)
                save_neon_stage2_checkpoint(checkpoint_path, module, metadata=save_metadata)
                if progress_reporter is not None:
                    progress_reporter.checkpoint_saved(
                        epoch=epoch,
                        checkpoint_path=checkpoint_path,
                        val_fit=float(val_fit),
                    )
        row["epoch_seconds"] = float(time.time() - epoch_start_time)
        if progress_reporter is not None:
            progress_reporter.epoch_end(
                row=row,
                best_epoch=best_epoch,
                best_val_fit=best_val,
                improved=improved,
                epoch_elapsed_sec=time.time() - epoch_start_time,
            )
        if latest_checkpoint_path is not None:
            latest_metadata = dict(checkpoint_metadata or {})
            latest_metadata["last_completed_epoch"] = int(epoch)
            latest_metadata["next_epoch"] = int(epoch) + 1
            latest_metadata["best_epoch"] = int(best_epoch)
            latest_metadata["best_val_fit"] = float(best_val)
            latest_metadata["selection_min_retention"] = retention_floor
            latest_metadata["n_ineligible_epochs"] = int(n_ineligible_epochs)
            save_neon_stage2_training_state(
                latest_checkpoint_path,
                module=module,
                optimizer=optimizer,
                metadata=latest_metadata,
                history=history,
                best_epoch=best_epoch,
                best_val_fit=best_val,
                next_epoch=int(epoch) + 1,
                generator=generator,
                particle_training_step=particle_training_step,
                n_ineligible_epochs=n_ineligible_epochs,
            )

    if checkpoint_path is not None and best_epoch < 0 and history:
        # If every epoch violates RMSE non-inferiority, retain the last state as
        # an explicitly ineligible diagnostic checkpoint rather than silently
        # presenting it as the selected model.
        fallback_row = history[-1]
        best_epoch = int(fallback_row.get("epoch", total_epochs - 1))
        best_val = float(fallback_row.get("val_fit", math.inf))
        save_metadata = dict(checkpoint_metadata or {})
        save_metadata["best_epoch"] = int(best_epoch)
        save_metadata["val_metrics"] = {"val_fit": float(best_val)}
        save_metadata["selection_min_retention"] = float(selection_min_retention)
        save_metadata["selection_fallback_all_ineligible"] = True
        save_metadata["n_ineligible_epochs"] = int(n_ineligible_epochs)
        save_neon_stage2_checkpoint(checkpoint_path, module, metadata=save_metadata)

    if progress_reporter is not None:
        progress_reporter.training_end(
            best_epoch=best_epoch,
            best_val_fit=best_val,
            n_epochs=total_epochs,
        )
    return NEONTrainingResult(history=history, best_epoch=best_epoch, best_val_fit=best_val)


def build_epinet_from_config(
    config: Any,
    *,
    feature_channels: int,
    out_channels: int,
    hidden_channels: int = 64,
) -> NEONEpistemicCorrection:
    """Build a NEONEpistemicCorrection from a NEONStage2Config + feature width.

    ``feature_channels`` is the frozen FGNO feature width (e.g. the decoder
    pre-projection channel count) and must be probed from the model. ``alpha``
    is taken from the config when explicit, else a placeholder (0.1) that the
    caller should overwrite via prior-scale auto-calibration.
    """
    alpha = 0.1 if getattr(config, "alpha", None) is None else float(config.alpha)
    return NEONEpistemicCorrection(
        feature_channels=int(feature_channels),
        out_channels=int(out_channels),
        epistemic_dim=int(config.d_e),
        hidden_channels=int(hidden_channels),
        train_hidden_channels=int(getattr(config, "train_hidden_channels", hidden_channels)),
        prior_hidden_channels=int(getattr(config, "prior_hidden_channels", hidden_channels)),
        alpha=alpha,
        n_hidden_layers=int(getattr(config, "branch_layers", 2)),
        branch_activation=str(getattr(config, "branch_activation", "gelu")),
        branch_type=str(getattr(config, "branch_type", "projected")),
        concat_index=bool(getattr(config, "concat_index", True)),
        lead_time_dim=int(getattr(config, "lead_time_dim", 0)),
        za_dependent=str(getattr(config, "dependency", "za_dependent")).strip().lower()
        == "za_dependent",
        prior_rff_dim=int(getattr(config, "prior_rff_dim", 0)),
        prior_rff_lengthscale=float(getattr(config, "prior_rff_lengthscale", 0.25)),
        prior_rff_include_lead=bool(getattr(config, "prior_rff_include_lead", True)),
        epistemic_basis=str(getattr(config, "epistemic_basis", "identity")),
        epistemic_linear_terms=bool(getattr(config, "epistemic_linear_terms", True)),
        epistemic_quadratic_terms=int(
            getattr(config, "epistemic_quadratic_terms", 16)
        ),
        epistemic_basis_seed=int(getattr(config, "epistemic_basis_seed", 123)),
        deterministic_head=bool(getattr(config, "deterministic_head", False)),
        deterministic_head_feature=str(
            getattr(config, "deterministic_head_feature", "canonical_aleatory_mean")
        ),
    )
