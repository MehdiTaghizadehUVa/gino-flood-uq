"""Training helpers for NEON-aligned Stage-2 FGNO experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from neuralop.flood.neon import (
    NEONEpistemicCorrection,
    NEONStage2LossOutput,
    NEONStage2LossWeights,
    compute_stage2_loss,
    freeze_stage1_model,
    per_epistemic_fair_crps,
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
    edge_index: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    zero_threshold: float | torch.Tensor = 0.0,
    loss_weights: NEONStage2LossWeights | None = None,
    grad_clip_norm: float | None = None,
    objective: str = "per_epistemic_fcrps",
    epistemic_chunk_size: int | None = None,
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
    optimizer.zero_grad(set_to_none=True)
    total_m = int(z_e.shape[0])
    chunk = int(epistemic_chunk_size) if epistemic_chunk_size else total_m
    chunk = max(1, min(chunk, total_m))
    if chunk >= total_m:
        out = module(base_prediction, features, z_e)
        losses = compute_stage2_loss(
            prediction=out.prediction,
            reference=reference,
            correction=out.correction,
            module=module,
            weights=weights,
            edge_index=edge_index,
            edge_weights=edge_weights,
            zero_threshold=zero_threshold,
            loss_weights=loss_weights,
            objective=objective,
        )
        losses.total.backward()
    else:
        # Chunk the epistemic (M) axis with gradient accumulation to cap
        # activation memory. Per-epistemic fair CRPS and the correction
        # penalties are computed independently per z_e, so the mean over M
        # equals the size-weighted sum of per-chunk means (exact gradients).
        agg = {n: 0.0 for n in ("total", "fit", "rpf", "graph", "time", "pos", "mag")}
        for start in range(0, total_m, chunk):
            z_chunk = z_e[start : start + chunk]
            scale = float(z_chunk.shape[0]) / float(total_m)
            out = module(base_prediction, features, z_chunk)
            losses_c = compute_stage2_loss(
                prediction=out.prediction,
                reference=reference,
                correction=out.correction,
                module=module,
                weights=weights,
                edge_index=edge_index,
                edge_weights=edge_weights,
                zero_threshold=zero_threshold,
                loss_weights=loss_weights,
                objective=objective,
            )
            (losses_c.total * scale).backward()
            for n in agg:
                agg[n] += float(getattr(losses_c, n).item()) * scale
        losses = NEONStage2LossOutput(
            total=torch.tensor(agg["total"]),
            fit=torch.tensor(agg["fit"]),
            rpf=torch.tensor(agg["rpf"]),
            graph=torch.tensor(agg["graph"]),
            time=torch.tensor(agg["time"]),
            pos=torch.tensor(agg["pos"]),
            mag=torch.tensor(agg["mag"]),
        )
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(module.trainable_branch.parameters(), float(grad_clip_norm))
    optimizer.step()
    return losses


def neon_stage2_eval_forward(
    *,
    module: NEONEpistemicCorrection,
    base_prediction: torch.Tensor,
    features: torch.Tensor,
    z_e: torch.Tensor,
) -> torch.Tensor:
    """Inference helper returning nested corrected predictions only."""

    module.eval()
    with torch.no_grad():
        return module(base_prediction, features, z_e).prediction


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


@dataclass
class NEONTrainingResult:
    """Outcome of a Stage-2 training run."""

    history: list[dict[str, float]]
    best_epoch: int
    best_val_fit: float


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
) -> float:
    """Mean configured Stage-2 fit score over validation families (no gradients)."""
    module.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for family in families:
            batch = feature_collector(family, num_aleatory=int(k), generator=generator)
            z_e = sample_epistemic_indices(
                int(m), int(d_e),
                device=batch.features.device,
                dtype=batch.features.dtype,
                generator=generator,
            )
            ref = _reference_with_batch_dim(family.reference)
            total_m = int(z_e.shape[0])
            chunk = int(epistemic_chunk_size) if epistemic_chunk_size else total_m
            chunk = max(1, min(chunk, total_m))
            fit_val = 0.0
            for start in range(0, total_m, chunk):
                z_chunk = z_e[start : start + chunk]
                scale = float(z_chunk.shape[0]) / float(total_m)
                out = module(batch.base_prediction, batch.features, z_chunk)
                fit = stage2_fit_score(
                    out.prediction,
                    ref.to(device=out.prediction.device, dtype=out.prediction.dtype),
                    weights=family.weights,
                    objective=objective,
                )
                fit_val += float(fit.item()) * scale
            total += fit_val
            count += 1
    return total / max(count, 1)


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

    history: list[dict[str, float]] = []
    best_val = math.inf
    best_epoch = -1

    for epoch in range(int(n_epochs)):
        module.train()
        fit_sum = 0.0
        total_sum = 0.0
        n_batches = 0
        for family in train_families:
            batch = feature_collector(family, num_aleatory=int(k_train), generator=generator)
            z_e = sample_epistemic_indices(
                int(m_train), int(d_e),
                device=batch.features.device,
                dtype=batch.features.dtype,
                generator=generator,
            )
            losses = neon_stage2_training_step(
                module=module,
                optimizer=optimizer,
                base_prediction=batch.base_prediction,
                features=batch.features,
                reference=_reference_with_batch_dim(family.reference).to(
                    device=batch.features.device, dtype=batch.features.dtype
                ),
                z_e=z_e,
                weights=family.weights,
                edge_index=family.edge_index,
                edge_weights=family.edge_weights,
                zero_threshold=zero_threshold,
                loss_weights=loss_weights,
                grad_clip_norm=grad_clip_norm,
                objective=objective,
                epistemic_chunk_size=epistemic_chunk_size,
            )
            fit_sum += float(losses.fit.item())
            total_sum += float(losses.total.item())
            n_batches += 1

        # A fixed val_seed redraws the SAME validation z_e (and any collector
        # sampling) every epoch, so best-epoch selection compares like with
        # like instead of riding sampling noise.
        val_generator = generator
        if val_seed is not None:
            val_generator = torch.Generator()
            val_generator.manual_seed(int(val_seed))
        val_fit = _evaluate_neon_validation(
            module=module,
            families=val_families,
            feature_collector=feature_collector,
            m=int(m_train),
            k=int(k_train),
            d_e=int(d_e),
            generator=val_generator,
            objective=objective,
            epistemic_chunk_size=epistemic_chunk_size,
        )
        history.append(
            {
                "epoch": int(epoch),
                "train_fit": fit_sum / max(n_batches, 1),
                "train_total": total_sum / max(n_batches, 1),
                "val_fit": float(val_fit),
            }
        )

        if val_fit < best_val:
            best_val = float(val_fit)
            best_epoch = int(epoch)
            if checkpoint_path is not None:
                save_metadata = dict(checkpoint_metadata or {})
                save_metadata["best_epoch"] = int(epoch)
                save_metadata["val_metrics"] = {"val_fit": float(val_fit)}
                save_neon_stage2_checkpoint(checkpoint_path, module, metadata=save_metadata)

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
        alpha=alpha,
        lead_time_dim=int(getattr(config, "lead_time_dim", 0)),
        za_dependent=str(getattr(config, "dependency", "za_dependent")).strip().lower()
        == "za_dependent",
    )
