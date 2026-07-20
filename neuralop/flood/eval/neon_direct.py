"""Law-matched direct-particle re-optimization on frozen FGNO features."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from neuralop.flood.neon import per_epistemic_fair_crps


@dataclass
class DirectFamilyBatch:
    """One frozen family used by the Phase-5 direct benchmark."""

    family_id: str
    base_prediction: torch.Tensor  # [K,T,Nv,C], including the fixed delta-0 mean head
    features: torch.Tensor  # [K,T,Nv,C_phi]
    reference: torch.Tensor  # [R,T,Nv,C]
    score_weights: torch.Tensor | None = None  # [T,Nv,C]


class BatchedDirectLastLayer(nn.Module):
    """Independent non-indexed linear correction heads fitted in parallel."""

    def __init__(self, num_particles: int, *, feature_channels: int, out_channels: int) -> None:
        super().__init__()
        if int(num_particles) < 1 or int(feature_channels) < 1 or int(out_channels) < 1:
            raise ValueError("particle, feature, and output counts must be positive.")
        self.num_particles = int(num_particles)
        self.feature_channels = int(feature_channels)
        self.out_channels = int(out_channels)
        self.weight = nn.Parameter(
            torch.zeros(self.num_particles, self.feature_channels, self.out_channels)
        )
        self.bias = nn.Parameter(torch.zeros(self.num_particles, self.out_channels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or int(features.shape[-1]) != self.feature_channels:
            raise ValueError(
                f"features must be [K,T,Nv,{self.feature_channels}], got {tuple(features.shape)}."
            )
        return (
            torch.einsum("ktnf,dfo->dktno", features, self.weight)
            + self.bias[:, None, None, None, :]
        )

    def copy_particle_from(self, source: "BatchedDirectLastLayer", *, source_particle: int) -> None:
        """Initialize every direct draw from one verified anchor particle."""

        if (
            self.feature_channels != source.feature_channels
            or self.out_channels != source.out_channels
        ):
            raise ValueError("source and destination direct heads have incompatible shapes.")
        index = int(source_particle)
        with torch.no_grad():
            self.weight.copy_(source.weight[index : index + 1].expand_as(self.weight))
            self.bias.copy_(source.bias[index : index + 1].expand_as(self.bias))


class _DirectMLP(nn.Module):
    def __init__(self, feature_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class BatchedDirectFullHead(nn.Module):
    """Independent broader correction heads for the eight-draw sensitivity check."""

    def __init__(
        self,
        num_particles: int,
        *,
        feature_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.num_particles = int(num_particles)
        self.heads = nn.ModuleList(
            [
                _DirectMLP(feature_channels, hidden_channels, out_channels)
                for _ in range(self.num_particles)
            ]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(features) for head in self.heads], dim=0)

    def copy_from_shared_anchor(self, source: "BatchedDirectFullHead", source_particle: int = 0) -> None:
        state = copy.deepcopy(source.heads[int(source_particle)].state_dict())
        for head in self.heads:
            head.load_state_dict(state)


def subset_direct_module(module: nn.Module, indices: Sequence[int]) -> nn.Module:
    """Clone selected independent particles without changing their predictions."""

    selected = [int(index) for index in indices]
    if not selected:
        raise ValueError("indices must be non-empty.")
    if min(selected) < 0 or max(selected) >= int(getattr(module, "num_particles", 0)):
        raise IndexError("direct-particle subset index is out of range.")
    parameter = next(module.parameters())
    if isinstance(module, BatchedDirectLastLayer):
        result: nn.Module = BatchedDirectLastLayer(
            len(selected),
            feature_channels=module.feature_channels,
            out_channels=module.out_channels,
        ).to(device=parameter.device, dtype=parameter.dtype)
        with torch.no_grad():
            result.weight.copy_(module.weight[selected])
            result.bias.copy_(module.bias[selected])
        return result
    if isinstance(module, BatchedDirectFullHead):
        first_linear = module.heads[0].network[0]
        last_linear = module.heads[0].network[-1]
        result = BatchedDirectFullHead(
            len(selected),
            feature_channels=int(first_linear.in_features),
            hidden_channels=int(first_linear.out_features),
            out_channels=int(last_linear.out_features),
        ).to(device=parameter.device, dtype=parameter.dtype)
        for destination, source_index in zip(result.heads, selected):
            destination.load_state_dict(copy.deepcopy(module.heads[source_index].state_dict()))
        return result
    raise TypeError(f"unsupported direct module type: {type(module).__name__}")


def direct_weighted_scores(
    module: nn.Module,
    family: DirectFamilyBatch,
    *,
    particle_weights: torch.Tensor,
    prior_slice: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return one weighted fair-CRPS risk contribution per direct particle."""

    correction = module(family.features)
    if correction.ndim != 5:
        raise ValueError("direct module output must be [D,K,T,Nv,C].")
    D = int(correction.shape[0])
    weights = torch.as_tensor(particle_weights, device=correction.device, dtype=correction.dtype)
    if weights.shape != (D,):
        raise ValueError(f"particle_weights must have shape {(D,)}, got {tuple(weights.shape)}.")
    base = family.base_prediction.to(device=correction.device, dtype=correction.dtype)
    prediction = base.unsqueeze(0) + correction
    if prior_slice is not None:
        prior = prior_slice.to(device=correction.device, dtype=correction.dtype)
        if prior.shape != prediction.shape:
            raise ValueError("prior_slice must match [D,K,T,Nv,C] direct predictions.")
        prediction = prediction + prior
    reference = family.reference.to(device=correction.device, dtype=correction.dtype)
    score_weights = (
        None
        if family.score_weights is None
        else family.score_weights.to(device=correction.device, dtype=correction.dtype)
    )
    scores = per_epistemic_fair_crps(
        prediction.unsqueeze(0),
        reference.unsqueeze(0),
        weights=score_weights,
        reduction="none",
    )[0]
    return scores * weights


def _mean_particle_l2(module: nn.Module) -> torch.Tensor:
    """Return the mean per-particle squared parameter norm.

    The batched direct module represents independent optimizations whose risks
    are averaged over the particle axis.  Its regularizer must use the same
    averaging convention; otherwise merely increasing the diagnostic draw
    count strengthens regularization and changes every particle optimum.
    """

    particles = int(getattr(module, "num_particles", 0))
    if particles < 1:
        raise ValueError("direct module must expose a positive num_particles value.")
    terms = [parameter.square().sum() for parameter in module.parameters()]
    if not terms:
        raise ValueError("direct module has no parameters to regularize.")
    return sum(terms) / float(particles)


@dataclass(frozen=True)
class DirectFitResult:
    history: tuple[dict[str, float], ...]
    initial_objective: float
    final_objective: float
    initial_gradient_norm: float
    final_gradient_norm: float
    gradient_reduction_ratio: float
    best_epoch: int
    best_loss: float


def _full_objective_audit(
    module: nn.Module,
    *,
    n_family: int,
    load_family: Callable[[int], DirectFamilyBatch],
    normalized_weights: torch.Tensor,
    prior_slice: Callable[[int], torch.Tensor] | None,
    regularization: float,
    prefetch_family: Callable[[int], Any] | None = None,
) -> tuple[float, float]:
    """Return exact full-risk value and gradient norm with bounded memory."""

    module.zero_grad(set_to_none=True)
    objective = 0.0
    order = list(range(int(n_family)))
    if order and prefetch_family is not None:
        prefetch_family(order[0])
    for position, index in enumerate(order):
        family = load_family(index)
        if prefetch_family is not None and position + 1 < len(order):
            prefetch_family(order[position + 1])
        particle_weight = normalized_weights[index].to(
            device=next(module.parameters()).device,
            dtype=next(module.parameters()).dtype,
        )
        prior = None if prior_slice is None else prior_slice(index)
        contribution = direct_weighted_scores(
            module, family, particle_weights=particle_weight, prior_slice=prior
        ).mean() / float(n_family)
        objective += float(contribution.detach())
        contribution.backward()
    if float(regularization) > 0.0:
        penalty = 0.5 * float(regularization) * _mean_particle_l2(module)
        objective += float(penalty.detach())
        penalty.backward()
    squared = [
        parameter.grad.detach().double().norm().square()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    gradient_norm = (
        float(torch.stack(squared).sum().sqrt()) if squared else 0.0
    )
    module.zero_grad(set_to_none=True)
    return objective, gradient_norm


def _full_objective_value(
    module: nn.Module,
    *,
    n_family: int,
    load_family: Callable[[int], DirectFamilyBatch],
    normalized_weights: torch.Tensor,
    prior_slice: Callable[[int], torch.Tensor] | None,
    regularization: float,
    prefetch_family: Callable[[int], Any] | None = None,
) -> float:
    """Evaluate the exact empirical objective without building gradients."""

    parameter = next(module.parameters())
    objective = 0.0
    with torch.no_grad():
        order = list(range(int(n_family)))
        if order and prefetch_family is not None:
            prefetch_family(order[0])
        for position, index in enumerate(order):
            family = load_family(index)
            if prefetch_family is not None and position + 1 < len(order):
                prefetch_family(order[position + 1])
            particle_weight = normalized_weights[index].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            prior = None if prior_slice is None else prior_slice(index)
            contribution = direct_weighted_scores(
                module,
                family,
                particle_weights=particle_weight,
                prior_slice=prior,
            ).mean() / float(n_family)
            objective += float(contribution)
        if float(regularization) > 0.0:
            objective += float(0.5 * float(regularization) * _mean_particle_l2(module))
    return objective


def audit_direct_objective(
    module: nn.Module,
    *,
    family_ids: Sequence[str],
    load_family: Callable[[int], DirectFamilyBatch],
    family_particle_weights: torch.Tensor,
    prior_slice: Callable[[int], torch.Tensor] | None = None,
    regularization: float = 0.0,
    prefetch_family: Callable[[int], Any] | None = None,
) -> tuple[float, float]:
    """Audit the exact normalized direct risk and its gradient residual."""

    n_family = len(family_ids)
    weights = torch.as_tensor(family_particle_weights).double()
    particles = int(getattr(module, "num_particles", weights.shape[1]))
    if weights.shape != (n_family, particles):
        raise ValueError(
            "family_particle_weights must be [families,particles], got "
            f"{tuple(weights.shape)}."
        )
    normalized = weights / weights.mean(dim=0, keepdim=True).clamp_min(1.0e-15)
    return _full_objective_audit(
        module,
        n_family=n_family,
        load_family=load_family,
        normalized_weights=normalized,
        prior_slice=prior_slice,
        regularization=float(regularization),
        prefetch_family=prefetch_family,
    )


def fit_batched_direct_particles(
    module: nn.Module,
    *,
    family_ids: Sequence[str],
    load_family: Callable[[int], DirectFamilyBatch],
    family_particle_weights: torch.Tensor,
    prior_slice: Callable[[int], torch.Tensor] | None = None,
    epochs: int = 30,
    learning_rate: float = 1.0e-3,
    weight_decay: float = 1.0e-4,
    rpf_l2: float = 0.0,
    shuffle_seed: int = 0,
    grad_clip_norm: float | None = None,
    objective_audit_interval: int = 1,
    prefetch_family: Callable[[int], Any] | None = None,
) -> DirectFitResult:
    """Fit independent direct predictors with a shared numerical protocol.

    Each optimizer step uses one complete family. This keeps the reference
    ensemble intact and permits disk-backed frozen-feature providers. The
    particle weight matrix is normalized once per particle over all fit
    families, so each column defines a proper empirical bootstrap risk.
    """

    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1.")
    if int(objective_audit_interval) < 1:
        raise ValueError("objective_audit_interval must be >= 1.")
    n_family = len(family_ids)
    weights = torch.as_tensor(family_particle_weights).double()
    num_particles = int(getattr(module, "num_particles", weights.shape[1]))
    if weights.shape != (n_family, num_particles):
        raise ValueError(
            "family_particle_weights must be [families,particles], got "
            f"{tuple(weights.shape)}."
        )
    if bool((weights < 0).any()) or not torch.isfinite(weights).all():
        raise ValueError("direct family weights must be finite and nonnegative.")
    normalized = weights / weights.mean(dim=0, keepdim=True).clamp_min(1.0e-15)
    # Use explicit L2 regularization so the optimized mathematical objective
    # and the post-fit residual audit are identical. Decoupled AdamW decay is
    # an update rule, not the stated R(beta) in the direct-particle objective.
    regularization = float(weight_decay) + float(rpf_l2)
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))
    initial_objective, initial_gradient_norm = _full_objective_audit(
        module,
        n_family=n_family,
        load_family=load_family,
        normalized_weights=normalized,
        prior_slice=prior_slice,
        regularization=regularization,
        prefetch_family=prefetch_family,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_epoch = -1
    best_state = None
    for epoch in range(int(epochs)):
        order = torch.randperm(n_family, generator=generator).tolist()
        total = 0.0
        if order and prefetch_family is not None:
            prefetch_family(int(order[0]))
        for position, index in enumerate(order):
            family = load_family(int(index))
            if prefetch_family is not None and position + 1 < len(order):
                prefetch_family(int(order[position + 1]))
            optimizer.zero_grad(set_to_none=True)
            particle_weight = normalized[index].to(
                device=next(module.parameters()).device,
                dtype=next(module.parameters()).dtype,
            )
            prior = None if prior_slice is None else prior_slice(int(index))
            scores = direct_weighted_scores(
                module,
                family,
                particle_weights=particle_weight,
                prior_slice=prior,
            )
            loss = scores.mean()
            if regularization > 0.0:
                loss = loss + 0.5 * regularization * _mean_particle_l2(module)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), float(grad_clip_norm))
            optimizer.step()
            total += float(loss.detach())
        online_loss = total / max(n_family, 1)
        # The online loss is accumulated at different parameter values and is
        # therefore not a valid model-selection criterion. Audit the complete
        # fixed-parameter objective at each epoch boundary instead.
        should_audit = (
            (epoch + 1) % int(objective_audit_interval) == 0
            or epoch == int(epochs) - 1
        )
        exact_objective = (
            _full_objective_value(
                module,
                n_family=n_family,
                load_family=load_family,
                normalized_weights=normalized,
                prior_slice=prior_slice,
                regularization=regularization,
                prefetch_family=prefetch_family,
            )
            if should_audit
            else math.nan
        )
        history.append(
            {
                "epoch": float(epoch),
                "online_fit_loss": online_loss,
                "exact_fit_objective": exact_objective,
            }
        )
        if should_audit and exact_objective < best_loss:
            best_loss = exact_objective
            best_epoch = epoch
            best_state = copy.deepcopy(module.state_dict())
    if best_state is not None:
        module.load_state_dict(best_state)

    final_objective, final_gradient_norm = _full_objective_audit(
        module,
        n_family=n_family,
        load_family=load_family,
        normalized_weights=normalized,
        prior_slice=prior_slice,
        regularization=regularization,
        prefetch_family=prefetch_family,
    )
    reduction = final_gradient_norm / max(initial_gradient_norm, 1.0e-15)
    return DirectFitResult(
        tuple(history),
        initial_objective,
        final_objective,
        initial_gradient_norm,
        final_gradient_norm,
        reduction,
        best_epoch,
        best_loss,
    )


def prior_linear_representability_error(
    features: torch.Tensor,
    prior_slice: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    eps: float = 1.0e-12,
    chunk_rows: int = 65_536,
) -> dict[str, object]:
    r"""Project one or more prior slices onto the linear direct-head class.

    ``prior_slice`` may be ``[K,T,Nv,C]`` or ``[D,K,T,Nv,C]``.  The weighted
    design Gram matrix is shared by all ``D`` particles, so sufficient
    statistics are accumulated in row chunks instead of materializing a
    double-precision ``[D,K,T,Nv,C]`` regression matrix.
    """

    x = torch.as_tensor(features)
    y = torch.as_tensor(prior_slice)
    if x.ndim != 4:
        raise ValueError("features must be [K,T,Nv,F].")
    if y.ndim == 4:
        y = y.unsqueeze(0)
    if y.ndim != 5 or x.shape[:-1] != y.shape[1:-1]:
        raise ValueError(
            "prior must be [K,T,Nv,C] or [D,K,T,Nv,C] matching features."
        )
    if int(chunk_rows) < 1:
        raise ValueError("chunk_rows must be positive.")

    draws, channels = int(y.shape[0]), int(y.shape[-1])
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = y.reshape(draws, -1, channels)
    if weights is None:
        row_weights = torch.ones(
            x_flat.shape[0], device=x.device, dtype=torch.float64
        )
    else:
        w = torch.as_tensor(weights, device=x.device)
        if w.shape != y.shape[2:]:
            raise ValueError("weights must be [T,Nv,C] matching each prior slice.")
        # The direct head shares one design across output channels.  Preserve
        # the established multichannel convention by averaging channel weights.
        row_weights = (
            w.double()
            .mean(dim=-1)
            .unsqueeze(0)
            .expand(x.shape[0], -1, -1)
            .reshape(-1)
        )

    parameters = int(x.shape[-1]) + 1
    gram = torch.zeros(
        parameters, parameters, device=x.device, dtype=torch.float64
    )
    rhs = torch.zeros(
        draws, parameters, channels, device=x.device, dtype=torch.float64
    )
    target_energy = torch.zeros(draws, device=x.device, dtype=torch.float64)
    for start in range(0, int(x_flat.shape[0]), int(chunk_rows)):
        stop = min(start + int(chunk_rows), int(x_flat.shape[0]))
        feature = x_flat[start:stop].double()
        design = torch.cat(
            [
                feature,
                torch.ones(
                    feature.shape[0], 1, device=feature.device, dtype=feature.dtype
                ),
            ],
            dim=1,
        )
        weight = row_weights[start:stop].clamp_min(0.0)
        target = y_flat[:, start:stop].double()
        gram.add_(design.T @ (design * weight.unsqueeze(1)))
        rhs.add_(torch.einsum("np,n,dnc->dpc", design, weight, target))
        target_energy.add_(
            torch.einsum("n,dnc->d", weight, target.square())
        )

    # The pseudoinverse gives the minimum-norm weighted least-squares solution
    # when cached features are rank deficient.  Compute residual energy from
    # sufficient statistics to avoid a second pass over multi-GB tensors.
    inverse = torch.linalg.pinv(gram, hermitian=True)
    solution = torch.einsum("pq,dqc->dpc", inverse, rhs)
    linear = (solution * rhs).sum(dim=(1, 2))
    quadratic = torch.einsum("dpc,pq,dqc->d", solution, gram, solution)
    residual_energy = (target_energy - 2.0 * linear + quadratic).clamp_min(0.0)
    # The sufficient-statistic identity subtracts nearly equal O(||y||^2)
    # terms for an exactly representable slice. Remove only round-off at the
    # scale of that identity; genuine projection error remains unchanged.
    roundoff = (
        64.0
        * torch.finfo(residual_energy.dtype).eps
        * target_energy.abs().clamp_min(1.0)
    )
    residual_energy = torch.where(
        residual_energy <= roundoff,
        torch.zeros_like(residual_energy),
        residual_energy,
    )
    relative = residual_energy / (target_energy + float(eps))
    relative_values = [float(value) for value in relative]
    energy_values = [float(value) for value in target_energy]
    residual_values = [float(value) for value in residual_energy]
    return {
        "particle_count": draws,
        "relative_squared_error": float(relative.mean()),
        "relative_squared_error_per_particle": relative_values,
        "weighted_prior_energy": float(target_energy.mean()),
        "weighted_prior_energy_per_particle": energy_values,
        "weighted_residual_energy": float(residual_energy.mean()),
        "weighted_residual_energy_per_particle": residual_values,
    }


def _family_epistemic_variance(
    prediction: torch.Tensor,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    # prediction [D,K,T,Nv,C]; epistemic particles are means over K.
    particle_mean = prediction.double().mean(dim=1)
    if particle_mean.shape[0] < 2:
        return particle_mean.new_tensor(0.0)
    variance = particle_mean.var(dim=0, unbiased=True)
    if weights is None:
        return variance.mean()
    w = torch.as_tensor(weights).double()
    if w.shape != variance.shape:
        raise ValueError("score weight shape must match [T,Nv,C] prediction fields.")
    return (variance * w).sum() / w.sum().clamp_min(1.0e-15)


def direct_prediction_scale_interval(
    predictions_by_family: Sequence[torch.Tensor],
    *,
    score_weights: Sequence[torch.Tensor | None] | None = None,
    replicates: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Pinned direct epistemic scale with a families-and-draws bootstrap."""

    if not predictions_by_family:
        raise ValueError("predictions_by_family must be non-empty.")
    predictions = [torch.as_tensor(value).double() for value in predictions_by_family]
    draws = int(predictions[0].shape[0])
    if draws < 2 or any(value.ndim != 5 or int(value.shape[0]) != draws for value in predictions):
        raise ValueError("each direct prediction must be [draws,K,T,Nv,C] with draws >= 2.")
    if score_weights is None:
        all_weights: list[torch.Tensor | None] = [None] * len(predictions)
    else:
        if len(score_weights) != len(predictions):
            raise ValueError("score_weights must align with prediction families.")
        all_weights = list(score_weights)

    def scale(family_indices: torch.Tensor, draw_indices: torch.Tensor) -> float:
        contributions = []
        for raw_index in family_indices.tolist():
            selected = predictions[raw_index].index_select(0, draw_indices)
            contributions.append(_family_epistemic_variance(selected, all_weights[raw_index]))
        return float(torch.stack(contributions).mean().clamp_min(0.0).sqrt())

    family_identity = torch.arange(len(predictions))
    draw_identity = torch.arange(draws)
    estimate = scale(family_identity, draw_identity)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    for _ in range(int(replicates)):
        family_index = torch.randint(len(predictions), (len(predictions),), generator=generator)
        draw_index = torch.randint(draws, (draws,), generator=generator)
        samples.append(scale(family_index, draw_index))
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    return {
        "estimate": estimate,
        "ci95_lower": float(torch.quantile(sample_tensor, 0.025)),
        "ci95_upper": float(torch.quantile(sample_tensor, 0.975)),
        "n_families": float(len(predictions)),
        "n_draws": float(draws),
        "bootstrap_replicates": float(replicates),
        "bootstrap_seed": float(seed),
    }


def direct_particle_mean_scale_interval(
    particle_means_by_family: Sequence[torch.Tensor],
    *,
    score_weights: Sequence[torch.Tensor | None] | None = None,
    replicates: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Memory-efficient direct scale from saved ``[D,T,Nv,C]`` member means."""

    if not particle_means_by_family:
        raise ValueError("particle_means_by_family must be non-empty.")
    means = [torch.as_tensor(value) for value in particle_means_by_family]
    draws = int(means[0].shape[0])
    if draws < 2 or any(value.ndim != 4 or int(value.shape[0]) != draws for value in means):
        raise ValueError("each particle mean must be [draws,T,Nv,C] with draws >= 2.")
    if score_weights is None:
        all_weights: list[torch.Tensor | None] = [None] * len(means)
    else:
        if len(score_weights) != len(means):
            raise ValueError("score_weights must align with families.")
        all_weights = list(score_weights)

    grams = particle_mean_grams(means, score_weights=all_weights)
    return gram_scale_interval(grams, replicates=replicates, seed=seed)


def matched_particle_displacement_interval(
    fitted_particle_means_by_family: Sequence[torch.Tensor],
    anchor_particle_means_by_family: Sequence[torch.Tensor],
    *,
    score_weights: Sequence[torch.Tensor | None] | None = None,
    replicates: int = 2_000,
    seed: int = 0,
) -> dict[str, float | list[float]]:
    """RMS functional displacement from each draw's matched uniform anchor.

    Inputs retain the paired particle axis as ``[D,T,Nv,C]``.  Confidence
    intervals resample validation families and particle draws, never cells.
    This is deliberately distinct from between-particle spread: it isolates
    how much the bootstrap weights move each fitted predictor away from its
    own law-matched uniform-risk optimum.
    """

    fitted = [torch.as_tensor(value).double() for value in fitted_particle_means_by_family]
    anchors = [torch.as_tensor(value).double() for value in anchor_particle_means_by_family]
    if not fitted or len(fitted) != len(anchors):
        raise ValueError("fitted and anchor particle means must be non-empty and aligned.")
    draws = int(fitted[0].shape[0])
    expected_tail = fitted[0].shape[1:]
    if draws < 1 or len(expected_tail) != 3:
        raise ValueError("particle means must have shape [draws,T,Nv,C].")
    if any(
        value.shape != anchor.shape
        or int(value.shape[0]) != draws
        or value.shape[1:] != expected_tail
        for value, anchor in zip(fitted, anchors)
    ):
        raise ValueError("all fitted/anchor fields must share [draws,T,Nv,C] shape.")
    if score_weights is None:
        all_weights: list[torch.Tensor | None] = [None] * len(fitted)
    else:
        if len(score_weights) != len(fitted):
            raise ValueError("score_weights must align with displacement families.")
        all_weights = list(score_weights)

    family_draw_mse = []
    for value, anchor, raw_weight in zip(fitted, anchors, all_weights):
        difference = value - anchor
        if raw_weight is None:
            family_draw_mse.append(difference.square().mean(dim=(1, 2, 3)))
            continue
        weight = torch.as_tensor(raw_weight).double()
        if weight.shape != difference.shape[1:]:
            raise ValueError("score weight shape must match [T,Nv,C] displacement fields.")
        if not torch.isfinite(weight).all() or bool((weight < 0).any()):
            raise ValueError("displacement score weights must be finite and nonnegative.")
        denominator = weight.sum()
        if float(denominator) <= 0.0:
            raise ValueError("displacement score weights must have positive mass.")
        family_draw_mse.append(
            (difference.square() * weight.unsqueeze(0)).sum(dim=(1, 2, 3))
            / denominator
        )
    mse = torch.stack(family_draw_mse, dim=0)  # [families,draws]

    def _rms(family_index: torch.Tensor, draw_index: torch.Tensor) -> float:
        selected = mse.index_select(0, family_index).index_select(1, draw_index)
        return float(selected.mean().clamp_min(0.0).sqrt())

    estimate = _rms(torch.arange(mse.shape[0]), torch.arange(draws))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    for _ in range(int(replicates)):
        family_index = torch.randint(
            mse.shape[0], (mse.shape[0],), generator=generator
        )
        draw_index = torch.randint(draws, (draws,), generator=generator)
        samples.append(_rms(family_index, draw_index))
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    per_draw_rms = mse.mean(dim=0).clamp_min(0.0).sqrt()
    return {
        "estimate": estimate,
        "ci95_lower": float(torch.quantile(sample_tensor, 0.025)),
        "ci95_upper": float(torch.quantile(sample_tensor, 0.975)),
        "n_families": float(mse.shape[0]),
        "n_draws": float(draws),
        "bootstrap_replicates": float(replicates),
        "bootstrap_seed": float(seed),
        "per_draw_rms": [float(value) for value in per_draw_rms],
    }


def particle_mean_grams(
    particle_means_by_family: Sequence[torch.Tensor],
    *,
    score_weights: Sequence[torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """Return weighted draw Gram matrices ``[families,D,D]``."""

    if not particle_means_by_family:
        raise ValueError("particle_means_by_family must be non-empty.")
    means = [torch.as_tensor(value) for value in particle_means_by_family]
    draws = int(means[0].shape[0])
    if draws < 2 or any(value.ndim != 4 or int(value.shape[0]) != draws for value in means):
        raise ValueError("each particle mean must be [draws,T,Nv,C] with draws >= 2.")
    if score_weights is None:
        all_weights: list[torch.Tensor | None] = [None] * len(means)
    else:
        if len(score_weights) != len(means):
            raise ValueError("score_weights must align with families.")
        all_weights = list(score_weights)
    grams = []
    for value, weight in zip(means, all_weights):
        flat = value.double().reshape(draws, -1)
        if weight is None:
            row_weight = torch.full(
                (flat.shape[1],), 1.0 / float(flat.shape[1]), dtype=torch.float64
            )
        else:
            raw_weight = torch.as_tensor(weight).double()
            if raw_weight.shape != value.shape[1:]:
                raise ValueError("score weight shape must match [T,Nv,C] fields.")
            row_weight = raw_weight.reshape(-1)
            row_weight = row_weight / row_weight.sum().clamp_min(1.0e-15)
        weighted = flat * row_weight.sqrt().unsqueeze(0)
        grams.append(weighted @ weighted.T)
    return torch.stack(grams, dim=0)


def gram_scale_interval(
    gram_tensor: torch.Tensor,
    *,
    replicates: int = 2_000,
    seed: int = 0,
) -> dict[str, float]:
    """Pinned scale and family/draw bootstrap CI from weighted draw Grams."""

    gram_tensor = torch.as_tensor(gram_tensor).double()
    if gram_tensor.ndim != 3 or gram_tensor.shape[1] != gram_tensor.shape[2]:
        raise ValueError("gram_tensor must be [families,draws,draws].")
    families, draws, _ = gram_tensor.shape
    if families < 1 or draws < 2:
        raise ValueError("Gram scale requires at least one family and two draws.")

    def scale(family_index: torch.Tensor, draw_index: torch.Tensor) -> float:
        selected = gram_tensor.index_select(0, family_index)
        diagonal = selected[:, draw_index, draw_index].sum(dim=1)
        pair = selected.index_select(1, draw_index).index_select(2, draw_index).sum(
            dim=(1, 2)
        )
        variance = (diagonal - pair / float(draws)) / float(draws - 1)
        return float(variance.mean().clamp_min(0.0).sqrt())

    estimate = scale(torch.arange(families), torch.arange(draws))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    samples = []
    for _ in range(int(replicates)):
        family_index = torch.randint(families, (families,), generator=generator)
        draw_index = torch.randint(draws, (draws,), generator=generator)
        samples.append(scale(family_index, draw_index))
    sample_tensor = torch.tensor(samples, dtype=torch.float64)
    return {
        "estimate": estimate,
        "ci95_lower": float(torch.quantile(sample_tensor, 0.025)),
        "ci95_upper": float(torch.quantile(sample_tensor, 0.975)),
        "n_families": float(families),
        "n_draws": float(draws),
        "bootstrap_replicates": float(replicates),
        "bootstrap_seed": float(seed),
    }
