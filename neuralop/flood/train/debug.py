"""Training debug and sanity-check helpers."""

from __future__ import annotations

import copy
import warnings

import torch

_DETERMINISTIC_ERROR_FRAGMENT = "does not have a deterministic implementation"


def _deterministic_guard_enabled() -> bool:
    fn = getattr(torch, "are_deterministic_algorithms_enabled", None)
    if fn is None:
        return False
    return bool(fn())


def _deterministic_warn_only_enabled() -> bool:
    fn = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    if fn is None:
        return False
    return bool(fn())


def _set_deterministic_algorithms(enabled: bool, warn_only: bool = False) -> None:
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    except TypeError:
        torch.use_deterministic_algorithms(enabled)


def _is_deterministic_runtime_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    return _DETERMINISTIC_ERROR_FRAGMENT in msg and "deterministic" in msg.lower()


def _capture_debug_retry_state(trainer, optimizer=None):
    model_state = copy.deepcopy(trainer.model.state_dict())
    optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore():
        trainer.model.load_state_dict(model_state)
        if optimizer is not None and optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            optimizer.zero_grad(set_to_none=True)
        else:
            for param in trainer.model.parameters():
                param.grad = None
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    return restore


def _run_debug_with_deterministic_fallback(fn, *, context: str, restore_state=None):
    try:
        return fn()
    except RuntimeError as exc:
        if not (_deterministic_guard_enabled() and _is_deterministic_runtime_error(exc)):
            raise
        warnings.warn(
            f"[{context}] Encountered a non-deterministic CUDA op while deterministic algorithms were enabled. "
            "Retrying this debug-only check once with deterministic algorithms temporarily disabled.",
            UserWarning,
            stacklevel=2,
        )
        prev_enabled = _deterministic_guard_enabled()
        prev_warn_only = _deterministic_warn_only_enabled()
        if restore_state is not None:
            restore_state()
        _set_deterministic_algorithms(False, warn_only=False)
        try:
            return fn()
        finally:
            _set_deterministic_algorithms(prev_enabled, warn_only=prev_warn_only)


def verify_training_gradient_flow(trainer, train_loader, training_loss):
    """
    Verify that loss is differentiable and gradients flow to model parameters.

    If ``trainer.train_one_batch`` already performed backward internally (for example,
    truncated/adaptive AR paths), reuse those gradients instead of calling backward a
    second time.
    """
    batch = next(iter(train_loader))
    restore_state = _capture_debug_retry_state(trainer, getattr(trainer, "optimizer", None))

    def _run_once():
        trainer.model.train()
        if trainer.data_processor is not None:
            trainer.data_processor.train()
        if getattr(trainer, "optimizer", None) is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        result = trainer.train_one_batch(0, batch, training_loss)
        if isinstance(result, tuple) and len(result) == 2:
            loss, metrics = result
        else:
            loss, metrics = result, {}

        if not isinstance(loss, torch.Tensor):
            raise AssertionError(f"train_one_batch must return a Tensor (or (loss, metrics)), got {type(loss)}")

        backward_done = bool(getattr(metrics, "get", lambda *_args, **_kwargs: False)("_backward_done", False))
        if not backward_done:
            if not loss.requires_grad:
                raise AssertionError("Loss does not require grad; check that model output is used in loss.")
            if loss.grad_fn is None:
                raise AssertionError("Loss has no grad_fn; graph may be detached.")
            loss.backward()

        total_norm = 0.0
        num_params_with_grad = 0
        for p in trainer.model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                num_params_with_grad += 1
        total_norm = total_norm ** 0.5

        if num_params_with_grad == 0:
            raise AssertionError("No parameter received gradients; optimizer will not update the model.")
        if total_norm == 0.0:
            raise AssertionError("Total gradient norm is zero; loss may not depend on model parameters.")

        return loss.item(), total_norm, num_params_with_grad, backward_done

    loss_value, total_norm, num_params_with_grad, backward_done = _run_debug_with_deterministic_fallback(
        _run_once,
        context="Verify",
        restore_state=restore_state,
    )

    print(
        f"[Verify] Gradient flow OK: loss={loss_value:.6f}, grad_norm={total_norm:.6e}, "
        f"params_with_grad={num_params_with_grad}, backward_done={backward_done}"
    )
    return True


def overfit_sanity_check(trainer, train_loader, training_loss, optimizer, n_steps=15):
    """
    Overfit a single batch for n_steps. If optimization is correct, loss should decrease.

    Respects trainer-internal backward for adaptive/truncated AR paths.
    """
    batch = next(iter(train_loader))
    restore_state = _capture_debug_retry_state(trainer, optimizer)

    def _run_once():
        trainer.model.train()
        if trainer.data_processor is not None:
            trainer.data_processor.train()
        losses = []
        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            result = trainer.train_one_batch(0, batch, training_loss)
            if isinstance(result, tuple) and len(result) == 2:
                loss, metrics = result
            else:
                loss, metrics = result, {}
            backward_done = bool(getattr(metrics, "get", lambda *_args, **_kwargs: False)("_backward_done", False))
            if not backward_done:
                loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    losses = _run_debug_with_deterministic_fallback(
        _run_once,
        context="Overfit check",
        restore_state=restore_state,
    )

    if losses[-1] >= losses[0]:
        warnings.warn(
            f"[Overfit check] Loss did not decrease over {n_steps} steps "
            f"(start={losses[0]:.6f}, end={losses[-1]:.6f}). "
            "Check learning rate, loss scale, or data/model.",
            UserWarning,
            stacklevel=2,
        )
    else:
        print(f"[Overfit check] OK: loss decreased from {losses[0]:.6f} to {losses[-1]:.6f} over {n_steps} steps.")
    return losses
