"""Training debug and sanity-check helpers."""

from __future__ import annotations

import warnings

import torch

def verify_training_gradient_flow(trainer, train_loader, training_loss):
    """
    Verify that loss is differentiable and gradients flow to model parameters.
    Runs one forward + backward and checks loss.grad_fn and param.grad norms.
    """
    trainer.model.train()
    if trainer.data_processor is not None:
        trainer.data_processor.train()
    batch = next(iter(train_loader))
    result = trainer.train_one_batch(0, batch, training_loss)
    loss, _ = (result[0], result[1]) if isinstance(result, tuple) and len(result) == 2 else (result, {})

    if not isinstance(loss, torch.Tensor):
        raise AssertionError(f"train_one_batch must return a Tensor (or (loss, metrics)), got {type(loss)}")
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

    print(f"[Verify] Gradient flow OK: loss={loss.item():.6f}, grad_norm={total_norm:.6e}, params_with_grad={num_params_with_grad}")
    return True


def overfit_sanity_check(trainer, train_loader, training_loss, optimizer, n_steps=15):
    """
    Overfit a single batch for n_steps. If optimization is correct, loss should decrease.
    """
    trainer.model.train()
    if trainer.data_processor is not None:
        trainer.data_processor.train()
    batch = next(iter(train_loader))
    losses = []
    for step in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        result = trainer.train_one_batch(0, batch, training_loss)
        loss = result[0] if isinstance(result, tuple) and len(result) == 2 else result
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
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
