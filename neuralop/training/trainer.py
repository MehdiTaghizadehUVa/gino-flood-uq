from timeit import default_timer
from pathlib import Path
from typing import Union, Optional
import logging
import sys
import os

import torch
from torch.cuda import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Only import wandb and use if installed
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False

from neuralop.losses import LpLoss
from .determinism import (
    deterministic_seed_context,
    seed_dataloader_for_epoch,
    stable_seed_from_parts,
)
from .training_state import load_training_state, save_training_state

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


_TRAIN_SCHEDULER_MONITORS = {"avg_loss", "train_err"}


class Trainer:
    """
    A general Trainer class to train neural-operators on given datasets
    """
    def __init__(
        self,
        *,
        model: nn.Module,
        n_epochs: int,
        wandb_log: bool=False,
        device: str='cpu',
        mixed_precision: bool=False,
        data_processor: nn.Module=None,
        eval_interval: int=1,
        log_output: bool=False,
        use_distributed: bool=False,
        verbose: bool=False,
        logger: Optional[logging.Logger]=None,
        use_progress_bar: bool=True,
        scheduler_monitor: str="train_err",
        grad_accum_steps: int=1,
        deterministic_eval: bool=False,
        eval_seed: int | None=None,
        train_seed: int | None=None,
        early_stopping_enabled: bool=False,
        early_stopping_patience: int=20,
        early_stopping_min_delta: float=1e-4,
    ):
        """
        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is False
            whether to log results to wandb
        device : torch.device, or str 'cpu' or 'cuda'
        mixed_precision : bool, default is False
            whether to use torch.autocast to compute mixed precision
        data_processor : DataProcessor class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        eval_interval : int, default is 1
            how frequently to evaluate model and log training stats
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        logger : logging.Logger, optional
            if set, log messages go to this logger instead of print
        use_progress_bar : bool, default is True
            if True and verbose, show tqdm progress bar over training batches
        """

        self.model = model
        self.n_epochs = n_epochs
        # only log to wandb if a run is active
        self.wandb_log = False
        if wandb_available:
            self.wandb_log = (wandb_log and wandb.run is not None)
        self.eval_interval = eval_interval
        self.log_output = log_output
        self.verbose = verbose
        self.logger = logger
        self.use_progress_bar = use_progress_bar
        self.scheduler_monitor = str(scheduler_monitor).strip().lower()
        self.use_distributed = use_distributed
        self.device = device
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        self.deterministic_eval = bool(deterministic_eval)
        self.eval_seed = None if eval_seed is None else int(eval_seed)
        self.train_seed = int(torch.initial_seed()) if train_seed is None else int(train_seed)
        self.early_stopping_enabled = bool(early_stopping_enabled)
        self.early_stopping_patience = max(1, int(early_stopping_patience))
        self.early_stopping_min_delta = float(early_stopping_min_delta)
        # handle autocast device
        if isinstance(self.device, torch.device):
            self.autocast_device_type = self.device.type
        else:
            if "cuda" in self.device:
                self.autocast_device_type = "cuda"
            else:
                self.autocast_device_type = "cpu"
        self.mixed_precision = mixed_precision
        self.data_processor = data_processor
        self.scaler = amp.GradScaler(enabled=(self.mixed_precision and self.autocast_device_type == "cuda"))

        self._best_metric_value = float("inf")
        self._early_stopping_best = float("inf")
        self._early_stopping_bad_epochs = 0

        # Track starting epoch for checkpointing/resuming
        self.start_epoch = 0

    def train(
        self,
        train_loader,
        test_loaders,
        optimizer,
        scheduler,
        regularizer=None,
        training_loss=None,
        eval_losses=None,
        save_every: int=None,
        save_best: str=None,
        save_dir: Union[str, Path]="./ckpt",
        resume_from_dir: Union[str, Path]=None,
    ):
        """Trains the given model on the given dataset.

        If a device is provided, the model and data processor are loaded to device here. 

        Parameters
        -----------
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        scheduler: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        save_every: int, optional, default is None
            if provided, interval at which to save checkpoints
        save_best: str, optional, default is None
            if provided, key of metric f"{loader_name}_{loss_name}"
            to monitor and save model with best eval result
            Saves best checkpoint on eval_interval in addition to save_every.
        save_dir: str | Path, default "./ckpt"
            directory at which to save training states if
            save_every and/or save_best is provided
        resume_from_dir: str | Path, default None
            if provided, resumes training state (model, 
            optimizer, regularizer, scheduler) from state saved in
            `resume_from_dir`
        
        Returns
        -------
        all_metrics: dict
            dictionary keyed f"{loader_name}_{loss_name}"
            of metric results for last validation epoch across
            all test_loaders
            
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        if regularizer:
            self.regularizer = regularizer
        else:
            self.regularizer = None

        if training_loss is None:
            training_loss = LpLoss(d=2)
        
        if eval_losses is None:  # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)
        
        # accumulated wandb metrics
        self.wandb_epoch_metrics = None

        # attributes for checkpointing
        self.save_every = save_every
        self.save_best = save_best
        if resume_from_dir is not None:
            self.resume_state_from_dir(resume_from_dir)

        # Load model and data_processor to device
        self.model = self.model.to(self.device)

        if self.use_distributed and dist.is_initialized():
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            if isinstance(self.device, torch.device) and self.device.type == "cuda":
                self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank)
            else:
                self.model = DDP(self.model)

        if self.data_processor is not None:
            self.data_processor = self.data_processor.to(self.device)
        
        eval_metric_names = []
        for name in test_loaders.keys():
            for metric in eval_losses.keys():
                eval_metric_names.append(f"{name}_{metric}")

        # ensure save_best is a metric we collect
        if self.save_best is not None:
            assert self.save_best in eval_metric_names,\
                f"Error: expected a metric of the form <loader_name>_<metric>, got {save_best}"
        best_metric_value = float(self._best_metric_value)

        scheduler_uses_eval_metric = (
            isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            and self.scheduler_monitor not in _TRAIN_SCHEDULER_MONITORS
        )
        if scheduler_uses_eval_metric and self.scheduler_monitor not in eval_metric_names:
            raise ValueError(
                f"scheduler_monitor={self.scheduler_monitor!r} was not produced by validation. "
                f"Use one of {sorted(_TRAIN_SCHEDULER_MONITORS)} or one of {eval_metric_names}."
            )
        early_stopping_uses_eval_metric = (
            self.early_stopping_enabled
            and self.scheduler_monitor not in _TRAIN_SCHEDULER_MONITORS
        )
        if early_stopping_uses_eval_metric and self.scheduler_monitor not in eval_metric_names:
            raise ValueError(
                f"early stopping monitor={self.scheduler_monitor!r} was not produced by validation. "
                f"Use one of {sorted(_TRAIN_SCHEDULER_MONITORS)} or one of {eval_metric_names}."
            )

        if self.verbose:
            msg = (
                f"Training on {len(train_loader.dataset)} samples. "
                f"Testing on {[len(loader.dataset) for loader in test_loaders.values()]} samples "
                f"on resolutions {[name for name in test_loaders]}."
            )
            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
                sys.stdout.flush()
        
        epoch_metrics = dict()
        early_stopping_best = float(self._early_stopping_best)
        early_stopping_bad_epochs = int(self._early_stopping_bad_epochs)
        should_stop = False
        for epoch in range(self.start_epoch, self.n_epochs):
            save_best_now = False
            train_err, avg_loss, avg_lasso_loss, epoch_train_time =\
                  self.train_one_epoch(epoch, train_loader, training_loss)
            epoch_metrics = dict(
                train_err=train_err,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                epoch_train_time=epoch_train_time
            )
            
            if epoch % self.eval_interval == 0:
                # evaluate and gather metrics across each loader in test_loaders
                eval_metrics = self.evaluate_all(epoch=epoch,
                                                eval_losses=eval_losses,
                                                test_loaders=test_loaders)

                epoch_metrics.update(**eval_metrics)
                # save checkpoint if conditions are met
                if self.save_best is not None:
                    if eval_metrics[self.save_best] < best_metric_value:
                        best_metric_value = float(eval_metrics[self.save_best])
                        save_best_now = True

            if self.early_stopping_enabled and self.scheduler_monitor in epoch_metrics:
                monitor_value = float(epoch_metrics[self.scheduler_monitor])
                if monitor_value < early_stopping_best - self.early_stopping_min_delta:
                    early_stopping_best = monitor_value
                    early_stopping_bad_epochs = 0
                else:
                    early_stopping_bad_epochs += 1
                    if early_stopping_bad_epochs >= self.early_stopping_patience:
                        should_stop = True
                        msg = (
                            f"Early stopping at epoch {epoch}: {self.scheduler_monitor}="
                            f"{monitor_value:.6e}, best={early_stopping_best:.6e}, "
                            f"bad_epochs={early_stopping_bad_epochs}/"
                            f"{self.early_stopping_patience}."
                        )
                        if self.logger:
                            self.logger.info(msg)
                        elif self.verbose:
                            print(msg)
                            sys.stdout.flush()

            self._best_metric_value = best_metric_value
            self._early_stopping_best = early_stopping_best
            self._early_stopping_bad_epochs = early_stopping_bad_epochs

            if save_best_now:
                self.checkpoint(save_dir, save_name="best_model")

            if scheduler_uses_eval_metric and self.scheduler_monitor in epoch_metrics:
                self.scheduler.step(float(epoch_metrics[self.scheduler_monitor]))

            # Save last checkpoint on schedule, including the epoch that triggers early stopping.
            if self.save_every is not None:
                if epoch % self.save_every == 0:
                    self.checkpoint(save_dir, save_name="model")

            if should_stop:
                break

        return epoch_metrics

    def train_one_epoch(self, epoch, train_loader, training_loss):
        """train_one_epoch trains self.model on train_loader
        for one epoch and returns training metrics

        Parameters
        ----------
        epoch : int
            epoch number
        train_loader : torch.utils.data.DataLoader
            data loader of train examples
        test_loaders : dict
            dict of test torch.utils.data.DataLoader objects

        Returns
        -------
        all_errors
            dict of all eval metrics for the last epoch
        """
        self.on_epoch_start(epoch)
        avg_loss = 0
        avg_lasso_loss = 0
        self.model.train()
        if self.data_processor:
            self.data_processor.train()
        t1 = default_timer()
        train_err = 0.0
        train_err_weight = 0.0
        avg_loss_weight = 0.0

        # track number of training examples in batch
        self.n_samples = 0

        if self.use_distributed and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        elif self.train_seed is not None:
            seed_dataloader_for_epoch(train_loader, base_seed=self.train_seed, epoch=epoch)

        use_pbar = (
            self.verbose
            and self.use_progress_bar
            and tqdm is not None
            and (not self.use_distributed or (hasattr(dist, "get_rank") and dist.get_rank() == 0))
        )
        batch_iter = train_loader
        if use_pbar:
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{self.n_epochs}",
                unit="batch",
                leave=True,
                dynamic_ncols=True,
                ncols=100,
            )

        n_batches = len(train_loader)
        grad_accum = max(1, int(self.grad_accum_steps))
        oom_fallback_batches = 0
        if hasattr(self, "_epoch_oom_microbatch_fallbacks"):
            self._epoch_oom_microbatch_fallbacks = 0
        self._skip_internal_zero_grad = True
        self._current_accum_divisor = 1
        self.optimizer.zero_grad(set_to_none=True)

        try:
            for idx, sample in enumerate(batch_iter):
                did_oom_retry = False
                result = None
                is_last_batch = (idx + 1 == n_batches)
                if is_last_batch and ((idx + 1) % grad_accum != 0):
                    accum_divisor = (idx % grad_accum) + 1
                else:
                    accum_divisor = grad_accum
                self._current_accum_divisor = accum_divisor
                for attempt in range(2):
                    try:
                        if did_oom_retry and hasattr(self, "retry_batch_after_oom"):
                            result = self.retry_batch_after_oom(idx, sample, training_loss)
                        else:
                            result = self.train_one_batch(idx, sample, training_loss)
                        if isinstance(result, tuple) and len(result) == 2:
                            loss, metrics = result
                        else:
                            loss, metrics = result, {}

                        backward_done = bool(metrics.pop("_backward_done", False)) if isinstance(metrics, dict) else False
                        if not backward_done:
                            scaled_loss = loss / accum_divisor
                            if self.scaler.is_enabled():
                                self.scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()
                        break
                    except RuntimeError as exc:
                        can_retry = (
                            self._is_cuda_oom(exc)
                            and (not did_oom_retry)
                            and hasattr(self, "retry_batch_after_oom")
                        )
                        if not can_retry:
                            raise
                        did_oom_retry = True
                        oom_fallback_batches += 1
                        self._clear_cuda_oom_state()
                        continue

                should_step = ((idx + 1) % grad_accum == 0) or (idx + 1 == n_batches)
                if should_step:
                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # train_err: use relative L2 when provided, else training loss.
                # AR trainers can provide explicit totals/weights so rollout-step metrics remain true means.
                err_val = metrics.get("rel_l2", loss)
                with torch.no_grad():
                    if "_log_rel_l2_total" in metrics:
                        rel_total = metrics["_log_rel_l2_total"]
                        train_err += rel_total.item() if hasattr(rel_total, "item") else rel_total
                        train_err_weight += float(metrics.get("_log_rel_l2_weight", 1.0))
                    else:
                        train_err += err_val.item() if hasattr(err_val, "item") else err_val
                    if "_log_loss_total" in metrics:
                        loss_total = metrics["_log_loss_total"]
                        avg_loss += loss_total.item() if hasattr(loss_total, "item") else loss_total
                        avg_loss_weight += float(metrics.get("_log_loss_weight", 1.0))
                    else:
                        avg_loss += loss.item()
                    if self.regularizer:
                        avg_lasso_loss += self.regularizer.loss

                if use_pbar and hasattr(batch_iter, "set_postfix"):
                    err_scalar = err_val.item() if hasattr(err_val, "item") else err_val
                    loss_scalar = loss.item() if hasattr(loss, "item") else loss
                    batch_size = None
                    if isinstance(sample, dict):
                        if "y" in sample and torch.is_tensor(sample["y"]):
                            batch_size = sample["y"].shape[0]
                        elif "target" in sample and torch.is_tensor(sample["target"]):
                            batch_size = sample["target"].shape[0]
                    # Display per-sample batch error so pbar train_err is comparable to epoch-end train_err.
                    err_display = (err_scalar / batch_size) if (batch_size is not None and batch_size > 0) else err_scalar
                    # Display per-sample batch loss so pbar loss is comparable to epoch-end avg_loss.
                    loss_display = (loss_scalar / batch_size) if (batch_size is not None and batch_size > 0) else loss_scalar
                    batch_iter.set_postfix(
                        loss=f"{loss_display:.8f}",
                        train_err=f"{err_display:.6f}",
                        oom_retry=oom_fallback_batches,
                        refresh=False,
                    )
        finally:
            self._skip_internal_zero_grad = False
            self._current_accum_divisor = 1

        epoch_train_time = default_timer() - t1

        # train_err = mean per sample (same scale as eval); err_val must be sum over batch (loss or metrics["rel_l2"])
        if train_err_weight > 0:
            train_err /= train_err_weight
        elif self.n_samples > 0:
            train_err /= self.n_samples
        if avg_loss_weight > 0:
            avg_loss /= avg_loss_weight
        elif self.n_samples > 0:
            avg_loss /= self.n_samples
        if self.regularizer:
            avg_lasso_loss /= self.n_samples
        else:
            avg_lasso_loss = None

        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if self.scheduler_monitor in _TRAIN_SCHEDULER_MONITORS:
                monitored = avg_loss if self.scheduler_monitor == "avg_loss" else train_err
                self.scheduler.step(monitored)
        else:
            self.scheduler.step()
        
        lr = None
        for pg in self.optimizer.param_groups:
            lr = pg["lr"]
        if self.verbose and epoch % self.eval_interval == 0:
            self.log_training(
                epoch=epoch,
                time=epoch_train_time,
                avg_loss=avg_loss,
                train_err=train_err,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr
            )
            if oom_fallback_batches > 0:
                msg = f"Epoch {epoch}: adaptive OOM fallback used on {oom_fallback_batches} batch(es)."
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(msg)
                    sys.stdout.flush()
            microbatch_fallback_batches = int(getattr(self, "_epoch_oom_microbatch_fallbacks", 0))
            if microbatch_fallback_batches > 0:
                msg = (
                    f"Epoch {epoch}: adaptive microbatch fallback used on "
                    f"{microbatch_fallback_batches} batch(es)."
                )
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(msg)
                    sys.stdout.flush()

        if self.wandb_log and wandb_available and oom_fallback_batches > 0:
            wandb.log(
                data={"train_oom_fallback_batches": float(oom_fallback_batches)},
                step=epoch + 1,
                commit=False,
            )
        microbatch_fallback_batches = int(getattr(self, "_epoch_oom_microbatch_fallbacks", 0))
        if self.wandb_log and wandb_available and microbatch_fallback_batches > 0:
            wandb.log(
                data={"train_oom_fallback_microbatch_batches": float(microbatch_fallback_batches)},
                step=epoch + 1,
                commit=False,
            )

        return train_err, avg_loss, avg_lasso_loss, epoch_train_time

    def evaluate_all(self, epoch, eval_losses, test_loaders):
        # evaluate and gather metrics across each loader in test_loaders
        all_metrics = {}
        for loader_name, loader in test_loaders.items():
            loader_metrics = self.evaluate(
                eval_losses,
                loader,
                log_prefix=loader_name,
                epoch=epoch,
            )
            all_metrics.update(**loader_metrics)
        if self.verbose:
            self.log_eval(epoch=epoch,
                      eval_metrics=all_metrics)
        return all_metrics
    
    def evaluate(self, loss_dict, data_loader, log_prefix="", epoch=None):
        """Evaluates the model on a dictionary of losses

        Parameters
        ----------
        loss_dict : dict of functions
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary
        epoch : int | None
            current epoch. Used when logging both train and eval
            default None
        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """
        # Ensure model and data processor are loaded to the proper device

        self.model = self.model.to(self.device)
        if self.data_processor is not None and self.data_processor.device != self.device:
            self.data_processor = self.data_processor.to(self.device)
        
        self.model.eval()
        if self.data_processor:
            self.data_processor.eval()

        errors = {f"{log_prefix}_{loss_name}": 0 for loss_name in loss_dict.keys()}

        self.n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):
                return_output = False
                if idx == len(data_loader) - 1:
                    return_output = True
                eval_seed = self._seed_for_eval_batch(epoch=epoch, log_prefix=log_prefix, batch_idx=idx)
                with deterministic_seed_context(eval_seed):
                    eval_step_losses, outs = self.eval_one_batch(
                        sample, loss_dict, return_output=return_output
                    )
                batch_size = sample["y"].size(0)

                for loss_name, val_loss in eval_step_losses.items():
                    v = val_loss.item() if hasattr(val_loss, "item") else val_loss
                    # When loss has reduction='mean', val is mean over batch; weight by batch size so sum/n_samples = true mean
                    loss_fn = loss_dict.get(loss_name)
                    if hasattr(loss_fn, "reduction") and getattr(loss_fn, "reduction", None) == "mean":
                        errors[f"{log_prefix}_{loss_name}"] += v * batch_size
                    else:
                        errors[f"{log_prefix}_{loss_name}"] += v

        # n_samples is cumulative from eval_one_batch
        for key in errors.keys():
            if self.n_samples > 0:
                errors[key] /= self.n_samples

        # on last batch, log model outputs
        if self.log_output:
            errors[f"{log_prefix}_outputs"] = wandb.Image(outs)
        
        return errors

    def _seed_for_eval_batch(self, *, epoch: int | None, log_prefix: str, batch_idx: int) -> int | None:
        if not self.deterministic_eval or self.eval_seed is None:
            return None
        return stable_seed_from_parts(
            "trainer_eval",
            self.eval_seed,
            int(epoch) if epoch is not None else -1,
            str(log_prefix),
            int(batch_idx),
        )
    
    def on_epoch_start(self, epoch):
        """on_epoch_start runs at the beginning
        of each training epoch. This method is a stub
        that can be overwritten in more complex cases.

        Parameters
        ----------
        epoch : int
            index of epoch

        Returns
        -------
        None
        """
        self.epoch = epoch
        return None

    @staticmethod
    def _is_cuda_oom(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return (
            ("cuda out of memory" in msg)
            or ("out of memory" in msg)
            or ("cublas_status_alloc_failed" in msg)
        )

    def _clear_cuda_oom_state(self):
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def train_one_batch(self, idx, sample, training_loss):
        """Run one batch of input through model
           and return training loss on outputs

        Parameters
        ----------
        idx : int
            index of batch within train_loader
        sample : dict
            data dictionary holding one batch

        Returns
        -------
        loss: float | Tensor
            float value of training loss
        """

        if not getattr(self, "_skip_internal_zero_grad", False):
            self.optimizer.zero_grad(set_to_none=True)
        if self.regularizer:
            self.regularizer.reset()
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].shape[0]

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                out = self.model(**sample)
        else:
            out = self.model(**sample)
        
        if self.epoch == 0 and idx == 0 and self.verbose:
            print(f"Raw outputs of shape {out.shape}")

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)

        loss = 0.0

        if self.mixed_precision:
            with torch.autocast(device_type=self.autocast_device_type):
                loss += training_loss(out, **sample)
        else:
            loss += training_loss(out, **sample)

        if self.regularizer:
            loss += self.regularizer.loss
        
        return loss, {}

    def eval_one_batch(self,
                       sample: dict,
                       eval_losses: dict,
                       return_output: bool=False):
        """eval_one_batch runs inference on one batch
        and returns eval_losses for that batch.

        Parameters
        ----------
        sample : dict
            data batch dictionary
        eval_losses : dict
            dictionary of named eval metrics
        return_outputs : bool
            whether to return model outputs for plotting
            by default False
        Returns
        -------
        eval_step_losses : dict
            keyed "loss_name": step_loss_value for each loss name
        outputs: torch.Tensor | None
            optionally returns batch outputs
        """
        if self.data_processor is not None:
            sample = self.data_processor.preprocess(sample)
        else:
            # load data to device if no preprocessor exists
            sample = {
                k: v.to(self.device)
                for k, v in sample.items()
                if torch.is_tensor(v)
            }

        self.n_samples += sample["y"].size(0)

        out = self.model(**sample)

        if self.data_processor is not None:
            out, sample = self.data_processor.postprocess(out, sample)
        
        eval_step_losses = {}

        for loss_name, loss in eval_losses.items():
            val_loss = loss(out, **sample)
            eval_step_losses[loss_name] = val_loss
        
        if return_output:
            return eval_step_losses, out
        else:
            return eval_step_losses, None
    
    def log_training(self, 
            epoch:int,
            time: float,
            avg_loss: float,
            train_err: float,
            avg_lasso_loss: float=None,
            lr: float=None
            ):
        """Basic method to log results
        from a single training epoch. 
        

        Parameters
        ----------
        epoch: int
        time: float
            training time of epoch
        avg_loss: float
            average train_err per individual sample
        train_err: float
            train error for entire epoch
        avg_lasso_loss: float
            average lasso loss from regularizer, optional
        lr: float
            learning rate at current epoch
        """
        # accumulate info to log to wandb
        if self.wandb_log:
            values_to_log = dict(
                train_err=train_err,
                time=time,
                avg_loss=avg_loss,
                avg_lasso_loss=avg_lasso_loss,
                lr=lr)

        msg = (
            f"Epoch {epoch} | time={time:.2f}s | "
            f"avg_loss={avg_loss:.8f} | train_err={train_err:.6f}"
        )
        if avg_lasso_loss is not None:
            msg += f" | avg_lasso={avg_lasso_loss:.6f}"
        if lr is not None:
            msg += f" | lr={lr:.2e}"

        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
            sys.stdout.flush()

        if self.wandb_log:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=False)
    
    def log_eval(self,
                 epoch: int,
                 eval_metrics: dict):
        """log_eval logs outputs from evaluation
        on all test loaders to stdout and wandb

        Parameters
        ----------
        epoch : int
            current training epoch
        eval_metrics : dict
            metrics collected during evaluation
            keyed f"{test_loader_name}_{metric}" for each test_loader
       
        """
        values_to_log = {}
        msg = ""
        for metric, value in eval_metrics.items():
            if isinstance(value, float) or isinstance(value, torch.Tensor):
                v = value.item() if hasattr(value, "item") else value
                msg += f"{metric}={v:.3e}, "
            if self.wandb_log:
                values_to_log[metric] = value       
        
        msg = "Eval: " + msg[:-2]  # cut off last comma+space
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
            sys.stdout.flush()

        if self.wandb_log and wandb_available:
            wandb.log(data=values_to_log,
                      step=epoch+1,
                      commit=True)

    @staticmethod
    def _trainer_progress_path(save_dir: Union[str, Path]) -> Path:
        return Path(save_dir) / "trainer_progress.pt"

    def _load_trainer_progress_from_dir(self, save_dir: Union[str, Path]) -> None:
        progress_path = self._trainer_progress_path(save_dir)
        if not progress_path.exists():
            return
        try:
            progress = torch.load(progress_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trainer progress sidecar from {progress_path}."
            ) from exc
        if not isinstance(progress, dict):
            raise RuntimeError(f"Trainer progress sidecar is not a dictionary: {progress_path}")

        self._best_metric_value = float(progress.get("best_metric_value", self._best_metric_value))
        self._early_stopping_best = float(
            progress.get("early_stopping_best", self._early_stopping_best)
        )
        self._early_stopping_bad_epochs = int(
            progress.get("early_stopping_bad_epochs", self._early_stopping_bad_epochs)
        )

    def _save_trainer_progress(self, save_dir: Union[str, Path]) -> None:
        is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
        if is_rank0:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
            torch.save(
                {
                    "best_metric_value": float(self._best_metric_value),
                    "early_stopping_best": float(self._early_stopping_best),
                    "early_stopping_bad_epochs": int(self._early_stopping_bad_epochs),
                    "scheduler_monitor": self.scheduler_monitor,
                    "save_best": self.save_best,
                },
                self._trainer_progress_path(save_dir),
            )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def resume_state_from_dir(self, save_dir):
        """
        Resume training from save_dir created by `neuralop.training.save_training_state`
        
        Params
        ------
        save_dir: Union[str, Path]
            directory in which training state is saved
            (see neuralop.training.training_state)
        """
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        # Prefer last checkpoint for resume; fall back to best if last is unavailable.
        if (save_dir / "model_state_dict.pt").exists():
            save_name = "model"
        elif (save_dir / "best_model_state_dict.pt").exists():
            save_name = "best_model"
        else:
            raise FileNotFoundError("Error: resume_from_dir expects a model\
                                        state dict named model.pt or best_model.pt.")
        # returns model, loads other modules if provided
        self.model, self.optimizer, self.scheduler, self.regularizer, resume_epoch =\
            load_training_state(save_dir=save_dir, save_name=save_name,
                                                model=self.model,
                                                optimizer=self.optimizer,
                                                regularizer=self.regularizer,
                                                scheduler=self.scheduler,
                                                map_location={'cpu': self.device},
                                                restore_rng_state_on_load=True)

        if resume_epoch is not None:
            next_epoch = int(resume_epoch) + 1
            if next_epoch > self.start_epoch:
                self.start_epoch = next_epoch
                self._load_trainer_progress_from_dir(save_dir)
                if self.verbose:
                    if self.logger:
                        self.logger.info("Trainer resuming from epoch %s", next_epoch)
                    else:
                        print(f"Trainer resuming from epoch {next_epoch}")


    def checkpoint(self, save_dir, save_name: Optional[str]=None):
        """checkpoint saves current training state
        to a directory for resuming later. Only saves 
        training state on the first GPU. 
        See neuralop.training.training_state

        Parameters
        ----------
        save_dir : str | Path
            directory in which to save training state
        save_name : str, optional
            checkpoint prefix. Defaults to "best_model" if save_best is configured,
            otherwise "model".
        """
        is_rank0 = (not dist.is_initialized()) or (dist.get_rank() == 0)
        if save_name is None:
            if self.save_best is not None:
                save_name = "best_model"
            else:
                save_name = "model"
        save_training_state(save_dir=save_dir, 
                            save_name=save_name,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            regularizer=self.regularizer,
                            epoch=self.epoch
                            )
        self._save_trainer_progress(save_dir)
        if is_rank0 and self.verbose:
            if self.logger:
                self.logger.info("Saved training state to %s", save_dir)
            else:
                print(f"[Rank 0]: saved training state to {save_dir}")
