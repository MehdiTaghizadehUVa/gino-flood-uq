import os
from pathlib import Path
from typing import Dict, Union

from ...utils import count_tensor_params
from .base_transforms import Transform, DictTransform
import torch

class Normalizer(Transform):
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def transform(self, data):
        return (data - self.mean)/(self.std + self.eps)
    
    def inverse_transform(self, data):
        return (data * (self.std + self.eps)) + self.mean

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std.
    """

    def __init__(self, mean=None, std=None, eps=1e-7, dim=None, mask=None):
        """
        mean : torch.tensor or None
            has to include batch-size as a dim of 1
            e.g. for tensors of shape ``(batch_size, channels, height, width)``,
            the mean over height and width should have shape ``(1, channels, 1, 1)``
        std : torch.tensor or None
        eps : float, default is 0
            for safe division by the std
        dim : int list, default is None
            if not None, dimensions of the data to reduce over to compute the mean and std.

            .. important::

                Has to include the batch-size (typically 0).
                For instance, to normalize data of shape ``(batch_size, channels, height, width)``
                along batch-size, height and width, pass ``dim=[0, 2, 3]``

        mask : torch.Tensor or None, default is None
            If not None, a tensor with the same size as a sample,
            with value 0 where the data should be ignored and 1 everywhere else

        Notes
        -----
        The resulting mean will have the same size as the input MINUS the specified dims.
        If you do not specify any dims, the mean and std will both be scalars.

        Returns
        -------
        UnitGaussianNormalizer instance
        """
        super().__init__()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("mask", mask)

        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0
        self.squared_mean = None
        self._running_mean64 = None
        self._running_m2_64 = None

    def _reduce_dims(self):
        if self.dim is None:
            return None
        if isinstance(self.dim, int):
            return (self.dim,)
        return tuple(self.dim)

    def _set_running_stats(self, mean64, m2_64, n_elements, *, out_dtype):
        self._running_mean64 = mean64.detach().clone()
        self._running_m2_64 = m2_64.detach().clone()
        self.n_elements = int(n_elements)
        self.mean = self._running_mean64.to(dtype=out_dtype)
        if self.n_elements > 1:
            var64 = torch.clamp(self._running_m2_64 / (self.n_elements - 1), min=0.0)
            self.std = torch.sqrt(var64).to(dtype=out_dtype)
        else:
            self.std = torch.zeros_like(self.mean, dtype=out_dtype)
        if self.n_elements > 0:
            pop_var64 = self._running_m2_64 / self.n_elements
            self.squared_mean = (pop_var64 + self._running_mean64**2).to(dtype=out_dtype)
        else:
            self.squared_mean = torch.zeros_like(self.mean, dtype=out_dtype)

    def _compute_batch_stats(self, data_batch):
        reduce_dims = self._reduce_dims()
        batch64 = data_batch.to(dtype=torch.float64)
        if reduce_dims is None:
            batch_sum = batch64.sum()
            batch_sq_sum = (batch64**2).sum()
        else:
            batch_sum = batch64.sum(dim=reduce_dims, keepdim=True)
            batch_sq_sum = (batch64**2).sum(dim=reduce_dims, keepdim=True)
        n_elements = count_tensor_params(data_batch, self.dim)
        batch_mean64 = batch_sum / n_elements
        batch_m2_64 = torch.clamp(batch_sq_sum - n_elements * (batch_mean64**2), min=0.0)
        return batch_mean64, batch_m2_64, int(n_elements)

    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count : count + batch_size]
            # print(samples.shape)
            # if batch_size == 1:
            #     samples = samples.unsqueeze(0)
            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim  # Note this includes batch-size
        if self.mask is None:
            batch_mean64, batch_m2_64, n_elements = self._compute_batch_stats(data_batch)
            self._set_running_stats(batch_mean64, batch_m2_64, n_elements, out_dtype=data_batch.dtype)
        else:
            batch_size = data_batch.shape[0]
            dim = [i - 1 for i in self.dim if i]
            shape = [s for i, s in enumerate(self.mask.shape) if i not in dim]
            self.n_elements = torch.count_nonzero(self.mask, dim=dim) * batch_size
            self.mean = torch.zeros(shape)
            self.std = torch.zeros(shape)
            self.squared_mean = torch.zeros(shape)
            data_batch[:, self.mask == 1] = 0
            self.mean[self.mask == 1] = (
                torch.sum(data_batch, dim=dim, keepdim=True) / self.n_elements
            )
            self.squared_mean = (
                torch.sum(data_batch**2, dim=dim, keepdim=True) / self.n_elements
            )
            self.std = torch.std(data_batch, dim=self.dim, keepdim=True)

    def incremental_update_mean_std(self, data_batch):
        if self.mask is None:
            batch_mean64, batch_m2_64, n_elements = self._compute_batch_stats(data_batch)
            if self._running_mean64 is None or self._running_m2_64 is None or not self.n_elements:
                self._set_running_stats(batch_mean64, batch_m2_64, n_elements, out_dtype=data_batch.dtype)
                return
            total = self.n_elements + n_elements
            delta = batch_mean64 - self._running_mean64
            merged_mean64 = self._running_mean64 + delta * (n_elements / total)
            merged_m2_64 = (
                self._running_m2_64
                + batch_m2_64
                + (delta**2) * (self.n_elements * n_elements / total)
            )
            self._set_running_stats(merged_mean64, merged_m2_64, total, out_dtype=data_batch.dtype)
            return
        else:
            dim = [i - 1 for i in self.dim if i]
            n_elements = torch.count_nonzero(self.mask, dim=dim) * data_batch.shape[0]
            data_batch[:, self.mask == 1] = 0

        self.mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.mean + torch.sum(data_batch, dim=dim, keepdim=True)
        )
        self.squared_mean = (1.0 / (self.n_elements + n_elements)) * (
            self.n_elements * self.squared_mean
            + torch.sum(data_batch**2, dim=dim, keepdim=True)
        )
        self.n_elements += n_elements

        # 1/(n_i + n_j) * (n_i * sum(x_i^2)/n_i + sum(x_j^2) - (n_i*sum(x_i)/n_i + sum(x_j))^2)
        # = 1/(n_i + n_j)  * (sum(x_i^2) + sum(x_j^2) - sum(x_i)^2 - 2sum(x_i)sum(x_j) - sum(x_j)^2))
        # multiply by (n_i + n_j) / (n_i + n_j + 1) for unbiased estimator
        self.std = torch.sqrt(self.squared_mean - self.mean**2) * self.n_elements / (self.n_elements - 1)

    def transform(self, x):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x):
        return x * (self.std + self.eps) + self.mean

    def forward(self, x):
        return self.transform(x)

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def state_dict(self):
        """Return a serializable state dict (mean, std, eps, dim, mask) for persistence."""
        if self.mean is None or self.std is None:
            return None
        return {
            "mean": self.mean.cpu().clone(),
            "std": self.std.cpu().clone(),
            "eps": self.eps,
            "dim": self.dim,
            "mask": self.mask.cpu().clone() if self.mask is not None else None,
        }

    @classmethod
    def from_state_dict(cls, state_dict, device=None):
        """
        Build a UnitGaussianNormalizer from a state dict saved by state_dict().
        If device is set, move mean/std/mask to that device.
        """
        if state_dict is None:
            return None
        mean = state_dict["mean"]
        std = state_dict["std"]
        mask = state_dict.get("mask")
        if device is not None:
            mean = mean.to(device)
            std = std.to(device)
            if mask is not None:
                mask = mask.to(device)
        return cls(
            mean=mean,
            std=std,
            eps=state_dict.get("eps", 1e-7),
            dim=state_dict.get("dim"),
            mask=mask,
        )

    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances


class DictUnitGaussianNormalizer(DictTransform):
    """DictUnitGaussianNormalizer composes
    DictTransform and UnitGaussianNormalizer to normalize different
    fields of a model output tensor to Gaussian distributions w/
    mean 0 and unit variance.

        Parameters
        ----------
        normalizer_dict : Dict[str, UnitGaussianNormalizer]
            dictionary of normalizers, keyed to fields
        input_mappings : Dict[slice]
            slices of input tensor to grab per field, must share keys with above
        return_mappings : Dict[slice]
            _description_
        """
    def __init__(self, 
                 normalizer_dict: Dict[str, UnitGaussianNormalizer],
                 input_mappings: Dict[str, slice],
                 return_mappings: Dict[str, slice]):
        assert set(normalizer_dict.keys()) == set(input_mappings.keys()), \
            "Error: normalizers and model input fields must be keyed identically"
        assert set(normalizer_dict.keys()) == set(return_mappings.keys()), \
            "Error: normalizers and model output fields must be keyed identically"

        super().__init__(transform_dict=normalizer_dict,
                         input_mappings=input_mappings,
                         return_mappings=return_mappings)
    
    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None, mask=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset

        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
        instances = {key: cls(dim=dim, mask=mask) for key in keys}
        for i, data_dict in enumerate(dataset):
            for key, sample in data_dict.items():
                if key in keys:
                    instances[key].partial_fit(sample.unsqueeze(0))
        return instances


# ---------------------------------------------------------------------------
# Persistence: save/load dict of UnitGaussianNormalizer to disk
# ---------------------------------------------------------------------------
NORMALIZER_STATE_VERSION = 1


def save_normalizers(
    normalizers: Dict[str, UnitGaussianNormalizer],
    path: Union[str, Path],
    keys_to_save: tuple = ("geometry", "static", "boundary", "target"),
) -> Path:
    """
    Save a dict of UnitGaussianNormalizers to a single file.
    Only the given keys are persisted; typically omit "dynamic" (alias of "target").

    Parameters
    ----------
    normalizers : dict[str, UnitGaussianNormalizer]
        Dict keyed by name (e.g. geometry, static, boundary, target).
    path : str | Path
        Output file path (e.g. .pt or .pth).
    keys_to_save : tuple
        Keys to include; default matches flood script (geometry, static, boundary, target).

    Returns
    -------
    Path
        Resolved path where state was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {}
    for key in keys_to_save:
        if key not in normalizers:
            continue
        norm = normalizers[key]
        sd = norm.state_dict() if hasattr(norm, "state_dict") else None
        if sd is not None:
            state[key] = sd
    payload = {"version": NORMALIZER_STATE_VERSION, "normalizers": state}
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path.resolve()


def load_normalizers(
    path: Union[str, Path],
    device: Union[str, torch.device, None] = None,
    keys_expected: tuple = ("geometry", "static", "boundary", "target"),
    dynamic_alias: str = "target",
) -> Dict[str, UnitGaussianNormalizer]:
    """
    Load a dict of UnitGaussianNormalizers from a file saved by save_normalizers.
    Rebuilds UnitGaussianNormalizer via from_state_dict for each key.
    Optionally adds an alias (e.g. "dynamic" -> "target") so both keys point to the same normalizer.

    Parameters
    ----------
    path : str | Path
        Input file path.
    device : str | torch.device | None
        If set, move mean/std to this device; default None (leave on CPU).
    keys_expected : tuple
        Keys that should be present; used for validation. Missing keys are skipped with a warning.
    dynamic_alias : str
        If not None, after loading, normalizers["dynamic"] = normalizers[this_key].
        Set to None to disable. Default "target" for flood script compatibility.

    Returns
    -------
    dict[str, UnitGaussianNormalizer]
        Dict keyed by name; includes alias if dynamic_alias is set.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalizer state file not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    version = payload.get("version", 0)
    if version != NORMALIZER_STATE_VERSION:
        raise ValueError(
            f"Normalizer state version mismatch: file has {version}, "
            f"expected {NORMALIZER_STATE_VERSION}. Refit and save again."
        )
    state = payload.get("normalizers", payload)
    out = {}
    for key in keys_expected:
        if key not in state:
            continue
        out[key] = UnitGaussianNormalizer.from_state_dict(state[key], device=device)
    if dynamic_alias and dynamic_alias in out:
        out["dynamic"] = out[dynamic_alias]
    return out
