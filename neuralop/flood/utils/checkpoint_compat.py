"""Checkpoint <-> normalizer compatibility contract.

Resume-time training silently miscalibrates if the normalizer on disk has
drifted from the one the model was trained against. This module pins that
contract: every ``_save_checkpoint`` records the normalizer's fingerprint into
its payload, and every resume path asserts that the on-disk normalizer still
matches before the optimizer touches the model.

The contract is intentionally one-way: every fingerprint key the checkpoint
remembers must agree with the on-disk metadata. Adding *new* fingerprint keys
in the future is therefore backwards-compatible — old checkpoints just record
the subset of keys that existed at their save time.

This module is pure stdlib so it can be imported anywhere in the training
stack without pulling torch.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping


#: Subset of normalizer metadata that defines model-normalizer compatibility.
#: Keys here are the ones whose value *should* uniquely determine the fit, in
#: the sense that "if any of these changed, the model and normalizer no longer
#: agree on how to transform data". Adding a key is safe (one-way contract);
#: removing one is a breaking change that requires bumping a saved-format flag.
NORMALIZER_FINGERPRINT_KEYS: tuple[str, ...] = (
    "split_fingerprint",
    "boundary_spec_fingerprint",
    "hdf_paths_fingerprint",
    "split_sample_count",
)


class NormalizerCheckpointMismatchError(RuntimeError):
    """Raised when a resumed checkpoint and the on-disk normalizer disagree.

    Carries ``changed_keys`` (values differ) and ``missing_keys`` (key absent
    from the on-disk metadata) separately so callers can decide what to
    surface — changes are silent calibration drift; missing keys typically
    mean someone replaced or truncated the artifact entirely.
    """

    def __init__(
        self,
        message: str,
        changed_keys: tuple[str, ...] = (),
        missing_keys: tuple[str, ...] = (),
    ) -> None:
        super().__init__(message)
        self.changed_keys = changed_keys
        self.missing_keys = missing_keys


def normalizer_fingerprint(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return the fingerprint subset of ``metadata``.

    Unknown / future-extra keys in ``metadata`` are dropped. Missing keys are
    simply omitted from the result; the contract is "what we know about goes
    in", not "every key must be present".
    """
    return {key: metadata[key] for key in NORMALIZER_FINGERPRINT_KEYS if key in metadata}


def _diff_fingerprint(
    checkpoint_fp: Mapping[str, Any],
    on_disk_meta: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(changed_keys, missing_keys)`` ordered by NORMALIZER_FINGERPRINT_KEYS."""
    changed: list[str] = []
    missing: list[str] = []
    for key in NORMALIZER_FINGERPRINT_KEYS:
        if key not in checkpoint_fp:
            continue
        if key not in on_disk_meta:
            missing.append(key)
        elif checkpoint_fp[key] != on_disk_meta[key]:
            changed.append(key)
    return tuple(changed), tuple(missing)


def assert_normalizer_matches_checkpoint(
    *,
    checkpoint_fingerprint: Mapping[str, Any],
    on_disk_metadata: Mapping[str, Any],
    logger: logging.Logger,
    strict: bool = True,
) -> None:
    """Verify that the on-disk normalizer still agrees with a checkpoint's recorded fingerprint.

    Behavior:

    - Empty ``checkpoint_fingerprint`` (e.g., legacy checkpoint saved before
      this contract existed) is always accepted silently. Adding the contract
      must not retroactively break older training runs.
    - If every recorded fingerprint key matches, the call returns ``None``.
    - On mismatch with ``strict=True`` (default): raise
      :class:`NormalizerCheckpointMismatchError` with the offending keys.
    - On mismatch with ``strict=False``: emit a WARNING via ``logger`` but
      return. Useful for eval-only contexts where the operator has already
      accepted the risk.
    """
    if not checkpoint_fingerprint:
        return

    changed, missing = _diff_fingerprint(checkpoint_fingerprint, on_disk_metadata)
    if not changed and not missing:
        return

    parts: list[str] = []
    if changed:
        parts.append(f"changed_keys={list(changed)}")
    if missing:
        parts.append(f"missing_keys={list(missing)}")
    message = (
        "On-disk normalizer no longer matches the checkpoint's recorded "
        f"fingerprint ({', '.join(parts)}). Continuing would resume training "
        "against a normalizer the model was not calibrated for."
    )

    if strict:
        raise NormalizerCheckpointMismatchError(
            message=message,
            changed_keys=changed,
            missing_keys=missing,
        )
    logger.warning(message)
