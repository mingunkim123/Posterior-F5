"""Entropy-aware gates for mixing text and SSL reference conditions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _try_import_torch():
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _apply_gate(value, *, threshold: float, sharpness: float):
    """Apply ``sigmoid(sharpness * (threshold - value))`` to tensor / sequence / scalar."""

    torch = _try_import_torch()
    if torch is not None and torch.is_tensor(value):
        return torch.sigmoid(sharpness * (threshold - value))

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sigmoid(sharpness * (threshold - float(item))) for item in value]

    return _sigmoid(sharpness * (threshold - float(value)))


def entropy_to_text_weight(entropy, *, threshold: float = 1.0, sharpness: float = 4.0):
    """Return text-branch weight from posterior entropy.

    Low entropy means the ASR posterior is confident, so the returned text
    weight is high. High entropy pushes the hybrid toward SSL features.
    """

    return _apply_gate(entropy, threshold=threshold, sharpness=sharpness)


def blank_to_text_weight(blank_prob, *, threshold: float = 0.5, sharpness: float = 6.0):
    """Return text-branch weight from blank/filler mass.

    High blank mass usually means weak token evidence, so text weight decreases.
    """

    return _apply_gate(blank_prob, threshold=threshold, sharpness=sharpness)


def _multiply_gate(left, right):
    torch = _try_import_torch()
    if torch is not None and (torch.is_tensor(left) or torch.is_tensor(right)):
        return left * right

    if isinstance(left, list) and isinstance(right, list):
        return [a * b for a, b in zip(left, right)]
    if isinstance(left, list):
        return [a * right for a in left]
    if isinstance(right, list):
        return [left * b for b in right]
    return left * right


def combine_gate(entropy=None, blank_prob=None, *, entropy_threshold=1.0, blank_threshold=0.5):
    """Combine entropy and blank gates by multiplication."""

    if entropy is None and blank_prob is None:
        return None
    if entropy is None:
        return blank_to_text_weight(blank_prob, threshold=blank_threshold)
    if blank_prob is None:
        return entropy_to_text_weight(entropy, threshold=entropy_threshold)

    return _multiply_gate(
        entropy_to_text_weight(entropy, threshold=entropy_threshold),
        blank_to_text_weight(blank_prob, threshold=blank_threshold),
    )


def mix_conditions(soft_text, ssl, alpha):
    """Mix soft-text and SSL conditions with text weight alpha."""

    torch = _try_import_torch()
    if torch is None:
        raise RuntimeError("mix_conditions requires torch")

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=soft_text.device, dtype=soft_text.dtype)
    alpha = alpha.to(device=soft_text.device, dtype=soft_text.dtype)
    while alpha.ndim < soft_text.ndim:
        alpha = alpha.unsqueeze(-1)
    return alpha * soft_text + (1.0 - alpha) * ssl
