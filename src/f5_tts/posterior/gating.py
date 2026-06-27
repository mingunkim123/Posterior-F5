"""Entropy-aware gates for mixing text and SSL reference conditions."""

from __future__ import annotations

import math
from collections.abc import Sequence


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def entropy_to_text_weight(entropy, *, threshold: float = 1.0, sharpness: float = 4.0):
    """Return text-branch weight from posterior entropy.

    Low entropy means the ASR posterior is confident, so the returned text
    weight is high. High entropy pushes the hybrid toward SSL features.
    """

    try:
        import torch

        if torch.is_tensor(entropy):
            return torch.sigmoid(sharpness * (threshold - entropy))
    except ModuleNotFoundError:
        pass

    if isinstance(entropy, Sequence) and not isinstance(entropy, (str, bytes)):
        return [_sigmoid(sharpness * (threshold - float(value))) for value in entropy]
    return _sigmoid(sharpness * (threshold - float(entropy)))


def blank_to_text_weight(blank_prob, *, threshold: float = 0.5, sharpness: float = 6.0):
    """Return text-branch weight from blank/filler mass.

    High blank mass usually means weak token evidence, so text weight decreases.
    """

    try:
        import torch

        if torch.is_tensor(blank_prob):
            return torch.sigmoid(sharpness * (threshold - blank_prob))
    except ModuleNotFoundError:
        pass

    if isinstance(blank_prob, Sequence) and not isinstance(blank_prob, (str, bytes)):
        return [_sigmoid(sharpness * (threshold - float(value))) for value in blank_prob]
    return _sigmoid(sharpness * (threshold - float(blank_prob)))


def combine_gate(entropy=None, blank_prob=None, *, entropy_threshold=1.0, blank_threshold=0.5):
    """Combine entropy and blank gates by multiplication."""

    if entropy is None and blank_prob is None:
        return None
    if entropy is None:
        return blank_to_text_weight(blank_prob, threshold=blank_threshold)
    if blank_prob is None:
        return entropy_to_text_weight(entropy, threshold=entropy_threshold)

    entropy_weight = entropy_to_text_weight(entropy, threshold=entropy_threshold)
    blank_weight = blank_to_text_weight(blank_prob, threshold=blank_threshold)
    try:
        import torch

        if torch.is_tensor(entropy_weight) or torch.is_tensor(blank_weight):
            return entropy_weight * blank_weight
    except ModuleNotFoundError:
        pass

    if isinstance(entropy_weight, list) and isinstance(blank_weight, list):
        return [a * b for a, b in zip(entropy_weight, blank_weight)]
    if isinstance(entropy_weight, list):
        return [a * blank_weight for a in entropy_weight]
    if isinstance(blank_weight, list):
        return [entropy_weight * b for b in blank_weight]
    return entropy_weight * blank_weight


def mix_conditions(soft_text, ssl, alpha):
    """Mix soft-text and SSL conditions with text weight alpha."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("mix_conditions requires torch") from exc

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, device=soft_text.device, dtype=soft_text.dtype)
    alpha = alpha.to(device=soft_text.device, dtype=soft_text.dtype)
    while alpha.ndim < soft_text.ndim:
        alpha = alpha.unsqueeze(-1)
    return alpha * soft_text + (1.0 - alpha) * ssl
