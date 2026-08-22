from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorComparison:
    close: bool
    max_absolute_error: float
    mean_absolute_error: float


def compare_tensors(cached: Any, fresh: Any, atol: float, rtol: float) -> TensorComparison:
    """Compare torch/numpy-like tensors and retain auditable error magnitudes."""
    try:
        import torch

        if isinstance(cached, torch.Tensor) and isinstance(fresh, torch.Tensor):
            difference = (cached.float() - fresh.float()).abs()
            return TensorComparison(
                bool(torch.allclose(cached, fresh, atol=atol, rtol=rtol)),
                float(difference.max().item()) if difference.numel() else 0.0,
                float(difference.mean().item()) if difference.numel() else 0.0,
            )
    except ImportError:
        pass
    left = list(cached)
    right = list(fresh)
    if len(left) != len(right):
        return TensorComparison(False, float("inf"), float("inf"))
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right, strict=True)]
    close = all(
        diff <= atol + rtol * abs(float(b)) for diff, b in zip(differences, right, strict=True)
    )
    return TensorComparison(
        close,
        max(differences, default=0.0),
        sum(differences) / len(differences) if differences else 0.0,
    )
