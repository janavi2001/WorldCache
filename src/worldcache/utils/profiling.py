from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def profiling_range(name: str) -> Iterator[None]:
    """Emit an NVTX range when CUDA PyTorch is available, otherwise act as a no-op."""
    pushed = False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
    except (ImportError, RuntimeError):
        pass
    try:
        yield
    finally:
        if pushed:
            torch.cuda.nvtx.range_pop()
