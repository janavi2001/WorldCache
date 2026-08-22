from __future__ import annotations

import threading
from typing import Any


def synchronize(device: str = "cuda:0") -> None:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except ImportError:
        return


def memory_stats(device: str = "cuda:0") -> tuple[int, int]:
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
    except ImportError:
        pass
    return 0, 0


def tensor_nbytes(value: Any) -> int:
    if hasattr(value, "numel") and hasattr(value, "element_size"):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_nbytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_nbytes(item) for item in value)
    return len(value) if isinstance(value, bytes) else 0


class GPUUtilizationSampler:
    """Low-frequency NVML sampler used only around benchmark measurement windows."""

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.1) -> None:
        self.device_index = device_index
        self.interval = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pynvml: Any = None
        self._handle: Any = None

    def start(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except (ImportError, RuntimeError):
            return

        def sample() -> None:
            while not self._stop.wait(self.interval):
                try:
                    utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                    self.samples.append(float(utilization.gpu))
                except RuntimeError:
                    return

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float | int | bool | None]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._pynvml:
            try:
                self._pynvml.nvmlShutdown()
            except RuntimeError:
                pass
        return {
            "available": bool(self._pynvml),
            "samples": len(self.samples),
            "mean_percent": sum(self.samples) / len(self.samples) if self.samples else None,
            "peak_percent": max(self.samples) if self.samples else None,
        }
