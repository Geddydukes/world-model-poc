from __future__ import annotations

import torch
from contextlib import contextmanager
from typing import Optional, Union

DeviceLike = Union[str, torch.device, None]


def _device_type_str(device: DeviceLike) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str) and device:
        return device
    # Auto-detect if not provided
    if torch.cuda.is_available():
        return "cuda"
    # torch.backends.mps.is_available can raise if MPS not built; guard it
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def autocast_enabled(device: DeviceLike) -> bool:
    return _device_type_str(device) in ("cuda", "mps")


def amp_dtype(device: DeviceLike = None) -> torch.dtype:
    """
    Preferred autocast dtype for the given device.

    - float16 on CUDA/MPS
    - bfloat16 on CPU when available
    - float32 as a safe fallback
    """
    device_type = _device_type_str(device)
    if device_type in ("cuda", "mps"):
        return torch.float16
    # CPU
    if getattr(torch.backends.cpu, "has_bf16", False):
        return torch.bfloat16
    return torch.float32


@contextmanager
def maybe_autocast(device: DeviceLike = None):
    """
    Context manager that uses torch.autocast when supported; no-op otherwise.
    """
    device_type = _device_type_str(device)
    if autocast_enabled(device_type):
        with torch.autocast(device_type=device_type, dtype=amp_dtype(device_type)):
            yield
    else:
        yield


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
