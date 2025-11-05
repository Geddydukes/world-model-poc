from __future__ import annotations

import torch


def autocast_enabled(device: str | torch.device) -> bool:
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = str(device)
    return device_type in ("cuda", "mps")


def amp_dtype(device: torch.device | str | None = None) -> torch.dtype:
    if isinstance(device, torch.device):
        device_type = device.type
    elif isinstance(device, str) and device:
        device_type = device
    else:
        device_type = "cpu"

    if device_type == "cuda" and torch.cuda.is_available():
        is_bf16 = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        if is_bf16:
            return torch.bfloat16
        return torch.float16

    if device_type == "mps" and torch.backends.mps.is_available():
        return torch.bfloat16

    return torch.float32


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
