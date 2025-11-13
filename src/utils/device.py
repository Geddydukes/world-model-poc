"""Utility helpers for selecting runtime devices."""

from __future__ import annotations

import torch

__all__ = ["resolve_device", "device_type_str"]


def resolve_device(requested: str | None = "auto") -> torch.device:
    """Return a :class:`torch.device` based on *requested* preference."""

    if requested and requested != "auto":
        try:
            device = torch.device(requested)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return device
        except Exception:
            # Fall back to auto detection if explicit request fails.
            pass
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


def device_type_str(device: torch.device) -> str:
    """Return the lowercase device type ("cpu", "cuda", "mps")."""

    return device.type
