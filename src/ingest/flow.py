"""Optical flow helpers used during ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .ffmpeg_utils import ensure_dir


def compute_dense_flow(frames: np.ndarray) -> np.ndarray:
    """Compute a dense optical-flow volume using Farneback."""

    flows = []
    prev_gray: Optional[np.ndarray] = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        flows.append(flow.astype(np.float32))
        prev_gray = gray
    if not flows:
        return np.zeros((0, *frames.shape[1:3], 2), dtype=np.float32)
    return np.stack(flows, axis=0)


def save_flow(flow: np.ndarray, directory: str | Path) -> Path:
    """Persist flow arrays next to the clip."""

    out_dir = ensure_dir(directory)
    flow_path = out_dir / "flow.npy"
    np.save(flow_path, flow.astype(np.float32))
    return flow_path
