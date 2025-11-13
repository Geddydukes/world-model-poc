"""Qualitative visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_frame_grid(frames: np.ndarray, *, cols: int = 4, save_path: Optional[str | Path] = None) -> Path:
    rows = int(np.ceil(len(frames) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.asarray(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if idx < len(frames):
            ax.imshow(frames[idx])
    if save_path is None:
        save_path = Path("reports/figures/grid.png")
    else:
        save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
