"""Simple object tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TrackState:
    object_id: str
    frame_idx: int
    position: np.ndarray
    velocity: np.ndarray


class CentroidTracker:
    """Nearest-neighbour tracker placeholder."""

    def __init__(self, max_distance: float = 30.0) -> None:
        self.max_distance = max_distance
        self._next_id = 0
        self._tracks: Dict[str, np.ndarray] = {}

    def update(self, detections: List[np.ndarray], frame_idx: int) -> List[TrackState]:
        states = []
        for det in detections:
            best_id = None
            best_dist = float("inf")
            for object_id, prev in self._tracks.items():
                dist = np.linalg.norm(prev - det)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_id = object_id
            if best_id is None:
                best_id = f"obj_{self._next_id}"
                self._next_id += 1
            velocity = det - self._tracks.get(best_id, det)
            self._tracks[best_id] = det
            states.append(TrackState(object_id=best_id, frame_idx=frame_idx, position=det, velocity=velocity))
        return states
