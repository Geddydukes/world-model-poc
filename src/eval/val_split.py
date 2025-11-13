"""Deterministic validation split helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable, Tuple


def split_clip_ids(clip_ids: Iterable[str], *, val_ratio: float = 0.1) -> Tuple[list[str], list[str]]:
    train, val = [], []
    for clip_id in clip_ids:
        digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF
        (val if frac < val_ratio else train).append(clip_id)
    return train, val
