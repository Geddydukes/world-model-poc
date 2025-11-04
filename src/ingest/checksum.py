"""Checksum utilities for ingestion pipelines."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

_CHUNK_SIZE = 1024 * 1024


def _iter_chunks(handle: BinaryIO, chunk_size: int = _CHUNK_SIZE):
    """Yield chunks from an open file handle."""
    while True:
        chunk = handle.read(chunk_size)
        if not chunk:
            break
        yield chunk


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 checksum for *path*.

    The helper streams the file so that large videos can be hashed without
    exhausting memory.
    """

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in _iter_chunks(handle):
            digest.update(chunk)
    return digest.hexdigest()
