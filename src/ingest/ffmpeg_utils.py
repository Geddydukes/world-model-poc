"""Utilities for interacting with ffmpeg/ffprobe."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from .checksum import sha256_file


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg invocation fails."""


def run_ffmpeg(cmd: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """Run *cmd* while emitting a helpful error on failure."""

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if check and proc.returncode != 0:
        raise FFmpegError(
            "ffmpeg command failed"  # pragma: no cover - defensive logging
            f"\ncommand: {' '.join(cmd)}\nstdout: {proc.stdout.decode(errors='ignore')}\nstderr: {proc.stderr.decode(errors='ignore')}"
        )
    return proc


def probe_video(path: str | Path) -> dict:
    """Return ffprobe metadata for *path*."""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = run_ffmpeg(cmd)
    return json.loads(proc.stdout.decode("utf-8"))


def compute_clip_id(path: str | Path) -> str:
    """Derive a deterministic clip identifier from the file checksum."""

    return sha256_file(path)[:16]


def ensure_dir(path: str | Path) -> Path:
    """Create *path* if missing and return it as a :class:`Path`."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ffmpeg_segment_command(
    source: str | Path,
    destination_template: str | Path,
    *,
    clip_seconds: int,
) -> Iterable[str]:
    """Return the ffmpeg command that slices *source* into fixed-length clips."""

    return (
        "ffmpeg",
        "-i",
        str(source),
        "-c",
        "copy",
        "-map",
        "0",
        "-segment_time",
        str(clip_seconds),
        "-f",
        "segment",
        "-reset_timestamps",
        "1",
        str(destination_template),
    )
