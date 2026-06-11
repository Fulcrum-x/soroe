"""ffmpeg/ffprobe shelling: file checks, stream probing, and audio extraction."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from . import log
from .errors import SoroeError


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        log.info(msg)


def check_file(path: str) -> None:
    """Raise SoroeError if *path* is not an existing file."""
    if not os.path.isfile(path):
        raise SoroeError(f"file not found: {path}")


def count_audio_tracks(path: str) -> int:
    """Return the number of audio streams in *path*."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return sum(1 for line in r.stdout.splitlines() if line.strip())
    except FileNotFoundError:
        # System-level: ffprobe missing. Always terminates, even in batch mode.
        log.error("ffprobe not found. Make sure ffmpeg is installed and on PATH.")
        sys.exit(1)
    except subprocess.TimeoutExpired as e:
        raise SoroeError(f"ffprobe timed out on {path}") from e


def get_fps(path: str) -> float | None:
    """Return the framerate of the first video stream in *path*, or None if no video."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        val = r.stdout.strip()
        if not val or "/" not in val:
            return None
        num, den = val.split("/")
        return float(num) / float(den) if float(den) != 0 else None
    except (ValueError, ZeroDivisionError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def get_duration(path: str) -> float:
    """Return duration of *path* in seconds via ffprobe."""
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(r.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired) as e:
        raise SoroeError(f"could not determine duration of {path}") from e


def extract_audio(
    path: str, duration: int, audio_track: int, sample_rate: int, verbose: bool
) -> np.ndarray:
    """Extract mono *sample_rate* s16le PCM from *path* via ffmpeg pipe.

    Returns a DC-removed float32 array scaled to [-1, 1).
    """
    _log(
        f"Extracting audio from {path} (track {audio_track}, {duration}s, {sample_rate} Hz) …",
        verbose,
    )
    # Keep ffmpeg flag/value pairs on shared lines.
    # fmt: off
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-t", str(duration),
        "-i", path,
        "-map", f"0:a:{audio_track}",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    # fmt: on
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        # System-level: ffmpeg missing. Always terminates.
        log.error("ffmpeg not found. Make sure ffmpeg is installed and on PATH.")
        sys.exit(1)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        msg = err.decode(errors="replace").strip()
        raise SoroeError(f"ffmpeg failed on {path}: {msg}")
    if not raw:
        raise SoroeError(
            f"no audio data extracted from {path}. Check that the file has an audio track."
        )

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    samples /= 32768.0
    # Decoder DC bias correlates as a triangular ramp peaking at lag 0.
    samples -= samples.mean()
    _log(f" + Got {len(samples)} samples ({len(samples) / sample_rate:.1f}s)", verbose)
    return samples
