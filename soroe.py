"""
soroe - find the offset between two MKV files of the same content.

Determines how many milliseconds (and frames, given a known framerate) one file
is offset from the other, so you can mux tracks between them with the correct delay.

Requires: numpy, scipy, ffmpeg (on PATH)

Usage:
    python soroe.py file1.mkv file2.mkv
    python soroe.py file1.mkv file2.mkv --fps 23.976
    python soroe.py file1.mkv file2.mkv --duration 300 --verbose
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
from scipy.signal import fftconvolve

# Constants
SAMPLE_RATE = 16_000          # Hz - mono 16 kHz for audio correlation


# Helpers
def _log(msg: str, verbose: bool) -> None:
    if verbose:
        print(f"[soroe] {msg}", file=sys.stderr)


def _check_file(path: str) -> None:
    if not os.path.isfile(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)


def _run_ffprobe(path: str, stream_type: str) -> bool:
    """Return True if *path* contains at least one stream of *stream_type* ('audio' or 'video')."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a" if stream_type == "audio" else "v",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return bool(r.stdout.strip())
    except FileNotFoundError:
        print("Error: ffprobe not found. Make sure ffmpeg is installed and on PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print(f"Error: ffprobe timed out on {path}", file=sys.stderr)
        sys.exit(1)


# Audio extraction
def extract_audio(path: str, duration: int, audio_track: int, verbose: bool) -> np.ndarray:
    """Extract mono 16 kHz s16le PCM from *path* via ffmpeg pipe, return float32 array."""
    _log(f"Extracting audio from {path} (track {audio_track}, {duration}s) …", verbose)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-t", str(duration),
        "-i", path,
        "-map", f"0:a:{audio_track}",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, err = proc.communicate()
    if proc.returncode != 0:
        msg = err.decode(errors="replace").strip()
        print(f"Error: ffmpeg failed on {path}: {msg}", file=sys.stderr)
        sys.exit(1)
    if not raw:
        print(f"Error: no audio data extracted from {path}. Check that the file has an audio track.", file=sys.stderr)
        sys.exit(1)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    _log(f"  Got {len(samples)} samples ({len(samples)/SAMPLE_RATE:.1f}s)", verbose)
    return samples


# Audio cross-correlation
def _find_secondary_peak(corr: np.ndarray, primary_idx: int, min_distance: int) -> float:
    """Return the value of the highest peak that is at least *min_distance* away from *primary_idx*."""
    mask = np.ones(len(corr), dtype=bool)
    lo = max(0, primary_idx - min_distance)
    hi = min(len(corr), primary_idx + min_distance + 1)
    mask[lo:hi] = False
    masked = corr[mask]
    if len(masked) == 0:
        return 0.0
    return float(np.max(masked))


def audio_correlate(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fps: float | None,
    verbose: bool,
) -> dict:
    """Return dict with offset_ms, confidence, and optional frame info."""
    _log("Computing cross-correlation …", verbose)
    # Cross-correlation via convolution: corr(a,b) = convolve(a, b[::-1])
    corr = fftconvolve(sig_a, sig_b[::-1], mode="full")
    corr = np.abs(corr)

    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])

    # lag: positive means sig_b is delayed relative to sig_a
    # In 'full' mode the zero-lag position is at index len(sig_b)-1
    lag_samples = peak_idx - (len(sig_b) - 1)
    offset_ms = lag_samples / SAMPLE_RATE * 1000.0

    # Confidence: ratio of primary to secondary peak
    secondary = _find_secondary_peak(corr, peak_idx, min_distance=SAMPLE_RATE // 2)
    confidence = peak_val / secondary if secondary > 0 else float("inf")

    _log(f"  Peak at lag={lag_samples} samples, confidence={confidence:.2f}", verbose)

    result: dict = {
        "method": "audio cross-correlation",
        "confidence": round(confidence, 2),
        "offset_ms": round(offset_ms, 2),
    }

    if fps is not None:
        offset_frames = offset_ms / 1000.0 * fps
        aligned_frames = round(offset_frames)
        aligned_ms = aligned_frames / fps * 1000.0
        result["offset_frames"] = round(offset_frames, 2)
        result["fps"] = fps
        result["aligned_frames"] = aligned_frames
        result["aligned_ms"] = round(aligned_ms, 2)

    return result





def _confidence_label(ratio: float) -> str:
    """Human-readable confidence mapping."""
    if ratio >= 10:
        return "excellent"
    if ratio >= 5:
        return "good"
    if ratio >= 3:
        return "fair"
    if ratio >= 1.5:
        return "poor"
    return "unreliable"


# Output formatting
def format_result(result: dict) -> str:
    lines: list[str] = []
    lines.append(f"Method: {result['method']}")
    conf = result["confidence"]
    lines.append(f"Confidence: {_confidence_label(conf)} ({conf:.2f}x peak ratio)")

    ms = result["offset_ms"]
    if ms >= 0:
        desc = f"file2 starts {ms:.0f}ms after file1"
    else:
        desc = f"file1 starts {-ms:.0f}ms after file2"
    lines.append(f"Offset: {ms:.0f} ms ({desc})")

    if "fps" in result:
        fps = result["fps"]
        lines.append(f"Offset (frames): {result['offset_frames']:.2f} frames @ {fps}fps")
        lines.append(f"Frame-aligned offset: {result['aligned_ms']:.0f} ms ({result['aligned_frames']} frames)")

    return "\n".join(lines)


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="soroe",
        description="Find the temporal offset between two MKV files of the same content.",
    )
    parser.add_argument("file1", help="First MKV file")
    parser.add_argument("file2", help="Second MKV file")
    parser.add_argument("--fps", type=float, default=None, help="Framerate for frame-accurate offset (e.g., 23.976)")
    parser.add_argument("--duration", type=int, default=600, help="Seconds of content to analyze (default: 600)")
    parser.add_argument("--audio-track", type=int, default=0, help="Audio track index to use (default: 0)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress and debug info")
    args = parser.parse_args()

    _check_file(args.file1)
    _check_file(args.file2)

    if not _run_ffprobe(args.file1, "audio") or not _run_ffprobe(args.file2, "audio"):
        print("Error: one or both files lack an audio track.", file=sys.stderr)
        sys.exit(1)

    sig_a = extract_audio(args.file1, args.duration, args.audio_track, args.verbose)
    sig_b = extract_audio(args.file2, args.duration, args.audio_track, args.verbose)
    result = audio_correlate(sig_a, sig_b, args.fps, args.verbose)

    print(format_result(result))


if __name__ == "__main__":
    main()
