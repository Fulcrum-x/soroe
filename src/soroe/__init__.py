"""
soroe - find the offset between two audio/video files of the same content.

Determines how many milliseconds (and frames, given a known framerate) one file
is offset from the other, so you can mux tracks between them with the correct delay.

Supports any format ffmpeg can read (mkv, m4a, eac3, mka, mks, mp4, flac, wav, …).

Requires: numpy, scipy, ffmpeg (on PATH)

Usage:
    uv run soroe file1.mkv file2.mkv
    uv run soroe file1.mkv file2.mkv --duration 300 --verbose
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


def _get_fps(path: str) -> float | None:
    """Return the framerate of the first video stream in *path*, or None if no video."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        val = r.stdout.strip()
        if not val or "/" not in val:
            return None
        num, den = val.split("/")
        return float(num) / float(den) if float(den) != 0 else None
    except (ValueError, ZeroDivisionError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _get_duration(path: str) -> float:
    """Return duration of *path* in seconds via ffprobe."""
    try:
        r = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(r.stdout.strip())
    except (ValueError, subprocess.TimeoutExpired):
        print(f"Error: could not determine duration of {path}", file=sys.stderr)
        sys.exit(1)


def _format_timestamp(seconds: float) -> str:
    """Format *seconds* as HH:MM:SS.s."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:04.1f}"


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
        nearest_frame = round(offset_frames)
        result["offset_frames"] = round(offset_frames, 2)
        result["fps"] = fps
        result["nearest_frame"] = nearest_frame

    return result


# Output formatting
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
        lines.append(f"Nearest frame: {result['nearest_frame']}")

    return "\n".join(lines)


# Drift analysis - windowed cross-correlation with adaptive refinement

def correlate_window(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    window_start_samples: int,
    window_size_samples: int,
    search_radius_samples: int,
) -> tuple[float, float] | None:
    """Correlate a window from *sig_a* against a search region in *sig_b*.

    Returns ``(offset_ms, confidence)`` or *None* if the window is too small.
    """
    w_start = window_start_samples
    w_end = min(w_start + window_size_samples, len(sig_a))
    a_win = sig_a[w_start:w_end]

    if len(a_win) < window_size_samples // 4:
        return None

    b_start = max(0, w_start - search_radius_samples)
    b_end = min(len(sig_b), w_start + window_size_samples + search_radius_samples)
    b_search = sig_b[b_start:b_end]

    if len(b_search) < len(a_win):
        return None

    corr = fftconvolve(a_win, b_search[::-1], mode="full")
    corr = np.abs(corr)

    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])

    lag_in_corr = peak_idx - (len(b_search) - 1)
    offset_samples = w_start - b_start + lag_in_corr
    offset_ms = offset_samples / SAMPLE_RATE * 1000.0

    secondary = _find_secondary_peak(corr, peak_idx, min_distance=SAMPLE_RATE // 2)
    confidence = peak_val / secondary if secondary > 0 else float("inf")

    return (round(offset_ms, 2), round(confidence, 2))


def refine_change_point(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    t_start_s: float,
    t_end_s: float,
    offset_before_ms: float,
    offset_after_ms: float,
    min_window_s: float,
    search_radius_samples: int,
    threshold_ms: float,
) -> float:
    """Binary-search *[t_start_s, t_end_s]* to pinpoint where the offset changes."""
    if t_end_s - t_start_s <= min_window_s:
        return (t_start_s + t_end_s) / 2.0

    t_mid = (t_start_s + t_end_s) / 2.0
    win_samples = max(
        int(min_window_s * SAMPLE_RATE),
        int((t_end_s - t_start_s) / 4 * SAMPLE_RATE),
    )
    mid_start = int(t_mid * SAMPLE_RATE)

    result = correlate_window(sig_a, sig_b, mid_start, win_samples, search_radius_samples)
    if result is None:
        return (t_start_s + t_end_s) / 2.0

    mid_offset_ms, _ = result

    if abs(mid_offset_ms - offset_before_ms) >= threshold_ms:
        return refine_change_point(
            sig_a, sig_b, t_start_s, t_mid,
            offset_before_ms, mid_offset_ms,
            min_window_s, search_radius_samples, threshold_ms,
        )
    return refine_change_point(
        sig_a, sig_b, t_mid, t_end_s,
        mid_offset_ms, offset_after_ms,
        min_window_s, search_radius_samples, threshold_ms,
    )


def drift_analysis(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    window_s: int,
    threshold_ms: int,
    max_drift_s: int,
    fps: float | None,
    verbose: bool,
) -> dict:
    """Three-pass drift detection: coarse scan, change-point refinement, summary."""
    window_samples = int(window_s * SAMPLE_RATE)
    step_samples = window_samples // 2  # 50 % overlap
    search_radius_samples = int(max_drift_s * SAMPLE_RATE)
    min_window_s = 2.0
    total_samples = min(len(sig_a), len(sig_b))
    total_duration_s = total_samples / SAMPLE_RATE

    # Pass 1: coarse windowed cross-correlation
    _log("Pass 1: coarse windowed cross-correlation …", verbose)
    positions: list[int] = []
    pos = 0
    while pos + window_samples // 4 <= total_samples:
        positions.append(pos)
        pos += step_samples

    coarse: list[dict] = []
    for i, pos in enumerate(positions):
        actual_win = min(window_samples, total_samples - pos)
        _log(f"Analyzing window {i + 1}/{len(positions)} …", verbose)
        result = correlate_window(sig_a, sig_b, pos, actual_win, search_radius_samples)
        if result is not None:
            offset_ms, confidence = result
            coarse.append({
                "timestamp_s": pos / SAMPLE_RATE,
                "offset_ms": offset_ms,
                "confidence": confidence,
            })

    if not coarse:
        return {"segments": [], "change_points": [], "no_drift": True,
                "window_s": window_s, "threshold_ms": threshold_ms}

    # Filter outlier windows whose offset is far from both neighbors
    if len(coarse) > 2:
        max_jump_ms = max_drift_s * 1000
        filtered = [coarse[0]]
        for i in range(1, len(coarse) - 1):
            pt = coarse[i]
            close_prev = abs(pt["offset_ms"] - coarse[i - 1]["offset_ms"]) <= max_jump_ms
            close_next = abs(pt["offset_ms"] - coarse[i + 1]["offset_ms"]) <= max_jump_ms
            if close_prev or close_next:
                filtered.append(pt)
        filtered.append(coarse[-1])
        if len(filtered) >= 2:
            if abs(filtered[0]["offset_ms"] - filtered[1]["offset_ms"]) > max_jump_ms:
                filtered.pop(0)
        if len(filtered) >= 2:
            if abs(filtered[-1]["offset_ms"] - filtered[-2]["offset_ms"]) > max_jump_ms:
                filtered.pop()
        if filtered:
            coarse = filtered

    # Pass 2: change-point detection & refinement
    _log("Pass 2: detecting change points …", verbose)
    change_points: list[dict] = []
    for i in range(1, len(coarse)):
        prev, curr = coarse[i - 1], coarse[i]
        if abs(curr["offset_ms"] - prev["offset_ms"]) >= threshold_ms:
            t_start = prev["timestamp_s"]
            t_end = min(curr["timestamp_s"] + window_s, total_duration_s)
            _log(
                f"  Refining change between {_format_timestamp(t_start)} "
                f"and {_format_timestamp(t_end)} …",
                verbose,
            )
            cp_t = refine_change_point(
                sig_a, sig_b, t_start, t_end,
                prev["offset_ms"], curr["offset_ms"],
                min_window_s, search_radius_samples, threshold_ms,
            )
            change_points.append({
                "timestamp_s": cp_t,
                "offset_before_ms": prev["offset_ms"],
                "offset_after_ms": curr["offset_ms"],
            })

    change_points.sort(key=lambda cp: cp["timestamp_s"])

    # Pass 3: compile segments
    _log("Pass 3: compiling segments …", verbose)

    def _avg(pts: list[dict], key: str) -> float:
        return sum(p[key] for p in pts) / len(pts) if pts else 0.0

    if not change_points:
        avg_off = _avg(coarse, "offset_ms")
        avg_conf = _avg(coarse, "confidence")
        return {
            "segments": [{
                "start_s": 0.0, "end_s": total_duration_s,
                "offset_ms": round(avg_off, 1), "confidence": round(avg_conf, 2),
            }],
            "change_points": [],
            "no_drift": True,
            "window_s": window_s,
            "threshold_ms": threshold_ms,
        }

    transition_half = 1.5  # seconds each side of a change point
    segments: list[dict] = []
    seg_start = 0.0

    for cp in change_points:
        cp_t = cp["timestamp_s"]
        seg_end = max(seg_start, cp_t - transition_half)

        if seg_end > seg_start:
            pts = [r for r in coarse if seg_start <= r["timestamp_s"] < seg_end]
            segments.append({
                "start_s": seg_start, "end_s": seg_end,
                "offset_ms": round(_avg(pts, "offset_ms"), 1) if pts else cp["offset_before_ms"],
                "confidence": round(_avg(pts, "confidence"), 2) if pts else 0.0,
            })

        trans_end = max(seg_end, min(cp_t + transition_half, total_duration_s))
        segments.append({
            "start_s": seg_end, "end_s": trans_end,
            "offset_ms": None, "confidence": None,  # transition
        })
        seg_start = trans_end

    if seg_start < total_duration_s:
        pts = [r for r in coarse if r["timestamp_s"] >= seg_start]
        segments.append({
            "start_s": seg_start, "end_s": total_duration_s,
            "offset_ms": round(_avg(pts, "offset_ms"), 1) if pts else change_points[-1]["offset_after_ms"],
            "confidence": round(_avg(pts, "confidence"), 2) if pts else 0.0,
        })

    return {
        "segments": segments,
        "change_points": change_points,
        "no_drift": False,
        "window_s": window_s,
        "threshold_ms": threshold_ms,
    }


# Output formatting
def format_drift_result(result: dict, fps: float | None) -> str:
    """Format drift-analysis *result* for human consumption."""
    lines: list[str] = []
    w = result["window_s"]
    thr = result["threshold_ms"]
    lines.append(f"Drift analysis ({w}s windows, {thr}ms threshold)")
    lines.append("=" * 46)
    lines.append("")

    if result["no_drift"]:
        seg = result["segments"][0]
        off = seg["offset_ms"]
        conf = _confidence_label(seg["confidence"])
        sign = "+" if off >= 0 else ""
        lines.append("No drift detected.")
        lines.append("")
        lines.append("Segments:")
        ts0 = _format_timestamp(seg["start_s"])
        ts1 = _format_timestamp(seg["end_s"])
        entry = f"  {ts0} - {ts1}  offset: {sign}{off:.0f} ms  (confidence: {conf} ({seg["confidence"]:.2f}x))"
        if fps is not None:
            entry += f"  [{off / 1000.0 * fps:+.1f} frames]"
        lines.append(entry)
        return "\n".join(lines)

    lines.append("Segments:")
    for seg in result["segments"]:
        ts0 = _format_timestamp(seg["start_s"])
        ts1 = _format_timestamp(seg["end_s"])
        if seg["offset_ms"] is None:
            lines.append(f"  {ts0} - {ts1}  transition")
        else:
            off = seg["offset_ms"]
            conf = _confidence_label(seg["confidence"])
            sign = "+" if off >= 0 else ""
            entry = f"  {ts0} - {ts1}  offset: {sign}{off:.0f} ms  (confidence: {conf} ({seg["confidence"]:.2f}x))"
            if fps is not None:
                entry += f"  [{off / 1000.0 * fps:+.1f} frames]"
            lines.append(entry)

    lines.append("")
    lines.append("Change points:")
    for cp in result["change_points"]:
        ts = _format_timestamp(cp["timestamp_s"])
        before = cp["offset_before_ms"]
        after = cp["offset_after_ms"]
        delta = abs(after - before)
        sign_b = "+" if before >= 0 else ""
        sign_a = "+" if after >= 0 else ""
        entry = (
            f"  {ts}  offset shifts from {sign_b}{before:.0f} ms "
            f"to {sign_a}{after:.0f} ms (Δ{delta:.0f} ms)"
        )
        if fps is not None:
            entry += f"  [Δ{delta / 1000.0 * fps:.1f} frames]"
        lines.append(entry)

    lines.append("")
    lines.append("Global summary:")
    non_transition = [s for s in result["segments"] if s["offset_ms"] is not None]
    lines.append(f"  Segments: {len(non_transition)}")
    lines.append(f"  Total change points: {len(result['change_points'])}")
    if result["change_points"]:
        max_delta = max(
            abs(cp["offset_after_ms"] - cp["offset_before_ms"])
            for cp in result["change_points"]
        )
        lines.append(f"  Max drift: {max_delta:.0f} ms")

    return "\n".join(lines)


# Main
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="soroe",
        description="Find the temporal offset between two audio/video files of the same content.",
    )
    parser.add_argument("file1", help="First audio/video file")
    parser.add_argument("file2", help="Second audio/video file")
    parser.add_argument("--duration", type=int, default=None, help="Seconds of content to analyze (default: 600, or full file with --drift)")
    parser.add_argument("--audio-track", type=int, default=0, help="Audio track index to use (default: 0)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress and debug info")
    parser.add_argument("--drift", action="store_true", help="Enable drift/divergence detection mode")
    parser.add_argument("--drift-window", type=int, default=30, help="Window size in seconds for drift analysis (default: 30)")
    parser.add_argument("--drift-threshold", type=int, default=70, help="Minimum offset change in ms to count as a change point (default: 70)")
    parser.add_argument("--max-drift", type=int, default=5, help="Maximum expected drift in seconds - sets the search radius (default: 5)")
    args = parser.parse_args()

    _check_file(args.file1)
    _check_file(args.file2)

    if not _run_ffprobe(args.file1, "audio") or not _run_ffprobe(args.file2, "audio"):
        print("Error: one or both files lack an audio track.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect FPS from the first file with a video stream
    fps = _get_fps(args.file1) or _get_fps(args.file2)
    if fps is not None:
        _log(f"Detected framerate: {fps:.3f} fps", args.verbose)

    # Determine duration
    if args.duration is not None:
        duration = args.duration
    elif args.drift:
        dur1 = _get_duration(args.file1)
        dur2 = _get_duration(args.file2)
        duration = int(min(dur1, dur2))
        _log(f"Auto-detected duration: {duration}s (from shorter file)", args.verbose)
    else:
        duration = 600

    sig_a = extract_audio(args.file1, duration, args.audio_track, args.verbose)
    sig_b = extract_audio(args.file2, duration, args.audio_track, args.verbose)

    if args.drift:
        result = drift_analysis(
            sig_a, sig_b,
            window_s=args.drift_window,
            threshold_ms=args.drift_threshold,
            max_drift_s=args.max_drift,
            fps=fps,
            verbose=args.verbose,
        )
        print(format_drift_result(result, fps))
    else:
        result = audio_correlate(sig_a, sig_b, fps, args.verbose)
        print(format_result(result))


if __name__ == "__main__":
    main()
