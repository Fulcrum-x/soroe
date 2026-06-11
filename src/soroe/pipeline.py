"""Analysis pipelines shared by the single-pair CLI and batch mode."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import ffmpeg, log, report
from .dsp import audio_correlate, drift_analysis
from .errors import SoroeError

# Working sample rate for correlation. It lives here (not in dsp.py or
# ffmpeg.py) because the pipelines are what choose it; extraction and DSP both
# take sample_rate as a parameter. Parabolic peak interpolation already
# resolves well below a sample at 16 kHz, so a higher rate buys nothing.
SAMPLE_RATE = 16_000  # Hz - mono working rate (speed/precision tradeoff)


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        log.info(msg)


def _prepare_signals(
    file1: str,
    file2: str,
    duration: int | None,
    full_duration_if_unset: bool,
    audio_track: int,
    sample_rate: int,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Validate inputs and extract aligned-length audio signals from both files.

    Raises SoroeError on any expected per-input failure (missing file, no audio
    track, ffprobe/ffmpeg failure, same file passed twice).
    """
    ffmpeg.check_file(file1)
    ffmpeg.check_file(file2)

    if os.path.realpath(file1) == os.path.realpath(file2):
        raise SoroeError("both arguments point to the same file.")

    count1 = ffmpeg.count_audio_tracks(file1)
    count2 = ffmpeg.count_audio_tracks(file2)
    if count1 == 0 or count2 == 0:
        raise SoroeError("one or both files lack an audio track.")
    if audio_track >= count1 or audio_track >= count2:
        parts: list[str] = []
        if audio_track >= count1:
            parts.append(
                f"{os.path.basename(file1)} has {count1} track{'s' if count1 != 1 else ''}"
            )
        if audio_track >= count2:
            parts.append(
                f"{os.path.basename(file2)} has {count2} track{'s' if count2 != 1 else ''}"
            )
        log.warn(
            f"audio track index {audio_track} not available ({'; '.join(parts)}); "
            f"falling back to track 0."
        )
        audio_track = 0

    dur1 = ffmpeg.get_duration(file1)
    dur2 = ffmpeg.get_duration(file2)
    if abs(dur1 - dur2) > 240:
        log.warn(
            f"file durations differ by {abs(dur1 - dur2):.0f}s "
            f"({dur1:.0f}s vs {dur2:.0f}s). These may be different sources."
        )

    fps = ffmpeg.get_fps(file1) or ffmpeg.get_fps(file2)
    if fps is not None:
        _log(f"Detected framerate: {fps:.3f} fps", verbose)

    if duration is not None:
        dur = duration
    elif full_duration_if_unset:
        dur = int(min(dur1, dur2))
        _log(f"Auto-detected duration: {dur}s (from shorter file)", verbose)
    else:
        dur = 600

    if not verbose:
        log.progress(0, 2, "Extracting audio (parallel)")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(ffmpeg.extract_audio, file1, dur, audio_track, sample_rate, verbose)
        fut_b = ex.submit(ffmpeg.extract_audio, file2, dur, audio_track, sample_rate, verbose)
        sig_a = fut_a.result()
        sig_b = fut_b.result()

    return sig_a, sig_b, fps


def run_single_shot(
    file1: str,
    file2: str,
    duration: int | None,
    audio_track: int,
    verbose: bool,
) -> str:
    """Run global-offset analysis on one file pair and return the formatted output."""
    sig_a, sig_b, fps = _prepare_signals(
        file1,
        file2,
        duration,
        False,
        audio_track,
        SAMPLE_RATE,
        verbose,
    )
    if not verbose:
        log.progress(1, 2, "Computing correlation")
    result = audio_correlate(sig_a, sig_b, fps, SAMPLE_RATE, True, verbose)
    if not verbose:
        log.progress_clear()
    output = report.format_result(result)
    # Peak-to-sidelobe ratio of the whitened (GCC-PHAT) correlogram: a clean
    # global lock sits well above 5; below that the alignment is suspect.
    if result["confidence"] < 5:
        log.warn("low confidence may indicate offset variability across the timeline.")
        log.warn("try running with --drift to check for change points.")
    return output


def run_drift(
    file1: str,
    file2: str,
    duration: int | None,
    audio_track: int,
    drift_window: int | None,
    drift_threshold: int | None,
    max_drift: int | None,
    prescan: bool,
    verbose: bool,
    save_offsets_as: str | None = None,
) -> str:
    """Run drift analysis on one file pair and return the formatted output.

    The three drift parameters accept None for auto resolution inside
    dsp.drift_analysis; *prescan* requests the window-size calibration.
    If *save_offsets_as* is given, also writes the segments/change points to
    ./offsets/<save_offsets_as>.txt.
    """
    sig_a, sig_b, fps = _prepare_signals(
        file1,
        file2,
        duration,
        True,
        audio_track,
        SAMPLE_RATE,
        verbose,
    )
    if not verbose:
        log.progress(1, 2, "Analyzing drift")
    result = drift_analysis(
        sig_a,
        sig_b,
        window_s=drift_window,
        threshold_ms=drift_threshold,
        max_drift_s=max_drift,
        fps=fps,
        sample_rate=SAMPLE_RATE,
        prescan=prescan,
        verbose=verbose,
    )
    if not verbose:
        log.progress_clear()
    if save_offsets_as is not None:
        path = report.save_offsets(result, fps, save_offsets_as)
        log.info(f"Saved offsets to {path}")
    return report.format_drift_result(result, fps)
