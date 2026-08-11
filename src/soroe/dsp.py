"""Signal-processing core for soroe: audio cross-correlation and drift detection.

Pure DSP over numpy/scipy arrays — no ffmpeg shelling, argparse, or output
formatting. The working sample rate is passed in by the caller (pipeline.py owns
the rate constants), so this module stays independent of I/O and presentation.

Sign convention: a positive ``offset_ms`` means file2 is delayed relative to
file1 (file2 starts later), implemented via ``lag = peak position - (len(sig_b) - 1)``
on the full-lag correlation axis. Do not flip it.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import resample_poly

from . import log


def _log(msg: str, verbose: bool) -> None:
    if verbose:
        log.info(msg)


# Audio cross-correlation
_PHAT_RHO = 0.8  # spectral-whitening exponent: 1.0 = full PHAT, 0.0 = plain correlation

# Whitened peak-to-sidelobe ratios near 1 mean a tie — no usable alignment in
# the window (silence, unmatched content reads ~1.0-1.05 within the clamped
# band). Honest-but-smeared windows under strong linear drift can read as low
# as ~1.15, so the floor sits just under that: gate only true no-information
# windows and let confidence weighting handle the middle ground.
_CONFIDENCE_FLOOR = 1.1

# A near-zero sidelobe would read as infinite confidence and poison averages.
_MAX_CONFIDENCE = 1000.0


def _phat_correlate(sig_a: np.ndarray, sig_b: np.ndarray, rho: float = _PHAT_RHO) -> np.ndarray:
    """GCC-PHAT cross-correlation of *sig_a* against *sig_b*.

    Whitens the cross-spectrum by ``|R|**rho`` (epsilon-floored) before the
    inverse transform, so the peak sharpness no longer depends on the program
    material's own spectrum. ``rho=0`` skips whitening (plain correlation).
    Returns the full lag axis — length ``len(a) + len(b) - 1`` with index
    ``k`` holding lag ``k - (len(b) - 1)``, the same layout as
    ``fftconvolve(a, b[::-1], mode="full")`` — so the sign convention
    (positive lag => sig_b delayed) carries over unchanged.
    """
    n_a, n_b = len(sig_a), len(sig_b)
    nfft = next_fast_len(n_a + n_b - 1, real=True)
    spec_b = rfft(sig_b, nfft)
    np.conjugate(spec_b, out=spec_b)
    spec = rfft(sig_a, nfft)
    spec *= spec_b
    del spec_b
    if rho:
        mag = np.abs(spec)
        np.power(mag, rho, out=mag)
        mag += np.finfo(mag.dtype).tiny  # silent bins divide to 0, not NaN
        spec /= mag
        del mag
    corr = irfft(spec, nfft)
    del spec
    # Circular -> linear full layout: negative lags wrap to the top end.
    return np.concatenate((corr[nfft - (n_b - 1) :], corr[:n_a]))


def _find_secondary_peak(corr: np.ndarray, primary_idx: int, min_distance: int) -> float:
    """Return the value of the highest peak that is at least *min_distance* away from *primary_idx*."""
    lo = max(0, primary_idx - min_distance)
    hi = min(len(corr), primary_idx + min_distance + 1)
    # Two slice-maxes: views, no mask allocation or fancy-index copy.
    return float(max(corr[:lo].max(initial=0.0), corr[hi:].max(initial=0.0)))


def _parabolic_peak_offset(corr: np.ndarray, peak_idx: int) -> float:
    """Sub-sample peak refinement via parabolic interpolation around *peak_idx*.

    Returns a fractional sample delta clamped to [-0.5, 0.5]; 0.0 when the peak
    is at an edge or the curvature is degenerate.
    """
    if peak_idx <= 0 or peak_idx >= len(corr) - 1:
        return 0.0
    y0 = float(corr[peak_idx - 1])
    y1 = float(corr[peak_idx])
    y2 = float(corr[peak_idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (y0 - y2) / denom))


def audio_correlate(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    fps: float | None,
    sample_rate: int,
    interpolate: bool,
    verbose: bool,
) -> dict:
    """Return dict with offset_ms, confidence, and optional frame info."""
    _log("Computing cross-correlation …", verbose)
    corr = _phat_correlate(sig_a, sig_b)
    np.abs(corr, out=corr)

    # lag: positive means sig_b is delayed relative to sig_a
    # In the full-lag layout the zero-lag position is at index len(sig_b)-1
    zero_idx = len(sig_b) - 1
    peak_idx = int(np.argmax(corr))
    peak_val = float(corr[peak_idx])

    peak_pos = peak_idx + (_parabolic_peak_offset(corr, peak_idx) if interpolate else 0.0)
    lag_samples = peak_pos - zero_idx
    offset_ms = lag_samples / sample_rate * 1000.0

    # Confidence: peak-to-sidelobe ratio of the whitened correlogram
    secondary = _find_secondary_peak(corr, peak_idx, min_distance=sample_rate // 2)
    confidence = min(peak_val / secondary, _MAX_CONFIDENCE) if secondary > 0 else _MAX_CONFIDENCE

    _log(f" + Peak at lag={lag_samples:.2f} samples, confidence={confidence:.2f}", verbose)

    result: dict = {
        "method": "audio cross-correlation (GCC-PHAT)",
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


# Drift analysis - windowed cross-correlation with adaptive refinement

# Auto-resolution fallbacks and clamps, used when the drift CLI flags are left
# at their None/auto defaults. Explicit flag values bypass all of this.
_DEFAULT_WINDOW_S = 30  # fixed --drift-window default; --prescan calibrates instead
_DEFAULT_RADIUS_S = 5.0  # auto --max-drift fallback when the probes carry no dispersion info
_DEFAULT_THRESHOLD_MS = 70.0  # auto --drift-threshold fallback; also the probe-margin floor
_AUTO_RADIUS_FLOOR_S = 3.0  # the auto radius never narrows below this (covers real step edits)
# Ceiling on the auto radius. Pass-1 transient FFT memory per window is roughly
# (window + 2*radius) * sample_rate * ~28 bytes across os.cpu_count() threads:
# at 120 s that is ~120 MB/window, ~4 GiB on a 32-thread machine, while real
# drift around the anchor rarely spans more than tens of seconds. Explicit
# --max-drift and the widen-and-warn path are not clamped.
_AUTO_RADIUS_CEIL_S = 120.0
_AUTO_THRESHOLD_NOISE_MULT = 6.0  # sigma multiplier: scan jitter must not fire change points
_AUTO_THRESHOLD_RANGE_MS = (20.0, 70.0)  # sub-frame-jitter floor .. noticeable-desync ceiling
_PRESCAN_CANDIDATES_S = (10, 15, 20, 30, 45)
_PRESCAN_MIN_CONFIDENCE = 2.5  # median lock a candidate window size must clear across locations
_REPAIR_TRUST_CONFIDENCE = 5.0  # a neighbor-re-anchored window must clear this to be adopted


def _announce(msg: str) -> None:
    """Always-on parameter notice to stderr; clears any active progress bar first."""
    log.progress_clear()
    log.info(msg)


def correlate_window(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    window_start_samples: int,
    window_size_samples: int,
    search_radius_samples: int,
    sample_rate: int,
    interpolate: bool,
    anchor_offset_samples: int = 0,
) -> tuple[float, float] | None:
    """Correlate a window from *sig_a* against a search region in *sig_b*.

    The search region in *sig_b* is centered at the window position shifted by
    *anchor_offset_samples* (the expected global offset), so the search radius
    bounds drift around that anchor rather than the offset itself. Skip 
    interpolation if the peak lands on either edge of the search band. Returns
    ``(offset_ms, confidence)`` — offset_ms is the global offset — or *None*
    if there is too little signal to search.
    """
    w_start = window_start_samples
    w_end = min(w_start + window_size_samples, len(sig_a))
    a_win = sig_a[w_start:w_end]

    if len(a_win) < window_size_samples // 4:
        return None

    # Matching content for a positive offset sits *earlier* in sig_b.
    b_start = max(0, w_start - anchor_offset_samples - search_radius_samples)
    b_end = min(
        len(sig_b), w_start + window_size_samples - anchor_offset_samples + search_radius_samples
    )
    b_search = sig_b[b_start : max(b_start, b_end)]

    if len(b_search) < len(a_win):
        return None

    corr = _phat_correlate(a_win, b_search)
    np.abs(corr, out=corr)

    # Clamp the peak search to lags whose global offset lies within the
    # declared radius of the anchor. The full-lag axis reaches out to
    # ±(window+radius) via partial overlaps; everything beyond the radius is
    # out of contract (repeated cues, taper region) and can only mislead.
    zero_idx = len(b_search) - 1  # corr index of local lag 0
    shift = w_start - b_start  # local lag -> global offset, in samples
    k_lo = max(0, anchor_offset_samples - search_radius_samples - shift + zero_idx)
    k_hi = min(len(corr) - 1, anchor_offset_samples + search_radius_samples - shift + zero_idx)
    if k_hi < k_lo:
        return None
    band = corr[k_lo : k_hi + 1]

    band_idx = int(np.argmax(band))
    peak_idx = k_lo + band_idx
    peak_val = float(corr[peak_idx])
    if peak_val <= 0.0:
        return None  # no correlation energy in the allowed range (silence)

    # Band edge argmax cannot be trusted and the integer boundary should be 
    # considered the best in-band estimate in this case.
    at_band_edge = band_idx == 0 or band_idx == len(band) - 1
    do_interpolate = interpolate and not at_band_edge
    peak_pos = peak_idx + (_parabolic_peak_offset(corr, peak_idx) if do_interpolate else 0.0)
    lag_in_corr = peak_pos - zero_idx
    offset_samples = w_start - b_start + lag_in_corr
    offset_ms = offset_samples / sample_rate * 1000.0

    secondary = _find_secondary_peak(band, band_idx, min_distance=sample_rate // 2)
    confidence = min(peak_val / secondary, _MAX_CONFIDENCE) if secondary > 0 else _MAX_CONFIDENCE

    return (round(offset_ms, 2), round(confidence, 2))


def _scan_step_samples(window_samples: int) -> int:
    """Pass-1 scan step for a window size."""
    return window_samples // 2


def _n_scan_windows(window_samples: int, step_samples: int, total_samples: int) -> int:
    """Number of windows the pass-1 position loop generates for these sizes."""
    if window_samples // 4 > total_samples:
        return 0
    return (total_samples - window_samples // 4) // step_samples + 1


def _calibrate_window_s(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    locations: list[int],
    search_radius_samples: int,
    sample_rate: int,
    total_samples: int,
    anchor_offset_samples: int,
    verbose: bool,
) -> int | None:
    """Prescan: smallest candidate window size whose lock holds up across *locations*.

    Probes candidate sizes smallest-first at every location and returns the
    first whose median confidence clears the margin — sparse or quiet audio
    fails to lock at small sizes and pushes the choice up, while strong linear
    drift smears large windows' peaks and pushes it down. Candidates that
    would leave the coarse scan with too few windows for the linear fit are
    excluded. Returns None when no candidate qualifies (silent locations,
    unmatched content) or there is nothing to probe.
    """
    viable = [
        w
        for w in _PRESCAN_CANDIDATES_S
        if _n_scan_windows(
            int(w * sample_rate), _scan_step_samples(int(w * sample_rate)), total_samples
        )
        >= 6
    ]
    if not viable or not locations:
        return None
    workers = min(len(locations), os.cpu_count() or 4)
    for w in viable:
        win_samples = int(w * sample_rate)
        starts = [min(loc, max(0, total_samples - win_samples)) for loc in locations]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    correlate_window,
                    sig_a,
                    sig_b,
                    start,
                    win_samples,
                    search_radius_samples,
                    sample_rate,
                    True,
                    anchor_offset_samples,
                )
                for start in starts
            ]
            confidences: list[float] = []
            for fut in futures:
                result = fut.result()
                confidences.append(result[1] if result is not None else 0.0)
        median_conf = float(np.median(confidences))
        _log(f" + Prescan {w}s windows: median confidence {median_conf:.2f}", verbose)
        if median_conf >= _PRESCAN_MIN_CONFIDENCE:
            return w
    return None


def _repair_coarse(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    positions: list[int],
    results_by_pos: dict[int, tuple[float, float] | None],
    window_samples: int,
    total_samples: int,
    search_radius_samples: int,
    sample_rate: int,
    interpolate: bool,
    verbose: bool,
) -> int:
    """Re-search weak coarse windows anchored on confident neighbors.

    Pass 1 centers every window's search on a single global base offset, so a
    window whose true offset falls outside that band locks onto in-band noise
    (low confidence) even though a clean peak exists elsewhere. Here each weak
    window is re-correlated with the search band re-centered on a confident
    neighbor's offset, and the result is adopted only if the re-search itself
    clears the trust bar. Confident windows flood-fill outward into contiguous
    weak regions over repeated passes. Returns the number of windows repaired.
    """
    trust = _REPAIR_TRUST_CONFIDENCE

    def _solid(p: int) -> bool:
        r = results_by_pos.get(p)
        return r is not None and r[1] >= trust

    solid = {p: _solid(p) for p in positions}
    # Nothing to propagate from, or nothing left to propagate into.
    if not any(solid.values()) or all(solid.values()):
        return 0

    repaired = 0
    for _ in range(len(positions)):  # bounded flood-fill; one frontier step per pass
        improved = False
        for idx, p in enumerate(positions):
            if solid[p]:
                continue
            anchors_ms: list[float] = []
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(positions):
                    neighbor = results_by_pos[positions[j]]
                    if neighbor is not None and neighbor[1] >= trust:
                        anchors_ms.append(neighbor[0])
            if not anchors_ms:
                continue
            best: tuple[float, float] | None = None
            for off_ms in anchors_ms:
                anchor = int(round(off_ms / 1000.0 * sample_rate))
                res = correlate_window(
                    sig_a,
                    sig_b,
                    p,
                    min(window_samples, total_samples - p),
                    search_radius_samples,
                    sample_rate,
                    interpolate,
                    anchor,
                )
                if res is not None and res[1] >= trust and (best is None or res[1] > best[1]):
                    best = res
            if best is not None:
                results_by_pos[p] = best
                solid[p] = True
                improved = True
                repaired += 1
        if not improved:
            break
    return repaired


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
    sample_rate: int,
    interpolate: bool,
    anchor_offset_samples: int = 0,
    trend_slope_ms_per_s: float = 0.0,
    trend_intercept_ms: float = 0.0,
) -> float:
    """Binary-search *[t_start_s, t_end_s]* to pinpoint where the offset changes.

    When a linear trend is supplied, *offset_before_ms*/*offset_after_ms* are
    trend residuals and each probe measurement is detrended before comparing.
    """
    if t_end_s - t_start_s <= min_window_s:
        return (t_start_s + t_end_s) / 2.0

    t_mid = (t_start_s + t_end_s) / 2.0
    win_samples = max(
        int(min_window_s * sample_rate),
        int((t_end_s - t_start_s) / 4 * sample_rate),
    )
    # Center the probe on t_mid: a window *starting* at t_mid measures
    # [t_mid, t_mid+win] and would bias the search early by half a window.
    mid_start = max(0, int(t_mid * sample_rate) - win_samples // 2)

    result = correlate_window(
        sig_a,
        sig_b,
        mid_start,
        win_samples,
        search_radius_samples,
        sample_rate,
        interpolate,
        anchor_offset_samples,
    )
    if result is None:
        return (t_start_s + t_end_s) / 2.0

    mid_offset_ms, mid_confidence = result
    if mid_confidence < _CONFIDENCE_FLOOR:
        # A low-confidence probe carries no information; branching on it would
        # send the search irreversibly into the wrong half.
        return (t_start_s + t_end_s) / 2.0
    mid_offset_ms -= trend_intercept_ms + trend_slope_ms_per_s * t_mid

    if abs(mid_offset_ms - offset_before_ms) >= threshold_ms:
        return refine_change_point(
            sig_a,
            sig_b,
            t_start_s,
            t_mid,
            offset_before_ms,
            mid_offset_ms,
            min_window_s,
            search_radius_samples,
            threshold_ms,
            sample_rate,
            interpolate,
            anchor_offset_samples,
            trend_slope_ms_per_s,
            trend_intercept_ms,
        )
    return refine_change_point(
        sig_a,
        sig_b,
        t_mid,
        t_end_s,
        mid_offset_ms,
        offset_after_ms,
        min_window_s,
        search_radius_samples,
        threshold_ms,
        sample_rate,
        interpolate,
        anchor_offset_samples,
        trend_slope_ms_per_s,
        trend_intercept_ms,
    )


def _fit_linear_drift(coarse: list[dict], threshold_ms: float) -> dict | None:
    """Robust linear model of offset vs. time over the coarse scan, or None.

    The slope is the median of adjacent-pair slopes: a step change contributes
    a single outlier pair, so steps and ramps cannot masquerade as each other.
    The fit is significant when the slope clears its own standard error and
    the drift accumulated over the scanned span clears *threshold_ms*.

    ``speed_ratio`` is seconds of file2 per second of file1 (< 1 means file2
    plays the same content in less time).
    """
    if len(coarse) < 6:
        return None
    t = np.array([p["timestamp_s"] for p in coarse])
    y = np.array([p["offset_ms"] for p in coarse])
    slopes = np.diff(y) / np.diff(t)  # ms/s; window positions strictly increase
    slope = float(np.median(slopes))
    span = float(t[-1] - t[0])
    if abs(slope) * span < threshold_ms:
        return None
    # Standard error of the median, sigma estimated from the MAD.
    mad = float(np.median(np.abs(slopes - slope)))
    se = 1.2533 * (1.4826 * mad) / float(np.sqrt(len(slopes)))
    if abs(slope) < 3.0 * se:
        return None
    intercept = float(np.median(y - slope * t))
    return {
        "slope_ms_per_s": slope,
        "intercept_ms": intercept,
        "slope_ms_per_min": round(slope * 60.0, 2),
        "speed_ratio": round(1.0 - slope / 1000.0, 6),
        "total_drift_ms": round(slope * span, 1),
    }


def _resolve_threshold_ms(coarse: list[dict], fps: float | None, verbose: bool) -> int:
    """Auto change threshold: max of the scan's noise floor and one frame, clamped.

    The noise floor is a MAD-based sigma of adjacent-window offset differences
    with the median difference removed first — a linear ramp shifts every
    difference by the same amount and must not inflate the estimate, while
    step changes contribute outlier pairs the MAD resists. Confidence-gated
    points are preferred so garbage windows don't set the threshold.
    """
    lo_ms, hi_ms = _AUTO_THRESHOLD_RANGE_MS
    pts = [p for p in coarse if p["confidence"] >= _CONFIDENCE_FLOOR]
    if len(pts) < 4:
        pts = coarse
    noise_ms = 0.0
    if len(pts) >= 4:
        diffs = np.diff([p["offset_ms"] for p in pts])
        sigma = 1.4826 * float(np.median(np.abs(diffs - np.median(diffs))))
        noise_ms = _AUTO_THRESHOLD_NOISE_MULT * sigma
    frame_ms = 1000.0 / fps if fps else 0.0
    _log(
        f" + Threshold evidence: noise floor {noise_ms:.1f} ms, frame floor {frame_ms:.1f} ms",
        verbose,
    )
    return int(round(min(max(noise_ms, frame_ms, lo_ms), hi_ms)))


def drift_analysis(
    sig_a: np.ndarray,
    sig_b: np.ndarray,
    window_s: int | None,
    threshold_ms: int | None,
    max_drift_s: int | None,
    fps: float | None,
    sample_rate: int,
    prescan: bool,
    verbose: bool,
) -> dict:
    """Three-pass drift detection: coarse scan, change-point refinement, summary.

    ``window_s``, ``threshold_ms``, and ``max_drift_s`` accept None, meaning
    auto: the search radius comes from the base-offset probes' dispersion, the
    window from the prescan calibration (requested via *prescan*) or the fixed
    default, and the threshold from the coarse scan's own noise floor after
    pass 1. Explicit values bypass resolution entirely.
    """
    # Sub-sample peak interpolation is essentially free, so it is always on.
    interpolate = True
    min_window_s = 2.0
    total_samples = min(len(sig_a), len(sig_b))
    total_duration_s = total_samples / sample_rate
    # Where each parameter came from; "window" may flip to "prescan" below.
    sources = {
        "window": "explicit" if window_s is not None else "default",
        "threshold": "explicit" if threshold_ms is not None else "auto",
        "radius": "explicit" if max_drift_s is not None else "auto",
    }
    # The probe-stage margin needs a threshold floor before the auto value can
    # exist (it is derived from the coarse scan itself), so it falls back to
    # the old fixed default until then.
    margin_floor_ms = float(threshold_ms) if threshold_ms is not None else _DEFAULT_THRESHOLD_MS

    # Global anchor: centers every window search, so the search radius bounds
    # drift around the base offset instead of bounding the offset itself.
    # Estimated as the median of a few short decimated probe windows searched
    # over the whole file. The probes use *plain* correlation: its broad ridge
    # integrates across the smear a drifting source produces, where a whitened
    # full-search peak splinters into needles and loses to noise. The median is
    # robust to steps and to probes landing in silent or unmatched content.
    _log("Estimating global base offset …", verbose)
    # Anti-aliased decimation is essential here: alias products decorrelate
    # under drift and bury the probe peak (and dropping the stretch-sensitive
    # high band actively helps).
    dec = max(1, sample_rate // 8000)
    if dec > 1:
        a_dec = np.asarray(resample_poly(sig_a, 1, dec), dtype=np.float32)
        b_dec = np.asarray(resample_poly(sig_b, 1, dec), dtype=np.float32)
    else:
        a_dec, b_dec = sig_a, sig_b
    dec_rate = sample_rate // dec
    n_probes = 5
    probe_win = min(30 * dec_rate, len(a_dec))
    probe_span = max(1, len(a_dec) - probe_win)
    probe_offsets: list[float] = []
    probe_positions: list[int] = []  # full-rate window starts, reused by the prescan
    probes_dropped = 0
    for i in range(n_probes):
        p = int(probe_span * (i + 0.5) / n_probes)
        a_win = a_dec[p : p + probe_win]
        if len(b_dec) < len(a_win):
            continue
        # The prescan can still calibrate here even if the gate below drops the
        # probe.
        probe_positions.append(p * dec)
        corr = np.abs(_phat_correlate(a_win, b_dec, rho=0.0))
        peak_idx = int(np.argmax(corr))
        peak_val = float(corr[peak_idx])
        # Gate each probe on its peak-to-sidelobe ratio: an unanchored full-file
        # argmax over hours of audio can tie with noise, and one spurious lock
        # would blow up the dispersion evidence (the median survives an outlier;
        # the max-deviation and max-adjacent-diff terms do not).
        secondary = _find_secondary_peak(corr, peak_idx, min_distance=dec_rate // 2)
        confidence = (
            min(peak_val / secondary, _MAX_CONFIDENCE) if secondary > 0 else _MAX_CONFIDENCE
        )
        if peak_val <= 0.0 or confidence < _CONFIDENCE_FLOOR:
            probes_dropped += 1
            continue
        lag = peak_idx - (len(b_dec) - 1)
        probe_offsets.append((p + lag) / dec_rate * 1000.0)
    if probes_dropped:
        log.warn(
            f"discarded {probes_dropped} of {n_probes} base-offset probe(s) "
            f"with no usable alignment."
        )
    if probe_offsets:
        base_offset_ms = float(np.median(probe_offsets))
    else:
        base_offset_ms = 0.0
        log.warn("could not estimate a base offset; drift results may be unreliable.")
    anchor_samples = int(round(base_offset_ms / 1000.0 * sample_rate))
    _log(f" + Base offset {base_offset_ms:+.1f} ms ({len(probe_offsets)} probes)", verbose)

    # Probe dispersion drives the search radius. In auto mode it *is* the
    # radius: dispersion plus a margin, floored. With an explicit --max-drift
    # it can only widen it — a clamped band would force every out-of-band
    # window onto a garbage in-band peak. The margin is the largest
    # adjacent-probe jump, covering interpolation between probes and
    # extrapolation past the file ends at the drift rate already observed.
    deviation_ms = margin_ms = 0.0
    if len(probe_offsets) >= 2:
        deviation_ms = float(np.max(np.abs(np.array(probe_offsets) - base_offset_ms)))
        margin_ms = max(float(np.max(np.abs(np.diff(probe_offsets)))), margin_floor_ms)
    if max_drift_s is not None:
        search_radius_samples = int(max_drift_s * sample_rate)
        if len(probe_offsets) >= 2 and deviation_ms + margin_ms > max_drift_s * 1000.0:
            needed_ms = deviation_ms + margin_ms
            search_radius_samples = int(needed_ms / 1000.0 * sample_rate)
            log.warn(
                f"offsets spread up to {deviation_ms / 1000.0:.1f}s around the base, "
                f"beyond --max-drift {max_drift_s}s; widening search radius "
                f"to {needed_ms / 1000.0:.1f}s."
            )
    elif len(probe_offsets) >= 2:
        _log(
            f" + Radius evidence: probe dispersion {deviation_ms:.0f} ms, "
            f"margin {margin_ms:.0f} ms"
            + (f", {probes_dropped} probe(s) dropped" if probes_dropped else ""),
            verbose,
        )
        radius_ms = max(deviation_ms + margin_ms, _AUTO_RADIUS_FLOOR_S * 1000.0)
        if radius_ms > _AUTO_RADIUS_CEIL_S * 1000.0:
            log.warn(
                f"probe evidence asked for a ±{radius_ms / 1000.0:.0f}s search radius; "
                f"capping at ±{_AUTO_RADIUS_CEIL_S:g}s to bound memory. "
                f"Pass --max-drift to search wider."
            )
            radius_ms = _AUTO_RADIUS_CEIL_S * 1000.0
        search_radius_samples = int(radius_ms / 1000.0 * sample_rate)
    else:
        search_radius_samples = int(_DEFAULT_RADIUS_S * sample_rate)
    search_radius_s = round(search_radius_samples / sample_rate, 1)
    _log(
        f"Search radius: ±{search_radius_s:g}s"
        + (" (auto)" if sources["radius"] == "auto" else ""),
        verbose,
    )

    # Drift window: explicit > prescan calibration > fixed default. The
    # calibration reuses the probe locations and the just-resolved radius.
    if window_s is not None:
        if prescan:
            _announce("Explicit --drift-window given; skipping the prescan calibration.")
    elif prescan:
        _log("Prescan: calibrating the window size …", verbose)
        window_s = _calibrate_window_s(
            sig_a,
            sig_b,
            probe_positions,
            search_radius_samples,
            sample_rate,
            total_samples,
            anchor_samples,
            verbose,
        )
        if window_s is not None:
            sources["window"] = "prescan"
        else:
            window_s = _DEFAULT_WINDOW_S
            _announce("Prescan calibration inconclusive; using the default window.")
    else:
        window_s = _DEFAULT_WINDOW_S
    window_samples = int(window_s * sample_rate)
    step_samples = _scan_step_samples(window_samples)
    _log(
        f"Drift window: {window_s}s" + (" (prescan)" if sources["window"] == "prescan" else ""),
        verbose,
    )

    # Pass 1: coarse windowed cross-correlation (parallel across windows)
    _log("Pass 1: coarse windowed cross-correlation …", verbose)
    positions: list[int] = []
    pos = 0
    while pos + window_samples // 4 <= total_samples:
        positions.append(pos)
        pos += step_samples

    results_by_pos: dict[int, tuple[float, float] | None] = {}
    workers = min(len(positions), os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(
                correlate_window,
                sig_a,
                sig_b,
                p,
                min(window_samples, total_samples - p),
                search_radius_samples,
                sample_rate,
                interpolate,
                anchor_samples,
            ): p
            for p in positions
        }
        completed = 0
        for fut in as_completed(futures):
            p = futures[fut]
            results_by_pos[p] = fut.result()
            completed += 1
            _log(f"Analyzed window {completed}/{len(positions)}", verbose)
            if not verbose:
                log.progress(completed, len(positions), "Scanning windows")
    if not verbose:
        log.progress_clear()

    # Repair windows whose true offset fell outside the global anchor's band by
    # re-searching them centered on confident neighbors (continuity). This is
    # what rescues large staircase drifts whose extremes the single base-offset
    # anchor plus a symmetric radius cannot all reach at once.
    repaired = _repair_coarse(
        sig_a,
        sig_b,
        positions,
        results_by_pos,
        window_samples,
        total_samples,
        search_radius_samples,
        sample_rate,
        interpolate,
        verbose,
    )
    if repaired:
        _log(f" + Repaired {repaired} weak window(s) by re-anchoring on neighbors", verbose)

    coarse: list[dict] = []
    for p in positions:
        result = results_by_pos.get(p)
        if result is not None:
            offset_ms, confidence = result
            # Label each measurement by its window *center* — a window
            # starting at t characterizes [t, t+win], not t itself — and keep
            # the span for change-point overlap tests.
            eff_len = min(window_samples, total_samples - p)
            coarse.append(
                {
                    "timestamp_s": (p + eff_len / 2) / sample_rate,
                    "window_start_s": p / sample_rate,
                    "window_end_s": (p + eff_len) / sample_rate,
                    "offset_ms": offset_ms,
                    "confidence": confidence,
                }
            )

    if not coarse:
        return {
            "segments": [],
            "change_points": [],
            "no_drift": True,
            "linear_drift": None,
            "base_offset_ms": base_offset_ms,
            "window_s": window_s,
            "threshold_ms": (
                threshold_ms if threshold_ms is not None else int(_DEFAULT_THRESHOLD_MS)
            ),
            "search_radius_s": search_radius_s,
            "param_sources": sources,
        }

    # Filter outlier windows whose offset is far from both neighbors
    if len(coarse) > 2:
        max_jump_ms = search_radius_samples / sample_rate * 1000
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

    # Change threshold: explicit > the scan's own noise floor vs. one frame.
    # Resolvable only now — the coarse offsets are its evidence.
    if threshold_ms is None:
        threshold_ms = _resolve_threshold_ms(coarse, fps, verbose)
    refine_threshold_ms = float(threshold_ms)
    _log(
        f"Change threshold: {threshold_ms}ms"
        + (" (auto)" if sources["threshold"] == "auto" else ""),
        verbose,
    )

    # Linear-drift fit: ramps (clock skew, PAL speedup) cannot be represented
    # by the step model, so detect them first and run step detection on the
    # residuals.
    linear = _fit_linear_drift(coarse, float(threshold_ms))
    slope_ms_per_s = linear["slope_ms_per_s"] if linear else 0.0
    intercept_ms = linear["intercept_ms"] if linear else 0.0
    if linear:
        _log(
            f" + Linear drift {linear['slope_ms_per_min']:+.2f} ms/min "
            f"(speed ratio {linear['speed_ratio']:.6f})",
            verbose,
        )

    def _trend_at(t_s: float) -> float:
        return intercept_ms + slope_ms_per_s * t_s

    def _resid(pt: dict) -> float:
        return pt["offset_ms"] - _trend_at(pt["timestamp_s"])

    # Gate no-information windows out of step detection and segment levels.
    # The linear fit above deliberately ran ungated — its median is robust,
    # and dropping smeared-but-honest windows starves it — but a garbage
    # offset crossing the threshold here would fabricate a change point.
    usable = [p for p in coarse if p["confidence"] >= _CONFIDENCE_FLOOR]
    if len(usable) < len(coarse):
        _log(f" + Dropped {len(coarse) - len(usable)} low-confidence window(s)", verbose)
    if len(usable) <= len(coarse) // 2:
        log.warn(
            f"only {len(usable)} of {len(coarse)} windows had a usable alignment; "
            f"results may be unreliable."
        )
    if usable:
        coarse = usable

    # Pass 2: change-point detection & refinement on the (de)trended offsets
    _log("Pass 2: detecting change points …", verbose)
    step_s = step_samples / sample_rate
    change_points: list[dict] = []
    for i in range(1, len(coarse)):
        prev, curr = coarse[i - 1], coarse[i]
        prev_resid, curr_resid = _resid(prev), _resid(curr)
        if abs(curr_resid - prev_resid) >= threshold_ms:
            # With center-labeled windows the change lies between the two
            # centers, give or take half a scan step.
            t_start = max(0.0, prev["timestamp_s"] - step_s / 2)
            t_end = min(curr["timestamp_s"] + step_s / 2, total_duration_s)
            # Anchor the refinement probes on this boundary's own two levels,
            # not the global base offset: a change into or out of a repaired
            # segment lies outside the global band, so a globally anchored probe
            # could not lock either side and would stall at the bracket midpoint.
            cp_anchor = int(round((prev["offset_ms"] + curr["offset_ms"]) / 2000.0 * sample_rate))
            _log(
                f" + Refining change between {log.format_timestamp(t_start)} "
                f"and {log.format_timestamp(t_end)} …",
                verbose,
            )
            cp_t = refine_change_point(
                sig_a,
                sig_b,
                t_start,
                t_end,
                prev_resid,
                curr_resid,
                min_window_s,
                search_radius_samples,
                refine_threshold_ms,
                sample_rate,
                interpolate,
                cp_anchor,
                slope_ms_per_s,
                intercept_ms,
            )
            change_points.append(
                {
                    "timestamp_s": cp_t,
                    "offset_before_ms": round(_trend_at(cp_t) + prev_resid, 2),
                    "offset_after_ms": round(_trend_at(cp_t) + curr_resid, 2),
                    "resid_before_ms": prev_resid,
                    "resid_after_ms": curr_resid,
                }
            )

    change_points.sort(key=lambda cp: cp["timestamp_s"])

    # Pass 3: compile segments. Each segment is a residual level over the
    # trend; absolute offsets are reconstructed at the segment edges so a
    # drifting segment reports its start and end values.
    _log("Pass 3: compiling segments …", verbose)

    def _avg(pts: list[dict], key: str) -> float:
        return sum(p[key] for p in pts) / len(pts) if pts else 0.0

    def _level(pts: list[dict], fallback: float) -> float:
        """Confidence-weighted mean residual, so shaky windows count less."""
        wsum = sum(p["confidence"] for p in pts)
        if wsum <= 0.0:
            return fallback
        return sum(_resid(p) * p["confidence"] for p in pts) / wsum

    def _make_segment(start_s: float, end_s: float, level: float, conf: float) -> dict:
        seg = {
            "start_s": start_s,
            "end_s": end_s,
            "offset_ms": round(level + _trend_at((start_s + end_s) / 2.0), 1),
            "confidence": round(conf, 2),
        }
        if linear:
            seg["offset_start_ms"] = round(level + _trend_at(start_s), 1)
            seg["offset_end_ms"] = round(level + _trend_at(end_s), 1)
        return seg

    if not change_points:
        return {
            "segments": [
                _make_segment(
                    0.0, total_duration_s, _level(coarse, 0.0), _avg(coarse, "confidence")
                )
            ],
            "change_points": [],
            "no_drift": linear is None,
            "linear_drift": linear,
            "base_offset_ms": base_offset_ms,
            "window_s": window_s,
            "threshold_ms": threshold_ms,
            "search_radius_s": search_radius_s,
            "param_sources": sources,
        }

    transition_half = 1.5  # seconds each side of a change point
    segments: list[dict] = []
    seg_start = 0.0
    cp_times = [cp["timestamp_s"] for cp in change_points]

    def _members(lo: float, hi: float) -> list[dict]:
        """Coarse points centered in [lo, hi) whose window straddles no change point.

        A 30 s window reaching across a change point reports the dominant
        side's offset and would poison the other side's mean.
        """
        return [
            r
            for r in coarse
            if lo <= r["timestamp_s"] < hi
            and not any(r["window_start_s"] < t < r["window_end_s"] for t in cp_times)
        ]

    for cp in change_points:
        cp_t = cp["timestamp_s"]
        seg_end = max(seg_start, cp_t - transition_half)

        if seg_end > seg_start:
            pts = _members(seg_start, seg_end)
            segments.append(
                _make_segment(
                    seg_start,
                    seg_end,
                    _level(pts, cp["resid_before_ms"]),
                    _avg(pts, "confidence") if pts else 0.0,
                )
            )

        trans_end = max(seg_end, min(cp_t + transition_half, total_duration_s))
        segments.append(
            {
                "start_s": seg_end,
                "end_s": trans_end,
                "offset_ms": None,
                "confidence": None,  # transition
            }
        )
        seg_start = trans_end

    if seg_start < total_duration_s:
        pts = _members(seg_start, float("inf"))
        segments.append(
            _make_segment(
                seg_start,
                total_duration_s,
                _level(pts, change_points[-1]["resid_after_ms"]),
                _avg(pts, "confidence") if pts else 0.0,
            )
        )

    return {
        "segments": segments,
        "change_points": change_points,
        "no_drift": False,
        "linear_drift": linear,
        "base_offset_ms": base_offset_ms,
        "window_s": window_s,
        "threshold_ms": threshold_ms,
        "search_radius_s": search_radius_s,
        "param_sources": sources,
    }
