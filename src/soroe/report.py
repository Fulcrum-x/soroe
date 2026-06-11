"""Human-readable stdout formatting for single-shot and drift results."""

from __future__ import annotations

import os
import re

from . import log


# Single-shot formatting
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


def _confidence_colored(ratio: float) -> str:
    label = _confidence_label(ratio)
    color = {
        "excellent": log.green,
        "good": log.green,
        "fair": log.yellow,
        "poor": log.red,
        "unreliable": log.red,
    }[label]
    return color(label)


def _header(title: str) -> list[str]:
    text = f"soroe · {title}"
    return [log.bold(text), log.dim("─" * len(text))]


def _kv(key: str, value: str, note: str = "") -> str:
    line = f"  {log.dim(key.ljust(12))}{value}"
    if note:
        line += "  " + log.dim(note)
    return line


def _fmt_offset(ms: float) -> str:
    sign = "+" if ms >= 0 else ""
    return f"{sign}{ms:.0f} ms"


def format_result(result: dict) -> str:
    lines: list[str] = _header(result["method"])
    lines.append("")

    ms = result["offset_ms"]
    if ms >= 0:
        desc = f"file2 delayed by {ms:.0f} ms"
    else:
        desc = f"file1 delayed by {-ms:.0f} ms"
    lines.append(_kv("Offset", log.bold(_fmt_offset(ms)), desc))

    if "fps" in result:
        fps = result["fps"]
        frames = result["offset_frames"]
        nearest = result["nearest_frame"]
        sign_f = "+" if frames >= 0 else ""
        lines.append(
            _kv(
                "Frames",
                log.bold(f"{sign_f}{frames:.2f}"),
                f"@ {fps:.3f} fps · nearest: {nearest}",
            )
        )

    conf = result["confidence"]
    lines.append(
        _kv(
            "Confidence",
            _confidence_colored(conf),
            f"{conf:.2f}x peak-to-sidelobe",
        )
    )

    return "\n".join(lines)


# Drift formatting
def _param_note(sources: dict, key: str) -> str:
    """Suffix for an auto-resolved parameter; empty for explicit/default values."""
    src = sources.get(key)
    return f" ({src})" if src in ("auto", "prescan") else ""


def _format_segment(seg: dict, fps: float | None) -> str:
    ts0 = log.format_timestamp(seg["start_s"])
    ts1 = log.format_timestamp(seg["end_s"])
    time_range = log.dim(f"{ts0} → {ts1}")

    if seg["offset_ms"] is None:
        return f"    {time_range}  {log.yellow('transition')}"

    conf_val = seg["confidence"]
    conf_str = f"{_confidence_colored(conf_val)} {log.dim(f'({conf_val:.2f}x)')}"

    if "offset_start_ms" in seg:
        # Segment rides a linear drift: report its start -> end offsets.
        off0, off1 = seg["offset_start_ms"], seg["offset_end_ms"]
        offset_str = log.bold(f"{_fmt_offset(off0)} → {_fmt_offset(off1)}".rjust(20))
        parts = [f"    {time_range}", offset_str, conf_str]
        if fps is not None:
            f0 = off0 / 1000.0 * fps
            f1 = off1 / 1000.0 * fps
            parts.append(log.dim(f"{f0:+.1f} → {f1:+.1f} frames"))
        return "  ".join(parts)

    off = seg["offset_ms"]
    offset_str = log.bold(_fmt_offset(off).rjust(8))
    parts = [f"    {time_range}", offset_str, conf_str]
    if fps is not None:
        frames = off / 1000.0 * fps
        sign = "+" if frames >= 0 else ""
        parts.append(log.dim(f"{sign}{frames:.1f} frames"))
    return "  ".join(parts)


def _format_change_point(cp: dict, fps: float | None) -> str:
    ts = log.format_timestamp(cp["timestamp_s"])
    before = cp["offset_before_ms"]
    after = cp["offset_after_ms"]
    entry = (
        f"    {log.dim(ts)}  "
        f"{log.bold(_fmt_offset(before))} {log.dim('→')} {log.bold(_fmt_offset(after))}"
    )
    if fps is not None:
        fb = before / 1000.0 * fps
        fa = after / 1000.0 * fps
        entry += "  " + log.dim(f"({fb:+.1f} → {fa:+.1f} frames)")
    return entry


def format_drift_result(result: dict, fps: float | None) -> str:
    """Format drift-analysis *result* for human consumption."""
    lines: list[str] = _header("drift analysis")
    sources = result.get("param_sources", {})
    w = result["window_s"]
    thr = result["threshold_ms"]
    params = (
        f"  {w}s windows{_param_note(sources, 'window')}"
        f" · {thr}ms threshold{_param_note(sources, 'threshold')}"
    )
    radius = result.get("search_radius_s")
    if radius is not None:
        params += f" · search radius {radius:g}s{_param_note(sources, 'radius')}"
    base = result.get("base_offset_ms")
    if base is not None:
        params += f" · base offset {_fmt_offset(base)}"
    lines.append(log.dim(params))
    lines.append("")

    if result["no_drift"]:
        lines.append(f"  {log.green('No drift detected.')}")
        lines.append("")

    lin = result.get("linear_drift")
    if lin:
        rate = f"{lin['slope_ms_per_min']:+.2f} ms/min"
        detail = (
            f"speed ratio {lin['speed_ratio']:.6f} (file2 s per file1 s) · "
            f"{lin['total_drift_ms']:+.0f} ms over the scan"
        )
        lines.append(f"  {log.bold('Linear drift')}")
        lines.append(f"    {log.bold(rate)}  {log.dim(detail)}")
        lines.append("")

    lines.append(f"  {log.bold('Segments')}")
    for seg in result["segments"]:
        lines.append(_format_segment(seg, fps))

    if result["change_points"]:
        lines.append("")
        lines.append(f"  {log.bold('Change points')}")
        for cp in result["change_points"]:
            lines.append(_format_change_point(cp, fps))

    lines.append("")
    lines.append(f"  {log.bold('Summary')}")
    non_transition = [s for s in result["segments"] if s["offset_ms"] is not None]
    lines.append(f"    {log.dim('Segments'.ljust(16))}{len(non_transition)}")
    lines.append(f"    {log.dim('Change points'.ljust(16))}{len(result['change_points'])}")
    if lin:
        ratio = f"{lin['slope_ms_per_min']:+.2f} ms/min (x{lin['speed_ratio']:.6f})"
        lines.append(f"    {log.dim('Linear drift'.ljust(16))}{log.bold(ratio)}")
    if result["change_points"]:
        max_delta = max(
            abs(cp["offset_after_ms"] - cp["offset_before_ms"]) for cp in result["change_points"]
        )
        label = "Max step" if lin else "Max drift"
        lines.append(f"    {log.dim(label.ljust(16))}{log.bold(f'{max_delta:.0f} ms')}")

    return "\n".join(lines)


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def format_offsets_for_save(result: dict, fps: float | None) -> str:
    """Plain-text Segments/Change points sections of a drift *result*, for --saveoffsets."""
    lines: list[str] = []
    lin = result.get("linear_drift")
    if lin:
        lines.append(
            f"  Linear drift: {lin['slope_ms_per_min']:+.2f} ms/min, "
            f"speed ratio {lin['speed_ratio']:.6f}, "
            f"{lin['total_drift_ms']:+.0f} ms over the scan"
        )
        lines.append("")
    lines.append("  Segments")
    for seg in result["segments"]:
        lines.append(_strip_ansi(_format_segment(seg, fps)))

    if result["change_points"]:
        lines.append("")
        lines.append("  Change points")
        for cp in result["change_points"]:
            lines.append(_strip_ansi(_format_change_point(cp, fps)))

    return "\n".join(lines)


def save_offsets(result: dict, fps: float | None, name: str) -> str:
    """Write the Segments/Change points sections of a drift *result* to ./offsets/<name>.txt.

    Returns the path written.
    """
    out_dir = "offsets"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(format_offsets_for_save(result, fps) + "\n")
    return path
