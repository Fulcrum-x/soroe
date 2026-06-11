"""Batch directory mode for soroe.

Pairs files between two directories by their ``SxxEyy`` token (case-insensitive)
and runs the standard single-shot or drift pipeline on each pair. Pairs run
sequentially; each pair's two audio extractions still parallelize internally.
"""

from __future__ import annotations

import argparse
import os
import re

from . import log
from .errors import SoroeError
from .pipeline import run_drift, run_single_shot

# Matches S00E00 / s1e4 / etc. Capture groups are the season and episode numbers
# so we can normalize to a canonical SxxEyy form regardless of input casing or padding.
TOKEN_RE = re.compile(r"[Ss](\d{1,2})[Ee](\d{1,2})")


def _extract_token(filename: str) -> str | None:
    m = TOKEN_RE.search(filename)
    if not m:
        return None
    return f"S{int(m.group(1)):02d}E{int(m.group(2)):02d}"


def _scan_dir(path: str) -> tuple[dict[str, str], int]:
    """Return ``(token_to_path, no_token_count)`` for files in *path* (top-level only).

    Duplicate tokens within a directory: keep the alphabetically-first file, warn on the rest.
    """
    if not os.path.isdir(path):
        raise SoroeError(f"directory not found: {path}")

    entries = sorted(os.listdir(path))
    result: dict[str, str] = {}
    no_token = 0

    for name in entries:
        full = os.path.join(path, name)
        if not os.path.isfile(full):
            continue
        token = _extract_token(name)
        if token is None:
            log.warn(f"no SxxEyy token in {name}; skipping")
            no_token += 1
            continue
        if token in result:
            log.warn(
                f"duplicate {token} in {path}: "
                f"keeping {os.path.basename(result[token])}, skipping {name}"
            )
            continue
        result[token] = full

    return result, no_token


def _pair_directories(
    dir_a: str,
    dir_b: str,
) -> tuple[list[tuple[str, str, str]], dict[str, int]]:
    """Pair files between two directories by SxxEyy token.

    Returns ``(pairs, skipped_counts)`` where ``pairs`` is a list of
    ``(token, path_in_a, path_in_b)`` sorted by token, and ``skipped_counts``
    records why files were excluded.
    """
    a_map, a_no_token = _scan_dir(dir_a)
    b_map, b_no_token = _scan_dir(dir_b)

    tokens_a, tokens_b = set(a_map), set(b_map)
    common = sorted(tokens_a & tokens_b)
    unmatched_a = sorted(tokens_a - tokens_b)
    unmatched_b = sorted(tokens_b - tokens_a)

    for token in unmatched_a:
        log.warn(f"{token}: no counterpart in {dir_b}; skipping {os.path.basename(a_map[token])}")
    for token in unmatched_b:
        log.warn(f"{token}: no counterpart in {dir_a}; skipping {os.path.basename(b_map[token])}")

    pairs = [(t, a_map[t], b_map[t]) for t in common]
    skipped = {
        "no_token": a_no_token + b_no_token,
        "unmatched": len(unmatched_a) + len(unmatched_b),
    }
    return pairs, skipped


def _format_pair_header(token: str, name_a: str, name_b: str) -> str:
    """Greppable per-pair header line printed to stdout above each pair's result."""
    return log.bold(f"[{token}] ") + f"{name_a}  <->  {name_b}"


def run(args: argparse.Namespace) -> None:
    """Drive the batch loop. Pairs run sequentially; failures are per-pair, not fatal."""
    pairs, skipped = _pair_directories(args.file1, args.file2)

    if not pairs:
        raise SoroeError("no matching SxxEyy pairs found between the two directories.")

    log.info(f"Found {len(pairs)} matching pair{'s' if len(pairs) != 1 else ''}")

    succeeded = 0
    failed = 0

    for i, (token, path_a, path_b) in enumerate(pairs, 1):
        name_a = os.path.basename(path_a)
        name_b = os.path.basename(path_b)
        log.info(f"Processing {token} ({i}/{len(pairs)}): {name_a} <-> {name_b}")

        try:
            if args.drift:
                output = run_drift(
                    path_a,
                    path_b,
                    duration=args.duration,
                    audio_track=args.audio_track,
                    drift_window=args.drift_window,
                    drift_threshold=args.drift_threshold,
                    max_drift=args.max_drift,
                    prescan=args.prescan,
                    verbose=args.verbose,
                    save_offsets_as=token if args.saveoffsets else None,
                )
            else:
                output = run_single_shot(
                    path_a,
                    path_b,
                    duration=args.duration,
                    audio_track=args.audio_track,
                    verbose=args.verbose,
                )
        except SoroeError as e:
            log.error(str(e))
            log.warn(f"{token}: skipping pair")
            failed += 1
            continue

        print(_format_pair_header(token, name_a, name_b))
        print(output)
        print()
        succeeded += 1

    parts = [f"{succeeded} pair{'s' if succeeded != 1 else ''} run"]
    total_skipped = failed + skipped["unmatched"] + skipped["no_token"]
    if total_skipped:
        reasons: list[str] = []
        if failed:
            reasons.append(f"{failed} failed")
        if skipped["unmatched"]:
            reasons.append(f"{skipped['unmatched']} unmatched")
        if skipped["no_token"]:
            reasons.append(f"{skipped['no_token']} no token")
        parts.append(f"{total_skipped} skipped ({', '.join(reasons)})")
    log.info("Batch complete: " + ", ".join(parts) + ".")
