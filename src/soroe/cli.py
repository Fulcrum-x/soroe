"""Command-line entry point: argument parsing and dispatch to the pipelines."""

from __future__ import annotations

import argparse
import os
import sys

from . import __version__, log
from .errors import SoroeError
from .pipeline import run_drift, run_single_shot


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="soroe",
        description=f"soroe {__version__} - find temporal offsets between audio/video files.",
    )
    parser.add_argument("file1", help="First audio/video file or directory")
    parser.add_argument("file2", help="Second audio/video file or directory")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help=f"show soroe {__version__} and exit",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="SEC",
        help="Seconds of content to analyze (default: 600; full file with --drift)",
    )
    parser.add_argument(
        "--audio-track",
        type=int,
        default=0,
        metavar="N",
        help="Audio track index to use (default: 0)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress and debug info"
    )
    parser.add_argument(
        "--drift", action="store_true", help="Enable drift/divergence detection mode"
    )
    parser.add_argument(
        "--drift-window",
        type=int,
        default=None,
        metavar="SEC",
        help="Window size in seconds for drift analysis (default: 30; see --prescan)",
    )
    parser.add_argument(
        "--drift-threshold",
        type=int,
        default=None,
        metavar="MS",
        help="Minimum offset change in ms to count as a change point "
        "(default: auto from scan noise and framerate; previously fixed 70)",
    )
    parser.add_argument(
        "--max-drift",
        type=int,
        default=None,
        metavar="SEC",
        help="Maximum expected drift in seconds around the detected base offset "
        "(default: auto from the base-offset probes; previously fixed 5)",
    )
    parser.add_argument(
        "--prescan",
        action="store_true",
        help="Calibrate the drift window size from the content before scanning (requires --drift)",
    )
    parser.add_argument(
        "--saveoffsets",
        "-so",
        action="store_true",
        help="Save drift segments/change points to ./offsets/<name>.txt (requires --drift)",
    )
    args = parser.parse_args()

    if args.saveoffsets and not args.drift:
        parser.error("--saveoffsets requires --drift")
    if args.prescan and not args.drift:
        parser.error("--prescan requires --drift")

    a_is_dir = os.path.isdir(args.file1)
    b_is_dir = os.path.isdir(args.file2)

    try:
        if a_is_dir != b_is_dir:
            raise SoroeError("both arguments must be files, or both must be directories.")
        if a_is_dir:
            from . import batch

            batch.run(args)
        else:
            from .batch import extract_token

            token1 = extract_token(os.path.basename(args.file1))
            token2 = extract_token(os.path.basename(args.file2))
            if token1 and token2 and token1 != token2:
                log.warn(
                    f"Episode mismatch: {os.path.basename(args.file1)} is {token1}, "
                    f"{os.path.basename(args.file2)} is {token2}; continuing anyway"
                )
            if args.drift:
                print(
                    run_drift(
                        args.file1,
                        args.file2,
                        duration=args.duration,
                        audio_track=args.audio_track,
                        drift_window=args.drift_window,
                        drift_threshold=args.drift_threshold,
                        max_drift=args.max_drift,
                        prescan=args.prescan,
                        verbose=args.verbose,
                        save_offsets_as="offsets" if args.saveoffsets else None,
                    )
                )
            else:
                print(
                    run_single_shot(
                        args.file1,
                        args.file2,
                        duration=args.duration,
                        audio_track=args.audio_track,
                        verbose=args.verbose,
                    )
                )
    except SoroeError as e:
        log.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
