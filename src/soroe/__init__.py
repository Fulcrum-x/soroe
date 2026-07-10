"""
soroe - find the offset between two audio/video files of the same content.

Determines how many milliseconds (and frames, given a known framerate) one file
is offset from the other, so you can mux tracks between them with the correct delay.

Supports any format ffmpeg can read (mkv, m4a, eac3, mka, mks, mp4, flac, wav, …).

Requires: numpy, scipy, ffmpeg (on PATH)

Usage:
    soroe file1.mkv file2.mkv
    soroe file1.mkv file2.mkv --duration 300 --verbose
    soroe file1.flac file2.flac --drift --drift-threshold 80
"""

from __future__ import annotations

__version__ = "1.0.2"
