# soroe

Finds the temporal offset between two audio/video files of the same content, including sources that drift or diverge over time.

soroe cross-correlates the audio (GCC-PHAT) and reports the shift in milliseconds and frames, either as one global offset or as change points across the timeline. Any format ffmpeg can read works (MKV, MP4, FLAC, WAV, EAC3, ...).


## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH
- [uv](https://docs.astral.sh/uv/)


## Install

```sh
uv tool install git+https://github.com/Fulcrum-x/soroe
```

This installs `soroe` as a global command on your PATH. To upgrade later, run `uv tool upgrade soroe`.


## Usage

```sh
soroe file1.mkv file2.mkv
```

```
soroe · audio cross-correlation (GCC-PHAT)
──────────────────────────────────────────

  Offset      -4107 ms  file1 delayed by 4107 ms
  Frames      -98.46    @ 23.976 fps · nearest: -98
  Confidence  excellent  118.24x peak-to-sidelobe
```
> [!NOTE]
> **Confidence** represents the whitened correlation’s peak to its maximum sidelobe ratio. ≥10 reads `excellent`, ≥5 `good`, ≥3 `fair`, below that `poor`/`unreliable`. Values under 5 trigger a warning
> and typically point to 3 different scenarios — drift along the alignment path, the files don't have enough overlap, or there is high signal repetition (loops, repeating theme music, you get the idea...).

### Options

| Flag | Description |
|---|---|
| `--version` | Print version and exit |
| `--duration SECONDS` | Seconds of content to analyze (default `600`; full file with `--drift`) |
| `--audio-track INT` | Audio track index (default `0`; falls back to track 0 with a warning if the index is missing) |
| `--verbose` | Detailed progress and debug info |
| `--drift` | Drift/divergence detection mode |
| `--drift-window INT` | Window size in seconds for drift analysis (default `30`) |
| `--drift-threshold INT` | Minimum offset change in ms to count as a change point (default: auto) |
| `--max-drift INT` | Search radius in seconds around the base offset (default: auto) |
| `--prescan` | Calibrate the drift window size from the content (requires `--drift`) |
| `--saveoffsets`, `-so` | Save segments/change points to `./offsets/` (requires `--drift`) |

```sh
soroe release_a.mkv release_b.mkv --duration 120 --audio-track 1 --verbose
soroe source.flac remux.m4a
```

### Batch mode

Pass two directories and soroe pairs files by their `SxxEyy` token (`S01E04`, `s12e09`, `S1E4`) and analyzes each pair. Unmatched files are skipped with a warning. Results stream to stdout prefixed with `[SxxEyy]`. Directory scanning is top-level only.

```sh
soroe ./release_a ./release_b
soroe ./release_a ./release_b --drift
```


## Drift detection

The `--drift` flag enables windowed analysis across the full timeline, detecting points where two files diverge relative to each other. This is useful for sources with frame insertions/deletions, speed changes, or edit differences.
> [!TIP]
> Drift analysis first measures the **base offset** between the files with a cheap full-length correlation and centers every window's search on it, so sources with a large fixed offset (a missing recap, different leaders) work out of the box: `--max-drift` only needs to cover how far the offset *drifts around that base*, not the offset itself.

```sh
soroe file1.flac file2.flac --drift
```

```
soroe · drift analysis
──────────────────────
  30s windows · 20ms threshold (auto) · search radius 5.5s (auto) · base offset +2937 ms

  Segments
    00:00:00.0 → 00:02:45.4    +18 ms  excellent (204.63x)
    00:02:45.4 → 00:02:48.4  transition
    00:02:48.4 → 00:04:29.4   +768 ms  excellent (236.87x)
    00:04:29.4 → 00:04:32.4  transition
    00:04:32.4 → 00:14:09.8  +1811 ms  excellent (112.05x)
    00:14:09.8 → 00:14:12.8  transition
    00:14:12.8 → 00:23:48.2  +2937 ms  excellent (137.80x)
    00:23:48.2 → 00:23:51.2  transition
    00:23:51.2 → 00:26:55.7  +4147 ms  excellent (154.94x)
    00:26:55.7 → 00:26:58.7  transition
    00:26:58.7 → 00:35:55.0  +5190 ms  excellent (129.79x)
    00:35:55.0 → 00:35:58.0  transition
    00:35:58.0 → 00:44:44.4  +6191 ms  excellent (107.93x)
    00:44:44.4 → 00:44:47.4  transition
    00:44:47.4 → 00:45:36.0  +7358 ms  excellent (298.08x)

  Change points
    00:02:46.9  +18 ms → +768 ms
    00:04:30.9  +768 ms → +1811 ms
    00:14:11.2  +1811 ms → +2937 ms
    00:23:49.7  +2937 ms → +4147 ms
    00:26:57.2  +4147 ms → +5190 ms
    00:35:56.5  +5190 ms → +6191 ms
    00:44:45.9  +6191 ms → +7358 ms

  Summary
    Segments        8
    Change points   7
    Max drift       1210 ms
```

### Linear drift

Steady offset ramps from mismatched playback speeds or framerates are detected separately from step changes and reported as a drift rate and speed ratio (seconds of file2 per second of file1):

```
soroe · drift analysis
──────────────────────
  30s windows · 26ms threshold (auto) · search radius 3s (auto) · base offset +86 ms

  Linear drift
    +62.32 ms/min  speed ratio 0.998961 (file2 s per file1 s) · +163 ms over the scan

  Segments
    00:00:00.0 → 00:02:59.0      +1 ms → +187 ms  unreliable (1.38x)

  Summary
    Segments        1
    Change points   0
    Linear drift    +62.32 ms/min (x0.998961)
```

Under linear drift, segments report their offset at the segment start and end, and step changes are still detected on the residuals. Per-window confidence reads low when drift is strong; the rate estimate aggregates all windows and is far more precise than any single one.

### Tuning drift parameters

By default soroe derives the drift parameters from the files themselves and reports what it chose — the parameter line of the output marks derived values with `(auto)` or `(prescan)`, and each value is also logged to stderr as it is resolved. Passing an explicit value always wins and switches that parameter back to fully manual behavior.

- **`--drift-window`**: Larger windows give more accurate offset estimates but may miss narrow changes; smaller windows are more sensitive but noisier. The default is a fixed 30 s. Pass `--prescan` to calibrate the size from the content instead: soroe probes a few spots across the file at several candidate sizes and picks the smallest window that still locks reliably — sparse or quiet audio pushes the choice up, strong drift pushes it down. When the calibration can't find a reliable size (e.g. it probed silence), it falls back to the default and says so.
- **`--drift-threshold`**: Auto by default — soroe picks the larger of the scan's own noise floor (how much consecutive windows jitter) and one frame duration, then clamps the result to 20–70 ms. That lands high enough that measurement jitter never fires a change point, but low enough that a one-frame edit still does. Raise it explicitly to ignore small jumps, lower it to catch subtle shifts.
- **`--max-drift`**: Auto by default — the search radius derives from how far the base-offset probes disperse, plus a safety margin, never narrower than ~3 s and never wider than ~120 s (probes that fail a confidence check are discarded first, and a capped radius is reported with a warning — memory use grows with the radius, so runaway evidence is not allowed to set it). A large but constant offset is handled by the base-offset anchor regardless. Set it explicitly to force a radius; explicit values the probes contradict are still widened automatically (with a warning) and are never capped. Larger radii slow down analysis.

### Saving offsets

`--saveoffsets` (`-so`) writes the segments and change points to `./offsets/`: `offsets.txt` for a single pair, one `<SxxEyy>.txt` per pair in batch runs. Plain text with ANSI styling stripped.

```sh
soroe file1.flac file2.flac --drift --saveoffsets
soroe ./release_a ./release_b --drift -so
```
