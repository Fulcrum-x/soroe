# soroe

Find the temporal offset between two audio/video files of the same content. Useful for muxing tracks between different releases with the correct delay.

Uses audio cross-correlation to determine how many milliseconds (and frames) one file is shifted relative to the other. Supports any format ffmpeg can read (MKV, MP4, FLAC, WAV, EAC3, M4A, MKA, etc.).

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) on PATH
- [uv](https://docs.astral.sh/uv/)

## Setup

```sh
git clone https://gitea.okami.icu/fulcrum/soroe.git
cd soroe
uv sync
```

## Usage

### Basic offset detection

Find the global offset between two files:

```sh
uv run soroe file1.mkv file2.mkv
```

If either file has a video stream, framerate is auto-detected via ffprobe and frame offset is shown automatically.

### Options

| Flag | Description |
|---|---|
| `--duration SECONDS` | Seconds of content to analyze (default: `600`) |
| `--audio-track INT` | Audio track index to use (default: `0`) |
| `--verbose` | Print detailed progress and debug info |
| `--drift` | Enable drift/divergence detection mode |
| `--drift-window INT` | Window size in seconds for drift analysis (default: `30`) |
| `--drift-threshold INT` | Minimum offset change in ms to count as a change point (default: `70`) |
| `--max-drift INT` | Maximum expected drift in seconds — sets the search radius (default: `5`) |

### Examples

Analyze only the first 2 minutes, using the second audio track:
```sh
uv run soroe release_a.mkv release_b.mkv --duration 120 --audio-track 1 --verbose
```

Compare two audio files directly:
```sh
uv run soroe source.flac remux.m4a
```

### Example output

```
Method: audio cross-correlation
Confidence: excellent (118.24x peak ratio)
Offset: -4107 ms (file1 starts 4107ms after file2)
Offset (frames): -98.46 frames @ 23.976fps
Nearest frame: -98
```

## Drift detection

The `--drift` flag enables windowed analysis across the full timeline, detecting points where two files diverge relative to each other. This is useful for sources with frame insertions/deletions, speed changes, or edit differences.

When `--drift` is used without `--duration`, the full file length is analyzed automatically.

```sh
uv run soroe file1.flac file2.flac --drift
uv run soroe file1.flac file2.flac --drift --drift-threshold 80 --verbose
```

### How drift detection works

1. **Pass 1 — Coarse scan**: Divides both audio streams into overlapping windows (default 30s, 50% overlap). Each window from file A is correlated against a search region in file B (±5s by default), producing a local offset and confidence at each point in the timeline.

2. **Pass 2 — Change point detection**: Identifies where the offset jumps by more than the threshold (default 70ms). Binary-search refinement narrows each change point to ±1 second accuracy.

3. **Pass 3 — Segment compilation**: Groups the timeline into contiguous segments sharing the same offset, separated by short transition zones around each change point.

### Drift output example

```
Drift analysis (30s windows, 70ms threshold)
==============================================

Segments:
  00:00:00.0 - 00:03:04.8  offset: +28 ms  (confidence: fair (3.33x))
  00:03:04.8 - 00:03:07.8  transition
  00:03:07.8 - 00:04:42.8  offset: +81 ms  (confidence: good (8.56x))
  00:04:42.8 - 00:04:45.8  transition
  00:04:45.8 - 00:18:05.3  offset: -89 ms  (confidence: good (6.68x))

Change points:
  00:03:06.3  offset shifts from +2 ms to +143 ms (Δ141 ms)
  00:04:44.3  offset shifts from +93 ms to +10 ms (Δ83 ms)

Global summary:
  Segments: 3
  Total change points: 2
  Max drift: 141 ms
```

### Tuning drift parameters

- **`--drift-window`**: Larger windows give more accurate offset estimates but may miss narrow changes. Smaller windows are more sensitive but noisier. Default 30s works well for most content.
- **`--drift-threshold`**: Raise this to ignore small jitter and only report significant jumps. Lower it to catch subtle shifts. Default 70ms.
- **`--max-drift`**: Sets the search radius around each window position. If the offset between your files could be more than 5 seconds, increase this. Larger values slow down analysis.

## How it works

1. Extracts audio from both files via ffmpeg (mono, 16 kHz, piped to stdout — no temp files)
2. Computes cross-correlation using `scipy.signal.fftconvolve`
3. The lag at the correlation peak gives the sample offset, converted to milliseconds
4. Confidence is the ratio of the primary peak to the next-highest non-adjacent peak (excellent ≥10x, good ≥5x, fair ≥3x, poor ≥1.5x, unreliable <1.5x)

A positive offset means file2 starts after file1. A negative offset means file1 starts after file2.