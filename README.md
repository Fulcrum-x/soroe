# soroe

Find the offset between two MKV files of the same content.

Uses audio cross-correlation to determine how many milliseconds (and frames) one file is shifted relative to the other.

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

```sh
uv run soroe file1.mkv file2.mkv
```

### Options

| Flag | Description |
|---|---|
| `--fps FLOAT` | Framerate for frame-accurate offset (e.g. `23.976`) |
| `--duration SECONDS` | Seconds of content to analyze (default: `600`) |
| `--audio-track INT` | Audio track index to use (default: `0`) |
| `--verbose` | Print detailed progress and debug info |

### Examples

Basic offset detection:
```sh
uv run soroe release_a.mkv release_b.mkv
```

With frame-accurate output:
```sh
uv run soroe release_a.mkv release_b.mkv --fps 23.976
```

Analyze only the first 2 minutes, using the second audio track:
```sh
uv run soroe release_a.mkv release_b.mkv --duration 120 --audio-track 1 --verbose
```

### Example output

```
Method: audio cross-correlation
Confidence: excellent (118.24x peak ratio)
Offset: -4107 ms (file1 starts 4107ms after file2)
Offset (frames): -98.46 frames @ 23.976fps
Frame-aligned offset: -4087 ms (-98 frames)
```

## Notes

- Extracts audio from both files via ffmpeg (mono, 16 kHz, piped — no temp files)
- Computes cross-correlation using `scipy.signal.fftconvolve`
- The lag at the correlation peak gives the sample offset, converted to milliseconds
- Confidence is the ratio of the primary peak to the next-highest non-adjacent peak

A positive offset means file2 starts after file1. A negative offset means file1 starts after file2.