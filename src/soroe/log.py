"""Colored terminal output for soroe."""

from __future__ import annotations

import shutil
import sys

# ANSI escape codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_CLEAR_LINE = "\033[2K\r"


def bold(s: str) -> str:
    return f"{_BOLD}{s}{_RESET}"


def dim(s: str) -> str:
    return f"{_DIM}{s}{_RESET}"


def red(s: str) -> str:
    return f"{_RED}{s}{_RESET}"


def green(s: str) -> str:
    return f"{_GREEN}{s}{_RESET}"


def yellow(s: str) -> str:
    return f"{_YELLOW}{s}{_RESET}"


def cyan(s: str) -> str:
    return f"{_CYAN}{s}{_RESET}"


def error(msg: str) -> None:
    """Print a red error message to stderr."""
    print(f"{_RED}(Error){_RESET}: {msg}", file=sys.stderr)


def warn(msg: str) -> None:
    """Print a yellow warning message to stderr."""
    print(f"{_YELLOW}(Warning){_RESET}: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    """Print a dim info/debug message to stderr."""
    print(f"{_DIM}[soroe]{_RESET} {msg}", file=sys.stderr)


def progress(current: int, total: int, label: str) -> None:
    """Draw a progress bar with an info label. Overwrites the current line."""
    cols = shutil.get_terminal_size((80, 24)).columns
    pct = current / total if total > 0 else 1.0
    tag = f"{_CYAN}(info){_RESET} {label}"
    # Reserve space: tag + percentage + brackets + padding
    tag_plain_len = 7 + len(label)  # "(info) " + label
    pct_str = f" {pct:>3.0%}"
    overhead = tag_plain_len + len(pct_str) + 5  # [] + spaces
    bar_width = max(10, cols - overhead)
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    line = f"{_CLEAR_LINE}{tag} [{bar}]{pct_str}"
    sys.stderr.write(line)
    sys.stderr.flush()


def progress_clear() -> None:
    """Clear the progress bar line."""
    sys.stderr.write(_CLEAR_LINE)
    sys.stderr.flush()
