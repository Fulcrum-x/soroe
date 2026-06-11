"""Shared exception types for soroe."""

from __future__ import annotations


class SoroeError(Exception):
    """Expected per-input failure. Caught at the top level for a clean exit, or
    per pair in batch mode so one bad file doesn't abort the whole run."""
