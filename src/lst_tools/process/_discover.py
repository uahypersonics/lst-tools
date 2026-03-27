"""Shared directory discovery helpers for process workflows."""


# --------------------------------------------------
# imports
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path


# --------------------------------------------------
# public helpers
# --------------------------------------------------
def discover_pattern_dirs(
    parent_dir: str | Path,
    pattern: str,
) -> list[Path]:
    """Discover and sort directories in *parent_dir* matching a glob *pattern*.

    Args:
        parent_dir: directory to search.
        pattern: glob pattern to match (for example ``"kc_*"``).

    Returns:
        Sorted list of matching directories.
    """
    # convert to Path object
    parent = Path(parent_dir)

    # discover and sort matching directories
    matches = sorted(
        d for d in parent.glob(pattern)
        if d.is_dir()
    )

    return matches
