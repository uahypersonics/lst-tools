"""Locate a configuration file by scanning for known filenames.

Fallback routine used when no explicit ``--cfg`` path is passed to a
CLI command.  Searches *search_path* for commonly used config file names
(``lst.cfg``, ``lst.toml``, ``config.toml``) and returns the first match.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# find a likely config file in a directory
# --------------------------------------------------
def find_config(search_path: str | Path = ".") -> Path | None:
    """Search *search_path* for a recognised configuration file.

    Iterates over a priority-ordered list of candidate filenames and
    returns the first one that exists on disk.

    Parameters
    ----------
    search_path : str | Path, optional
        Directory to scan (default: current working directory).

    Returns
    -------
    Path | None
        Path to the first matching config file, or ``None`` if no
        candidates are found.
    """

    # save search path in p -> if no search path was provided use default ./ (current run dir)
    search = Path(search_path)

    # list of possible configuration files sorted after priority
    names = [
        "lst.cfg",
        "lst.toml",
        "config.toml",
    ]

    # loop over config file names and return first match
    for name in names:
        p = search / name
        if p.exists():
            return p

    # no candidates found
    logger.error(
        "no config files found in %s (looked for: %s)",
        search.resolve(),
        ', '.join(names),
    )
    return None
