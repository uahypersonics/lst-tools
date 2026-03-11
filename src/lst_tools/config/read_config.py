"""Load and return an lst_tools configuration.

This module is the public entry-point for config loading.
The heavy lifting lives in ``schema.py`` (dataclasses + coercion).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

from .find_config import find_config
from .schema import Config

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function: read_config
# --------------------------------------------------
def read_config(
    path: str | Path | None = None,
) -> Config:
    """Load configuration from a TOML file.

    Args:
        path (str | Path | None): Explicit TOML path.  If omitted, searches
            for common config filenames in the CWD (``lst.cfg``, ``lst.toml``, …).

    Returns:
        Config: A ``Config`` dataclass.  Missing keys use built-in defaults.

    Raises:
        FileNotFoundError: If *path* is given but does not exist on disk.
    """
    # resolve the config file path
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"configuration file not found: {path}"
            )
    else:
        cfg_path = find_config(".")
        if cfg_path is None:
            logger.debug("no config file found in CWD -> using defaults")
            return Config()

    logger.debug("loading config from %s", cfg_path)

    # load and parse the config file, returning a Config dataclass
    return Config.from_toml(cfg_path)
