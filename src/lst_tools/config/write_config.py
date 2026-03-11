"""Write a TOML configuration file.

Serialises a config dictionary into a TOML file on disk.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations
from pathlib import Path
from typing import Any
import logging
import tomli_w as toml_w
import numpy as np


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# helper function to recursively convert config values to toml-serialisable types
# --------------------------------------------------
def _serialize_for_toml(obj):
    """Recursively convert *obj* so ``tomli_w`` can serialise it.

    Handles numpy scalars, complex numbers, Paths, and ``None``.
    """
    if isinstance(obj, (np.generic,)):
        return obj.item()

    # complex -> string "(a,b)"
    if isinstance(obj, complex):
        return f"({obj.real},{obj.imag})"

    # paths -> str
    if isinstance(obj, Path):
        return str(obj)

    # dict -> recurse, keep keys; map None to ""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if v is None:
                out[k] = ""
            else:
                out[k] = _serialize_for_toml(v)
        return out

    # list/tuple -> recurse elementwise; replace None with ""
    if isinstance(obj, (list, tuple)):
        return [_serialize_for_toml(v) if v is not None else "" for v in obj]

    return obj


# --------------------------------------------------
# main function: write_config
# --------------------------------------------------
def write_config(
    path: str | Path = "lst.cfg",
    overwrite: bool = False,
    cfg_data: dict[str, Any] | None = None,
) -> Path:
    """Write a TOML configuration file to *path*.

    Args:
        path (str | Path): Destination path for the config file.
        overwrite (bool): If ``False`` and *path* already exists,
            return immediately without writing.
        cfg_data (dict): Config dictionary to serialise.

    Returns:
        Path: The path the config was written to (or already existed at).
    """

    # resolve the output path
    p = Path(path)

    # check for existing file and overwrite flag
    if p.exists() and not overwrite:
        return p

    if cfg_data is None:
        raise ValueError("cfg_data is required")

    # ensure parent directory exists (mkdir -p semantics)
    p.parent.mkdir(parents=True, exist_ok=True)

    # serialize the config data for TOML output (handle numpy types, complex numbers, Paths, and None)
    cfg_data_final = _serialize_for_toml(cfg_data)

    # write the config file
    with p.open("wb") as f:
        toml_w.dump(cfg_data_final, f)

    # return path to config file
    return p

