"""Build and write HPC job scripts.

This is the thin public entry point.
Script rendering is handled by :mod:`._templates`.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

from ._resolve import ResolvedJob
from ._templates import render


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function: script_build
# --------------------------------------------------
def script_build(
    cfg: ResolvedJob,
    fpath: str | Path,
    *,
    args: list[str] | None = None,
    lst_exe: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Render a batch script and write it to *fpath*.

    Parameters
    ----------
    cfg:
        A :class:`ResolvedJob` (from :func:`hpc_configure`).
    fpath:
        Path where the script file is created.
    args:
        Command-line arguments for the solver.
    lst_exe:
        Name or path of the LST executable.
    extra_env:
        Additional environment variables to export.

    Returns
    -------
    Path
        Absolute path to the written script file.
    """
    logger.debug("hpc config: %s", cfg)
    logger.debug("output path: %s", fpath)
    logger.debug("extra env: %s", extra_env)

    if cfg.fname_run_script is None:
        raise ValueError(
            "Cannot build run script: scheduler is unknown "
            "(fname_run_script is None)."
        )

    # generate script content (autodetects scheduler and launcher)
    text = render(cfg, lst_exe=lst_exe or "lst.x", args=args, extra_env=extra_env)

    # make sure fpath is a Path object
    fpath = Path(fpath)
    # assemble the full path to the script file and write it
    fname = fpath / cfg.fname_run_script
    # write the script text to file (UTF-8 encoding)
    fname.write_text(text, encoding="utf-8")
    # make the script executable
    fname.chmod(0o755)

    return fname
