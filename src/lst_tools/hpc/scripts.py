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
    out_path: str | Path,
    *,
    args: list[str] | None = None,
    lst_exe: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> Path:
    """Render a batch script and write it to *out_path*.

    Parameters
    ----------
    cfg:
        A :class:`ResolvedJob` (from :func:`hpc_configure`).
    out_path:
        Directory where the script file is created.
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
    logger.debug("output path: %s", out_path)
    logger.debug("extra env: %s", extra_env)

    if cfg.fname_run_script is None:
        raise ValueError(
            "Cannot build run script: scheduler is unknown "
            "(fname_run_script is None)."
        )

    text = render(cfg, lst_exe=lst_exe or "lst.x", args=args, extra_env=extra_env)

    p = Path(out_path)
    fname_out = p / cfg.fname_run_script
    fname_out.write_text(text, encoding="utf-8")
    fname_out.chmod(0o755)

    return fname_out
