"""
Post-processing functionality for LST tracking calculations

This module orchestrates two post-processing steps:
  1. **maxima** — ridge extraction from each kc_* case directory
  2. **volume** — assembly of 2-D slices into a 3-D volume file

By default both steps run.  The ``do_maxima`` and ``do_volume`` flags
can be used as "only" selectors (e.g. ``do_maxima=True, do_volume=False``).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Mapping

import typer

from .maxima import extract_maxima
from .volume import assemble_volume


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function to process tracking results
# --------------------------------------------------
def tracking_process(
    *,
    cfg: Mapping[str, Any] | None = None,
    do_maxima: bool = True,
    do_volume: bool = True,
    work_dir: Path | None = None,
    kc_dirs: list[Path] | None = None,
    interpolate: bool = False,
) -> Path:
    """
    Process LST tracking calculation results.

    Parameters
    ----------
    cfg : Optional[Mapping[str, Any]]
        Configuration dictionary (currently unused, reserved for future).
    do_maxima : bool
        If True, run ridge-line maxima extraction for every kc_* case.
    do_volume : bool
        If True, assemble 2-D slices into a 3-D volume file.
    work_dir : Optional[Path]
        Working directory containing kc_* subdirectories.
        Defaults to the current working directory.
    kc_dirs : Optional[list[Path]]
        Specific kc_* directories to process. If None, discovers all
        kc_* directories in work_dir.

    Returns
    -------
    Path
        Path to the working directory.
    """

    # --------------------------------------------------
    # get working directory (default to current directory if nothing provided)
    # --------------------------------------------------
    if work_dir is None:
        work_dir = Path.cwd()

    logger.info("processing tracking results in %s", work_dir)

    # --------------------------------------------------
    # discover tracking directories (kc_*)
    # --------------------------------------------------
    if kc_dirs is None:
        kc_dirs = sorted(
            d for d in work_dir.iterdir()
            if d.is_dir() and d.name.startswith("kc_")
        )

    # nothing found to process, so just return the working directory
    if not kc_dirs:
        logger.warning("no kc_* directories found in %s", work_dir)
        return work_dir

    logger.debug("processing %d kc_* directories", len(kc_dirs))

    # --------------------------------------------------
    # processing maxima: ridge-line maxima extraction
    # --------------------------------------------------
    if do_maxima:

        typer.echo("starting maxima extraction...")

        total_files = 0

        for kc_dir in kc_dirs:
            typer.echo(f"  extracting maxima: {kc_dir.name}")
            written = extract_maxima(kc_dir, interpolate=interpolate)
            total_files += len(written)

        typer.echo(f"maxima extraction complete: {total_files} file(s) written")

    # --------------------------------------------------
    # processing volume: assemble 3-D volume from 2-D slices
    # --------------------------------------------------

    if do_volume:

        typer.echo("starting volume assembly...")

        vol_path = assemble_volume(work_dir)

        if vol_path is not None:
            typer.echo(f"volume assembly complete: {vol_path.name}")
        else:
            typer.echo("volume assembly produced no output")
            logger.warning("volume assembly produced no output")

    return work_dir
