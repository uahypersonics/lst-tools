"""lst-tools tracking-process — post-process LST tracking results."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import read_config
from lst_tools.process import tracking_process


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'tracking-process' cli option
# --------------------------------------------------
def cmd_tracking_process(
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Path to configuration file (default: search in current directory)."),
    ] = None,
    dir: Annotated[
        Optional[list[Path]],
        typer.Option("--dir", help="Specific kc_* directory to process (repeatable). Defaults to all kc_* in cwd."),
    ] = None,
    maxima: Annotated[
        bool,
        typer.Option("--maxima", help="Run only ridge-line maxima extraction."),
    ] = False,
    volume: Annotated[
        bool,
        typer.Option("--volume", help="Run only 3-D volume assembly."),
    ] = False,
    interpolate: Annotated[
        bool,
        typer.Option("--interpolate", help="Use parabolic interpolation for sub-grid peak refinement."),
    ] = False,
) -> None:
    """Post-process LST tracking calculation results."""

    logger.debug("processing LST tracking calculation results")

    # if neither flag is set, run both; if one is set, run only that one
    do_maxima = True
    do_volume = True

    if maxima or volume:
        do_maxima = maxima
        do_volume = volume

    # if specific dirs are given, disable volume (needs all slices)
    if dir:
        do_volume = False

    try:
        config = read_config(path=cfg)
        tracking_process(
            cfg=config,
            do_maxima=do_maxima,
            do_volume=do_volume,
            kc_dirs=dir,
            interpolate=interpolate,
        )
        typer.echo("tracking post-processing complete")
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("tracking post-processing failed", exc_info=True)
        raise typer.Exit(1)
