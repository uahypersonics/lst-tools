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
) -> None:
    """Post-process LST tracking calculation results."""

    logger.debug("processing LST tracking calculation results")

    try:
        config = read_config(path=cfg)
        result_path = tracking_process(cfg=config)
        typer.echo(f"tracking post-processing complete: {result_path}")
    except NotImplementedError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("tracking post-processing failed", exc_info=True)
        raise typer.Exit(1)
