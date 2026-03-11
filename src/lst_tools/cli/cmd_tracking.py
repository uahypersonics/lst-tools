"""lst-tools tracking — set up tracking step for the LST solver."""


# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import find_config, read_config
from lst_tools.setup.tracking import tracking_setup


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'tracking' cli option
# --------------------------------------------------
def cmd_tracking(
    ctx: typer.Context,
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Path to configuration file (e.g. lst.cfg)."),
    ] = None,
    auto_fill: Annotated[
        bool,
        typer.Option(
            "--auto-fill", "-a",
            help="Fill unset sweep parameters (beta_s, beta_e, d_beta, i_step) with defaults.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f",
            help="Force overwrite of existing parameter values when used with --auto-fill.",
        ),
    ] = False,
) -> None:
    """Set up tracking step for the LST solver (requires a solution from the parsing step).
\f

    Workflow:

    1. Load the project config (auto-discovered or via ``--cfg``).
    2. Read parsing solution to extract initial guesses for each
       frequency / wavenumber combination.
    3. For every combination, scaffold a case directory, generate an
       LST input deck, and write an HPC run script.
    4. Write a ``run_jobs.sh`` launcher script.
    5. If ``--debug`` is active, write diagnostic output to ``./debug/``.
    """

    # get debug flag from parent context (set by --debug in the main cli callback)
    debug = ctx.obj.get("debug", False) if ctx.obj else False

    # initialize debug_path to None; if debug mode is active, set to ./debug
    debug_path = None

    if debug:
        debug_path = Path("./debug")
        debug_path.mkdir(parents=True, exist_ok=True)

    # debug output for devs
    logger.debug("setting up input deck for tracking step")

    try:
        # load config (read_config.py in /config/)
        config = read_config(path=cfg)

        # resolve cfg_path for write-back (auto-discover if --cfg not given)
        resolved_cfg_path = cfg if cfg is not None else find_config(".")

        tracking_setup(
            cfg=config,
            debug_path=debug_path,
            auto_fill=auto_fill,
            force=force,
            cfg_path=resolved_cfg_path,
        )

        typer.echo("tracking setup complete")
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("tracking setup failed", exc_info=True)
        raise typer.Exit(1)
