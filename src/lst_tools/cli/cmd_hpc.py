"""CLI handler for ``lst-tools hpc``.

Reads the project configuration, resolves HPC scheduler settings
(PBS or SLURM), and writes a submittable run script.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import read_config
from lst_tools.hpc import script_build, hpc_configure


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'hpc' command
# --------------------------------------------------
def cmd_hpc(
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Path to configuration file (e.g. lst.cfg)."),
    ] = None,
) -> None:
    """Generate a runnable HPC job script.
\f
    Workflow:

    1. Load the project config (auto-discovered or via ``--cfg``).
    2. Resolve HPC scheduler settings (PBS / SLURM) with generous
       defaults applied via ``hpc_configure()``.
    3. Determine the LST executable (``cfg.lst_exe`` or ``lst.x``).
    4. Write the run script to the current directory via
       ``script_build()``.
    """

    # load config
    try:
        config = read_config(path=cfg)

        # configure HPC settings
        hpc_cfg = hpc_configure(config, set_defaults=True)

        # debug output (logger.debug will only print if --debug mode is enabled)
        logger.debug("hpc configuration: %s", hpc_cfg)

        # resolve LST executable
        lst_exe = config.lst_exe or "lst.x"

        # generate run script (logic in /hpc/)
        script_path = script_build(
            hpc_cfg,
            out_path="./",
            lst_exe=lst_exe,
            args=["lst_input.dat"],
        )

        # output for user
        typer.echo(f"run script written to {script_path}")
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("hpc setup failed", exc_info=True)
        raise typer.Exit(1)
