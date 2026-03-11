"""lst-tools spectra: set up spectral analysis calculations."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import read_config
from lst_tools.setup.spectra import spectra_setup


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'spectra' cli option
# --------------------------------------------------
def cmd_spectra(
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Path to configuration file (e.g. lst.cfg)."),
    ] = None,
) -> None:
    """Set up spectral analysis calculations for specific x-locations, frequencies, and wavenumbers.
\f

    Workflow:

    1. Load the project config (auto-discovered or via ``--cfg``).
    2. Read streamwise locations from the meanflow binary, applying
       ``x_s``, ``x_e``, and ``i_step`` from ``[lst.params]``.
    3. Build frequency and wavenumber arrays from ``[lst.params]``.
    4. For every (x, f, beta) combination, scaffold a case directory,
       generate an LST input deck, and write an HPC run script.
    5. Write a ``run_cases.sh`` launcher script.
    """

    # debug output for devs (only displayed if --debug mode is active in the parent context)
    logger.debug("setting up spectra calculations")

    try:
        # load config (read_config.py in /config/)
        config = read_config(path=cfg)

        # run spectra setup (logic in /setup/spectra.py)
        generated_files = spectra_setup(cfg=config)

        # output for user
        typer.echo("spectra setup complete")

        # debug output of all generated files/directories
        logger.debug("Generated files:")
        for file_path in generated_files:
            logger.debug("  %s", file_path)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("spectra setup failed", exc_info=True)
        raise typer.Exit(1)
