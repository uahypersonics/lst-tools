"""lst-tools spectra-process — post-process LST spectra results."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import read_config
from lst_tools.process import spectra_process


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'spectra-process' cli option
# --------------------------------------------------
def cmd_spectra_process(
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Path to configuration file (default: search in current directory)."),
    ] = None,
    animate: Annotated[
        bool,
        typer.Option("--animate", help="Run only raw spectra animation-file output."),
    ] = False,
    branches: Annotated[
        bool,
        typer.Option("--branches", help="Run only branch tracking output."),
    ] = False,
    classify: Annotated[
        bool,
        typer.Option("--classify", help="Run only isolation-score-based branch classification output."),
    ] = False,
) -> None:
    """Post-process LST spectra calculation results."""

    logger.debug("processing LST spectra calculation results")

    # if no selector flags are set, run the default useful outputs
    do_animate = True
    do_branches = True
    do_classify = False

    # if any selector is explicitly set, use the selection as an only-filter
    if animate or branches or classify:
        do_animate = animate
        do_branches = branches
        do_classify = classify

    # if classification is configured in the config and no selectors are provided,
    # include it in the default full processing path
    auto_classify = False

    try:
        config = read_config(path=cfg)
        if not (animate or branches or classify):
            auto_classify = config.processing.spectra.isolation_threshold is not None
            do_classify = auto_classify

        result_path = spectra_process(
            cfg=config,
            reporter=typer.echo,
            do_animate=do_animate,
            do_branches=do_branches,
            do_classify=do_classify,
        )
        typer.echo(f"spectra post-processing complete: {result_path}")
    except NotImplementedError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("spectra post-processing failed", exc_info=True)
        raise typer.Exit(1)
