"""Callbacks for global options and groups in the lst-tools CLI."""

# --------------------------------------------------
# imports
# --------------------------------------------------
from __future__ import annotations

import logging
import sys
from typing import Annotated

import typer


# --------------------------------------------------
# version callback
# --------------------------------------------------
def version_callback(value: bool) -> bool:
    """Print version and exit when --version is passed."""
    if value:
        from lst_tools import __version__

        typer.echo(f"lst-tools {__version__}")
        raise typer.Exit()
    return value

# --------------------------------------------------
# verbose callback
# --------------------------------------------------
def verbose_callback(value: bool) -> bool:
    """Enable diagnostic logging when --verbose is passed."""

    if value:
        # use debug-level logging for rich user diagnostics
        level = logging.DEBUG

        # get package logger and set level
        lst_logger = logging.getLogger("lst_tools")
        lst_logger.setLevel(level)

        # check whether a stderr stream handler already exists
        has_stream_handler = False
        for handler in lst_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(level)
                has_stream_handler = True

        # add a stream handler if none exists yet
        if not has_stream_handler:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(level)
            handler.setFormatter(
                logging.Formatter("[%(levelname)-7s] %(name)s: %(message)s")
            )
            lst_logger.addHandler(handler)

    return value

# --------------------------------------------------
# main app callback
# --------------------------------------------------
def cli_callback(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable diagnostic output.",
            callback=verbose_callback,
        ),
    ] = False,
) -> None:
    """lst-tools: Linear Stability Theory pre-/postprocessing toolkit."""

    # if verbose is not set, set logger level to warning
    if not verbose:
        logging.getLogger("lst_tools").setLevel(logging.WARNING)
