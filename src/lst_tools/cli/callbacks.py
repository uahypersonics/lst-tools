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
# configure package logging
# --------------------------------------------------
def _configure_console_logging(level: int) -> None:
    """Send package log messages at or above level to the current stderr."""

    lst_logger = logging.getLogger("lst_tools")
    lst_logger.setLevel(level)

    # reuse a console handler so repeated CliRunner invocations follow current stderr
    stream_handlers = [
        handler
        for handler in lst_logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]
    if stream_handlers:
        for handler in stream_handlers:
            handler.setLevel(level)
            handler.stream = sys.stderr
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(name)s: %(message)s"))
    lst_logger.addHandler(handler)


# --------------------------------------------------
# verbose callback
# --------------------------------------------------
def verbose_callback(value: bool) -> bool:
    """Enable diagnostic logging when --verbose is passed."""

    if value:
        _configure_console_logging(logging.DEBUG)

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

    # show warnings by default while reserving detailed diagnostics for --verbose
    log_level = logging.DEBUG if verbose else logging.WARNING
    _configure_console_logging(log_level)
