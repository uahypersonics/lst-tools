"""lst-tools command-line interface (typer).

Subcommands are defined in separate ``cmd_*.py`` modules and
registered on the shared ``cli`` Typer instance below.  Each
module exports either a plain function (registered with
``cli.command()``) or a sub-Typer app (registered with
``cli.add_typer()``).
"""

# --------------------------------------------------
# imports
# --------------------------------------------------
from __future__ import annotations

import logging
import sys
from typing import Annotated

import typer

from .cmd_hpc import cmd_hpc
from .cmd_init import cmd_init
from .cmd_lastrac import cmd_lastrac
from .cmd_parsing import cmd_parsing
from .cmd_spectra import cmd_spectra
from .cmd_spectra_process import cmd_spectra_process
from .cmd_tracking import cmd_tracking
from .cmd_tracking_process import cmd_tracking_process


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# app
# --------------------------------------------------
class _OrderedGroup(typer.core.TyperGroup):
    """Preserve command registration order in --help output."""

    def list_commands(self, ctx: typer.Context) -> list[str]:
        return list(self.commands)


cli = typer.Typer(
    name="lst-tools",
    help="lst-tools command-line interface",
    no_args_is_help=True,
    add_completion=False,
    cls=_OrderedGroup,
)

# --------------------------------------------------
# command groups: setup and process
# --------------------------------------------------
setup_app = typer.Typer(
    help="Set up LST calculations (parsing, tracking, spectra).",
    cls=_OrderedGroup,
)

process_app = typer.Typer(
    help="Postprocess LST results (tracking, spectra).",
    cls=_OrderedGroup,
)


# --------------------------------------------------
# --version callback
# --------------------------------------------------
def _version_callback(value: bool) -> None:
    if value:
        from lst_tools import __version__

        typer.echo(f"lst-tools {__version__}")
        raise typer.Exit()


@cli.callback()
def _main(
    ctx: typer.Context,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Enable debug output."),
    ] = False,
) -> None:
    """lst-tools: Linear Stability Theory pre-/postprocessing toolkit."""

    # set debug flag in context for access by subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    # if debug mode is active, set up debug logging for lst_tools modules
    if debug:

        # get the package-level logger ("lst_tools") — this is the parent
        # of every logger in the codebase (e.g. "lst_tools.setup.spectra",
        # "lst_tools.config.read_config", etc.).  setting its level to DEBUG
        # enables debug output for the entire package at once, because
        # child loggers propagate messages up to their parent.
        lst_logger = logging.getLogger("lst_tools")
        # set all loggers in the package to DEBUG level
        lst_logger.setLevel(logging.DEBUG)

        # attach a handler that prints to stderr so debug messages are
        # visible in the terminal.
        # the guard prevents adding duplicate handlers if this callback runs more than once
        # (NullHandler from __init__.py is always present, so check for StreamHandler)
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in lst_logger.handlers
        )
        if not has_stream_handler:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(
                logging.Formatter("[%(levelname)-7s] %(name)s: %(message)s")
            )
            lst_logger.addHandler(handler)


# --------------------------------------------------
# subcommands
# cli.command()(func)        — register a top-level command
# cli.add_typer(app, name=)  — register a command group with subcommands
# name= is only needed when the CLI name differs from the function name
#   (e.g. spectra-process vs spectra_process)
# --------------------------------------------------
cli.command(name="init")(cmd_init)
cli.add_typer(setup_app, name="setup")
cli.command(name="lastrac")(cmd_lastrac)
cli.add_typer(process_app, name="process")
cli.command(name="hpc")(cmd_hpc)

# setup subcommands
setup_app.command(name="parsing")(cmd_parsing)
setup_app.command(name="tracking")(cmd_tracking)
setup_app.command(name="spectra")(cmd_spectra)

# process subcommands
process_app.command(name="tracking")(cmd_tracking_process)
process_app.command(name="spectra")(cmd_spectra_process)


# --------------------------------------------------
# main entry point
# --------------------------------------------------
if __name__ == "__main__":
    cli()
