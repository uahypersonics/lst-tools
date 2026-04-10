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

from .cmd_clean import cmd_clean_parsing, cmd_clean_spectra, cmd_clean_tracking
from .cmd_hpc import cmd_hpc
from .cmd_info import cmd_info
from .cmd_init import cmd_init
from .cmd_lastrac import cmd_lastrac
from .cmd_parsing import cmd_parsing
from .cmd_spectra import cmd_spectra
from .cmd_spectra_process import cmd_spectra_process
from .cmd_tracking import cmd_tracking
from .cmd_tracking_process import cmd_tracking_process
from .cmd_visualize import cmd_visualize_parsing, cmd_visualize_tracking
from .cmd_visualize_meanflow import cmd_visualize_meanflow


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

clean_app = typer.Typer(
    help="Remove generated files (parsing, tracking, spectra).",
    cls=_OrderedGroup,
)

visualize_app = typer.Typer(
    help="Visualize LST results.",
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
            "--version", "-V",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable informational output."),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", "-d", help="Enable debug output."),
    ] = False,
) -> None:
    """lst-tools: Linear Stability Theory pre-/postprocessing toolkit."""

    # set debug flag in context for access by subcommands
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug

    # determine the effective log level
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = None

    if level is not None:
        lst_logger = logging.getLogger("lst_tools")
        lst_logger.setLevel(level)

        # attach a handler that prints to stderr so log messages are
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
            handler.setLevel(level)
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
cli.add_typer(clean_app, name="clean")
cli.command(name="info")(cmd_info)
cli.command(name="lastrac")(cmd_lastrac)
cli.add_typer(process_app, name="process")
cli.command(name="hpc")(cmd_hpc)
cli.add_typer(visualize_app, name="visualize")

# setup subcommands
setup_app.command(name="parsing")(cmd_parsing)
setup_app.command(name="tracking")(cmd_tracking)
setup_app.command(name="spectra")(cmd_spectra)

# process subcommands
process_app.command(name="tracking")(cmd_tracking_process)
process_app.command(name="spectra")(cmd_spectra_process)

# clean subcommands
clean_app.command(name="parsing")(cmd_clean_parsing)
clean_app.command(name="tracking")(cmd_clean_tracking)
clean_app.command(name="spectra")(cmd_clean_spectra)

# visualize subcommands
visualize_app.command(name="parsing")(cmd_visualize_parsing)
visualize_app.command(name="tracking")(cmd_visualize_tracking)
visualize_app.command(name="meanflow")(cmd_visualize_meanflow)


# --------------------------------------------------
# main entry point
# --------------------------------------------------
if __name__ == "__main__":
    cli()
