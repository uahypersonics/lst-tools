"""lst-tools command-line interface (typer).

Subcommands are defined in separate ``cmd_*.py`` modules and
registered on the shared ``cli`` Typer instance below. Each
module exports either a plain function (registered with
``cli.command()``) or a sub-Typer app (registered with
``cli.add_typer()``).
"""

# --------------------------------------------------
# imports
# --------------------------------------------------
from __future__ import annotations

import typer

from .callbacks import cli_callback
from .cmd_clean import cmd_clean_parsing, cmd_clean_spectra, cmd_clean_tracking
from .cmd_extract import cmd_extract
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


cli = typer.Typer(
    name="lst-tools",
    help="lst-tools command-line interface",
    no_args_is_help=True,
    add_completion=False,
)

# --------------------------------------------------
# command groups: setup and process
# --------------------------------------------------
setup_app = typer.Typer(
    help="Set up LST calculations (parsing, tracking, spectra).",
    no_args_is_help=True,
)

process_app = typer.Typer(
    help="Postprocess LST results (tracking, spectra).",
    no_args_is_help=True,
)

clean_app = typer.Typer(
    help="Remove generated files (parsing, tracking, spectra).",
    no_args_is_help=True,
)

visualize_app = typer.Typer(
    help="Visualize LST results.",
    no_args_is_help=True,
)

# --------------------------------------------------
# register callbacks
# --------------------------------------------------
cli.callback()(cli_callback)

# --------------------------------------------------
# subcommands
# cli.command()(func)        — register a top-level command
# cli.add_typer(app, name=)  — register a command group with subcommands
# name= is only needed when the CLI name differs from the function name
#   (e.g. spectra-process vs spectra_process)
# --------------------------------------------------
    
# workflow commands
cli.command(name="init", rich_help_panel="Workflow")(cmd_init)
cli.add_typer(setup_app, name="setup", rich_help_panel="Workflow")
cli.add_typer(clean_app, name="clean", rich_help_panel="Workflow")
cli.add_typer(process_app, name="process", rich_help_panel="Workflow")
cli.add_typer(
    visualize_app,
    name="visualize",
    rich_help_panel="Workflow",
)

# utility commands
cli.command(
    name="hpc",
    rich_help_panel="Utilities",
    help="Generate a runnable HPC job script.",
)(cmd_hpc)
cli.command(name="info", rich_help_panel="Utilities")(cmd_info)
cli.command(name="lastrac", rich_help_panel="Utilities")(cmd_lastrac)
cli.command(name="extract", rich_help_panel="Utilities")(cmd_extract)

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
