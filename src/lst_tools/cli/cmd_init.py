"""CLI handler for ``lst-tools init``.

Generates a default LST configuration file (``lst.cfg``) with sensible
defaults.  Optionally, a geometry preset (cone, ogive, flat-plate,
cylinder) can be applied and flow conditions read from a
``flow_conditions.dat`` file to seed the output.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import write_config
from lst_tools.config.geometry import GeometryPreset, GEOMETRY_TEMPLATES
from lst_tools.config.merge import merge_dicts, merge_flow_defaults
from lst_tools.config.schema import Config


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'init' command
#
# note: \f truncates docstring for --helpi cli output
# --------------------------------------------------
def cmd_init(
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Output config path.")
    ] = Path("lst.cfg"),
    force: Annotated[
        bool, typer.Option("--force", "-f", help="Overwrite if file exists.")
    ] = False,
    geometry: Annotated[
        Optional[GeometryPreset],
        typer.Option(
            "--geometry",
            "-g",
            help="Pre-populate config for a specific geometry (cone, ogive, flat-plate, cylinder).",
        ),
    ] = None,
    flow_path: Annotated[
        Optional[Path], typer.Option("--flow", "-F", help="Path to flow_conditions.dat.")
    ] = None,
) -> None:
    """Create a default lst.cfg configuration file.
\f

    The config is built in layers:

    1. Start from schema defaults.
    2. If ``--geometry`` is given, overlay the matching geometry preset.
    3. If a ``flow_conditions.dat`` is found (or provided via ``--flow``),
       merge recognised flow-condition keys into the seed.
    4. Write the final dict to *out* as TOML.

    Parameters
    ----------
    out : Path
        Destination path for the config file (default: ``lst.cfg``).
    force : bool
        If *True*, overwrite an existing file without prompting.
    geometry : GeometryPreset | None
        Optional geometry preset to pre-populate the config.
    flow_path : Path | None
        Explicit path to a ``flow_conditions.dat`` file.  When a geometry
        preset is selected and *flow_path* is not given, the current directory
        is searched for ``flow_conditions.dat`` automatically.
    """

    # coerce flow_path to a Path (default: flow_conditions.dat in cwd)
    if(flow_path) is not None:
        flow_path = Path(flow_path)
    else:
        flow_path = Path("flow_conditions.dat")

    # start from defaults, optionally overlay a geometry template
    default_cfg = copy.deepcopy(Config().to_dict())

    # if a --geometry preset is specified, merge those values into the default config
    # geometry presets are stored in config and provide flags for common geometries (cone, ogive, flat-plate, cylinder)
    if geometry is not None:
        default_cfg = merge_dicts(default_cfg, GEOMETRY_TEMPLATES[geometry])

    # merge flow conditions into the default config (from --flow path or auto-discovered flow_conditions.dat)
    cfg_init = merge_flow_defaults(default_cfg, flow_path)

    # overwrite guard:
    # if the file already exists and --force is not set, do not overwrite
    if out.exists() and not force:
        typer.echo(
            f"{out.resolve()} already exists; use --force to replace."
        )
        return

    try:
        # write the config file (overwrite=True since we already checked above)
        result = write_config(
            out, overwrite=True, cfg_data=cfg_init
        )
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    # output for user
    typer.echo(f"wrote config -> {result.resolve()}")
