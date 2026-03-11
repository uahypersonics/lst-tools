"""lst-tools parsing — generate a runnable input deck for the parsing step."""


# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from lst_tools.config import find_config, read_config
from lst_tools.setup.parsing import parsing_setup


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'parsing' cli option
# --------------------------------------------------
def cmd_parsing(
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Path to configuration file (e.g. lst.cfg)."),
    ] = None,
    out: Annotated[
        Path, typer.Option("--out", "-o", help="Directory to place the input deck.")
    ] = Path("."),
    name: Annotated[
        str, typer.Option("--name", "-n", help="Filename for the input deck.")
    ] = "lst_input.dat",
    auto_fill: Annotated[
        bool,
        typer.Option(
            "--auto-fill", "-a",
            help="Provide suggestions for x_s, x_e, i_step, f_min, f_max, d_f, beta_min, beta_max, d_beta based on meanflow.bin.",
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
    """Generate a runnable input deck for the global LST solver (parsing step).
\f

    Workflow:

    1. Load the project config (auto-discovered or via ``--cfg``).
    2. Call ``parsing_setup()`` to write the LST input deck.
    3. If ``--auto-fill`` is set, derive ``x_s``, ``x_e``, ``i_step``, ``f_min``, ``f_max``, ``d_f``, ``beta_min``, ``beta_max``, and ``d_beta``
       from the meanflow binary for any fields left unset in the config.
    4. Echo the path of the written input deck.
    """

    # debug output for devs
    logger.debug("setting up input deck for parsing step")

    try:
        # load config (read_config.py in /config/)
        config = read_config(path=cfg)

        # resolve cfg_path for write-back (auto-discover if --cfg not given)
        resolved_cfg_path = cfg if cfg is not None else find_config(".")

        # generate the input deck
        written = parsing_setup(
            cfg=config,
            out_dir=out,
            out_name=name,
            auto_fill=auto_fill,
            force=force,
            cfg_path=resolved_cfg_path,
        )

        # output for user
        typer.echo(written)

    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("parsing setup failed", exc_info=True)
        raise typer.Exit(1)
