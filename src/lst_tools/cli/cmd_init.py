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
import re
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
# helper function to tailor the generated init config
# --------------------------------------------------
def _prepare_init_config(cfg_data: dict) -> dict:
    """Prepare a user-facing init config from the schema defaults."""

    # copy the processing section so init can shape a cleaner scaffold
    processing_cfg = copy.deepcopy(cfg_data.get("processing", {}))

    # build a spectra block that exposes the most useful gating knobs up front
    spectra_cfg = copy.deepcopy(processing_cfg.get("spectra", {}))
    processing_cfg["spectra"] = {
        "alpr_min": spectra_cfg.get("alpr_min"),
        "alpr_max": spectra_cfg.get("alpr_max"),
        "alpi_min": spectra_cfg.get("alpi_min"),
        "alpi_max": spectra_cfg.get("alpi_max"),
    }

    # omit parsing until that workflow actually has user-facing options
    processing_cfg.pop("parsing", None)

    # write the tailored processing block back into the config seed
    cfg_data["processing"] = processing_cfg

    # omit the extract section: it belongs to the separate 'extract' subcommand
    # and would only confuse a new user filling in the standard solver workflow
    cfg_data.pop("extract", None)

    return cfg_data


def _inject_init_comments(config_text: str) -> str:
    """Inject short guidance comments into the init-generated TOML text."""

    # --------------------------------------------------
    # inline field comments
    #
    # the generated TOML uses single-space "key = value" formatting, so each
    # target string below is unique and a plain str.replace matches exactly
    # one line; the comment is appended after two spaces
    # --------------------------------------------------
    field_comments = [
        ('mach = ""', "REQUIRED — freestream Mach number"),
        ('re1 = ""', "REQUIRED — unit Reynolds number [1/m]"),
        ("gamma = 1.4", "ratio of specific heats"),
        ("cp = 1005.025", "specific heat at constant pressure [J/(kg K)]"),
        ("cv = 717.875", "specific heat at constant volume [J/(kg K)]"),
        ("rgas = 287.15", "specific gas constant [J/(kg K)]"),
        (
            'pres_0 = ""',
            "stagnation pressure [Pa]  (optional — not used by lst.x)",
        ),
        (
            'temp_0 = ""',
            "stagnation temperature [K]  (optional — not used by lst.x)",
        ),
        (
            'pres_inf = ""',
            "freestream pressure [Pa]  (optional — not used by lst.x)",
        ),
        ('temp_inf = ""', "REQUIRED — freestream temperature [K]"),
        (
            'dens_inf = ""',
            "freestream density [kg/m^3]  (optional — not used by lst.x)",
        ),
        (
            'uvel_inf = ""',
            "freestream velocity [m/s]  (optional — not used by lst.x)",
        ),
        ("visc_law = 0", "viscosity law: 0 = Sutherland, 1 = power law"),
        ("i_s = 0", "first station index to convert (0-based)"),
        ('i_e = ""', "REQUIRED — last station index (inclusive)"),
        ("d_i = 1", "station stride"),
        ('x_s = ""', "REQUIRED — sweep start x-station"),
        ('x_e = ""', "REQUIRED — sweep end x-station"),
        ('i_step = ""', "REQUIRED — station index step"),
        ('f_min = ""', "REQUIRED — minimum frequency [Hz]"),
        ('f_max = ""', "REQUIRED — maximum frequency [Hz]"),
        ('d_f = ""', "REQUIRED — frequency step [Hz]"),
        ('account = ""', "REQUIRED for HPC — scheduler account name"),
        ('nodes = ""', "REQUIRED for HPC — node count"),
        ('time = ""', "REQUIRED for HPC — wall-time limit (HH:MM:SS)"),
        ('partition = ""', "REQUIRED for HPC — queue or partition name"),
    ]

    # apply each field comment once, guarding against double-injection
    for target, comment in field_comments:
        commented_line = f"{target}  # {comment}"
        if commented_line not in config_text:
            config_text = config_text.replace(target, commented_line, 1)

    # --------------------------------------------------
    # input_file value can be auto-detected, so a fixed-string replace cannot
    # match it — append the comment to the line found by its key prefix
    # --------------------------------------------------
    if "# path to HDF5 baseflow file" not in config_text:
        config_text = re.sub(
            r"^(input_file = .*)$",
            r"\1  # path to HDF5 baseflow file",
            config_text,
            count=1,
            flags=re.MULTILINE,
        )

    # --------------------------------------------------
    # section-level comment blocks, injected directly above each section header
    # --------------------------------------------------
    section_blocks = [
        (
            "[flow_conditions]",
            "# -- flow conditions -------------------------------------------------------\n"
            "# Fill mach, re1, and temp_inf before running any subcommand.\n"
            "[flow_conditions]",
        ),
        (
            "[geometry]",
            "# -- geometry --------------------------------------------------------------\n"
            "# type: 0=flat-plate  1=ogive  2=cone  3=sphere-cone  4=cylinder  5=custom\n"
            "[geometry]",
        ),
        (
            "[meanflow_conversion]",
            "# -- meanflow conversion ---------------------------------------------------\n"
            "# Controls which stations are written to meanflow.bin by lst-tools lastrac.\n"
            "[meanflow_conversion]",
        ),
        (
            "[lst.solver]",
            "# -- LST solver ------------------------------------------------------------\n"
            "[lst.solver]",
        ),
        (
            "[lst.params]",
            "# -- solver sweep parameters -----------------------------------------------\n"
            "# Set x_s/x_e/i_step and f_min/f_max/d_f before running lst-tools setup parsing.\n"
            "[lst.params]",
        ),
        (
            "[hpc]",
            "# -- HPC scheduler ---------------------------------------------------------\n"
            "# Leave blank if running interactively.\n"
            "[hpc]",
        ),
        (
            "[processing.tracking]",
            "# -- post-processing -------------------------------------------------------\n"
            "[processing.tracking]",
        ),
    ]

    # apply each section block once, guarding against double-injection
    for header, block in section_blocks:
        if block not in config_text:
            config_text = config_text.replace(header, block, 1)

    # --------------------------------------------------
    # spectra header guidance (existing behaviour, kept intact)
    # --------------------------------------------------
    spectra_header = "[processing.spectra]"
    spectra_header_with_comments = (
        "[processing.spectra]\n"
        "# Optional alpha-space gates for spectra post-processing.\n"
        "# Leave any bound empty to disable it."
    )
    if spectra_header_with_comments not in config_text:
        config_text = config_text.replace(
            spectra_header,
            spectra_header_with_comments,
            1,
        )

    return config_text


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

    # build a cleaner user-facing init scaffold
    cfg_init = _prepare_init_config(cfg_init)

    # auto-detect HDF5 meanflow file in the current directory
    h5_files = list(Path(".").glob("*.h5")) + list(Path(".").glob("*.hdf5"))
    if len(h5_files) == 1:
        cfg_init["input_file"] = h5_files[0].name
        logger.info("auto-detected meanflow file: %s", h5_files[0].name)

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

        # write short guidance comments into the generated scaffold
        if result.exists():
            config_text = result.read_text(encoding="utf-8")
            config_text = _inject_init_comments(config_text)
            result.write_text(config_text, encoding="utf-8")
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1)

    # output for user
    typer.echo(f"wrote config -> {result.resolve()}")
