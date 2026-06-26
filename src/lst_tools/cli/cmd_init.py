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

    return cfg_data


def _inject_init_comments(config_text: str) -> str:
    """Inject short guidance comments into the init-generated TOML text."""

    # --------------------------------------------------
    # above-line field comments
    #
    # each entry is (target_string, comment_text); the comment is injected as
    # a standalone "# ..." line immediately above the matching field line.
    # str.replace uses count=1 so only the first occurrence is annotated — for
    # fields that appear in multiple sections with the same key=value
    # (gate_tol = 0.1), the second occurrence is handled via regex below.
    # --------------------------------------------------
    above_line_comments = [
        # top-level
        ('lst_exe = "lst.x"', "lst solver executable name"),
        # flow_conditions
        ('mach = ""', "freestream mach number (required)"),
        ('re1 = ""', "unit reynolds number (1/m) (required)"),
        ("gamma = 1.4", "ratio of specific heats"),
        ("cp = 1005.025", "specific heat at constant pressure [J/(kg K)]"),
        ("cv = 717.875", "specific heat at constant volume [J/(kg K)]"),
        ("rgas = 287.15", "specific gas constant [J/(kg K)]"),
        ('pres_0 = ""', "stagnation pressure [Pa] (optional)"),
        ('temp_0 = ""', "stagnation temperature [K] (optional)"),
        ('pres_inf = ""', "freestream pressure [Pa] (optional)"),
        ('temp_inf = ""', "freestream temperature [K] (required)"),
        ('dens_inf = ""', "freestream density [kg/m^3] (optional)"),
        ('uvel_inf = ""', "freestream velocity [m/s] (optional)"),
        ("visc_law = 0", "viscosity law: 0 = Sutherland, 1 = power law"),
        # geometry
        (
            'type = ""',
            "geometry type (required): 0=flat-plate, 1=cylinder, "
            "2=cone, 3=generalized-axisymmetric (ogive, flared cone, ...)",
        ),
        ('theta_deg = ""', "half-angle [deg] — cone and generalized-axisymmetric geometries"),
        ('r_nose = ""', "nose radius [m] — cone and generalized-axisymmetric geometries"),
        ("l_ref = 1.0", "reference length [m]"),
        (
            "is_body_fitted = false",
            "cone only: true if grid is body-fitted "
            "(radius = x·sin θ + r_nose·cos θ); false uses y-coordinate as radius",
        ),
        # meanflow_conversion
        ("i_s = 0", "first station index to convert (0-based)"),
        ('i_e = ""', "last station index, inclusive (required)"),
        ("d_i = 1", "station stride"),
        ("set_v_zero = true", "zero wall-normal velocity in the converted meanflow"),
        # lst.solver
        ("type = 1", "solver type: 1=global parallel, 2=tracking, 3=3-D tracking"),
        ("is_simplified = true", "use simplified (adiabatic) energy equation"),
        ("alpha_i_threshold = -100.0", "discard modes with growth rate below this threshold"),
        ("generalized = 0", "generalized eigenvalue formulation (0=standard, 1=generalized)"),
        ("spatial_temporal = 1", "analysis type: 1=spatial, 2=temporal"),
        ("energy_formulation = 1", "energy equation formulation"),
        # lst.options
        ('geometry_switch = ""', "override geometry type for the internal solver"),
        ("longitudinal_curvature = 0", "include longitudinal curvature effects (0=off, 1=on)"),
        # lst.params
        ("ny = 150", "number of wall-normal Chebyshev points"),
        ("yl_in = 0.0", "inner wall-normal boundary location"),
        ("tol_lst = 1e-05", "eigenvalue convergence tolerance"),
        ("max_iter = 15", "maximum secant iterations"),
        ('x_s = ""', "sweep start x-station (required)"),
        ('x_e = ""', "sweep end x-station (required)"),
        ('i_step = ""', "station index step (required)"),
        ("tracking_dir = 1", "tracking direction: 1=downstream, -1=upstream"),
        ('f_min = ""', "minimum frequency [Hz] (required)"),
        ('f_max = ""', "maximum frequency [Hz] (required)"),
        ('d_f = ""', "frequency step [Hz] (required)"),
        ("f_init = 0.0", "initial frequency for tracking (0 = use f_min)"),
        ('beta_s = ""', "spanwise wavenumber sweep start"),
        ('beta_e = ""', "spanwise wavenumber sweep end"),
        ('d_beta = ""', "spanwise wavenumber step"),
        ("beta_init = 0.0", "initial spanwise wavenumber"),
        ('alpha_0 = "(0.0,0.0)"', "initial eigenvalue guess (real, imag)"),
        # lst.io
        ('baseflow_input = "meanflow.bin"', "meanflow binary file written by lst-tools lastrac"),
        ('solution_output = "growth_rate.dat"', "output file for eigenvalue results"),
        # hpc
        ('account = ""', "scheduler account name (required)"),
        ('nodes = ""', "number of nodes (required)"),
        ('time = ""', "wall-time limit in HH:MM:SS (required)"),
        ('partition = ""', "queue or partition name (required)"),
        ('extra_env = ""', "extra environment variables to pass to the job"),
        # processing.tracking
        ("interpolate = false", "interpolate tracking results onto a regular grid"),
        ("gate_tol = 0.1", "growth rate acceptance tolerance"),  # first occurrence → tracking
        ("min_valid = 40", "minimum valid stations required to keep a mode"),
        ("peak_order = 1", "order of the peak-detection filter"),
        # processing.spectra
        ('alpr_min = ""', "lower bound on alpha_real gate (optional)"),
        ('alpr_max = ""', "upper bound on alpha_real gate (optional)"),
        ('alpi_min = ""', "lower bound on alpha_imag gate (optional)"),
        ('alpi_max = ""', "upper bound on alpha_imag gate (optional)"),
        # seed_table
        ("enabled = false", "enable automatic seed selection from seed table"),
        ('source_file = ""', "path to the seed table file"),
        ("n_seeds = 12", "number of seed eigenvalues to use per station"),
        ("min_growth = 10.0", "minimum growth rate for seed selection [1/m]"),
        # gate_tol = 0.1 second occurrence (seed_table) handled separately below
        ("min_valid = 5", "minimum number of valid points to accept a seed"),
        ("smooth_passes = 0", "number of smoothing passes over seed eigenvalues"),
        ("gate_by_keep_mask = true", "restrict seeds to the keep-mask region"),
        ("x_range = []", "x-station range for seed selection (empty = all)"),
        ("f_range = []", "frequency range for seed selection [Hz] (empty = all)"),
        ("threshold = 0.15", "N-factor threshold for seed table generation"),
        ('output_file = "seed_alpha.dat"', "output path for the generated seed table"),
        # extract
        ('input_file = ""', "path to Tecplot FE-quad mesh file"),
        ('hdf5_out = ""', "output HDF5 file path (default: extracted_baseflow.hdf5)"),
        ('profiles_out = ""', "output path for wall-normal profiles Tecplot ASCII file (optional)"),
        ('wall_out = ""', "output path for wall curve Tecplot ASCII file (optional)"),
        ('surface = ""', "surface to extract: lower or upper (default: lower)"),
        ('n_eta = ""', "number of wall-normal sample points (default: 200)"),
        ('eta_distribution = ""', "point distribution: cosine or linear (default: cosine)"),
        ('stations = ""', "list of x-stations to extract; example: [0.1, 0.2, 0.3]"),
    ]

    # apply each above-line comment once, guarding against double-injection
    for target, comment in above_line_comments:
        above_line = f"# {comment}\n{target}"
        if above_line not in config_text:
            config_text = config_text.replace(target, above_line, 1)

    # --------------------------------------------------
    # gate_tol = 0.1 appears in both [processing.tracking] and [seed_table];
    # the loop above annotated the first occurrence (tracking).  now annotate
    # the second occurrence (seed_table) using a fixed-length negative
    # lookbehind so the already-annotated tracking line is not double-injected.
    # --------------------------------------------------
    _tracking_gate_comment = "# growth rate acceptance tolerance\n"
    if _tracking_gate_comment + "gate_tol = 0.1" in config_text:
        config_text = re.sub(
            r"(?<!# growth rate acceptance tolerance\n)^gate_tol = 0\.1$",
            "# growth rate acceptance tolerance\ngate_tol = 0.1",
            config_text,
            count=1,
            flags=re.MULTILINE,
        )

    # --------------------------------------------------
    # top-level input_file: value is auto-detected so use regex to locate the
    # line; match only non-empty values to avoid re-annotating the extract
    # section's input_file = "" (already handled by above_line_comments above)
    # --------------------------------------------------
    if "# path to HDF5 baseflow file (required)" not in config_text:
        config_text = re.sub(
            r'^(input_file = ".+")$',
            r"# path to HDF5 baseflow file (required)\n\1",
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
            "# --------------------------------------------------\n"
            "# flow conditions\n"
            "# --------------------------------------------------\n\n"
            "[flow_conditions]",
        ),
        (
            "[geometry]",
            "# --------------------------------------------------\n"
            "# geometry\n"
            "# --------------------------------------------------\n\n"
            "[geometry]",
        ),
        (
            "[meanflow_conversion]",
            "# --------------------------------------------------\n"
            "# meanflow conversion\n"
            "# --------------------------------------------------\n\n"
            "[meanflow_conversion]",
        ),
        (
            "[lst.solver]",
            "# --------------------------------------------------\n"
            "# LST solver\n"
            "# --------------------------------------------------\n\n"
            "[lst.solver]",
        ),
        (
            "[lst.params]",
            "# --------------------------------------------------\n"
            "# solver sweep parameters\n"
            "# --------------------------------------------------\n\n"
            "[lst.params]",
        ),
        (
            "[hpc]",
            "# --------------------------------------------------\n"
            "# HPC scheduler\n"
            "# --------------------------------------------------\n\n"
            "[hpc]",
        ),
        (
            "[extract]",
            "# --------------------------------------------------\n"
            "# extract (optional)\n"
            "# --------------------------------------------------\n\n"
            "[extract]",
        ),
        (
            "[processing.tracking]",
            "# --------------------------------------------------\n"
            "# post-processing\n"
            "# --------------------------------------------------\n\n"
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
