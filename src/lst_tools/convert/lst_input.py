"""Generate LASTRAC input decks from a configuration object.

Formats all solver parameters, flow conditions, and options into
the fixed-format text file that LASTRAC reads as ``input.dat``.
"""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------
from __future__ import annotations
import logging
from pathlib import Path
from pprint import pformat
from lst_tools.config.schema import Config


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# helper functions for formatting values for the input deck
# --------------------------------------------------
def _fmt_bool_01(v: bool) -> str:
    """Format a boolean as ``'1'`` (True) or ``'0'`` (False)."""
    return "1" if bool(v) else "0"


def _fmt_bool_tf(v: bool) -> str:
    """Format a boolean as ``'T'`` (True) or ``'F'`` (False)."""
    return "T" if bool(v) else "F"


def _fmt_i(n: int, width: int = 0) -> str:
    """Format an integer, optionally left-padded to *width*."""
    return f"{n:<{width}d}" if width else str(n)


def _fmt_f(x: float, width: int = 0, prec: int = 10) -> str:
    """Format a float with *prec* decimal places, optionally padded to *width*."""
    s = f"{x:<.{prec}f}"
    return f"{s:<{width}}" if width else s


def _fmt_with_comment(values: str, comment: str, width: int = 50) -> str:
    """Append an inline comment aligned at *width* characters."""
    return f"{values:<{width}} : {comment}"


# --------------------------------------------------
# main function to generate the input deck
# --------------------------------------------------
def generate_lst_input_deck(
    *,
    cfg: Config | None = None,
    out_path: str | Path,
) -> Path:
    """Write a complete LASTRAC input deck to *out_path*.

    Args:
        cfg (Config | None): Configuration object.  ``None`` falls back
            to schema defaults.
        out_path (str | Path): Destination file path for the input deck.

    Returns:
        Path: The path the input deck was written to.
    """

    # if no config is provided, use defaults from the schema
    if cfg is None:
        cfg = Config()

    # fail explicitly if any of the required parameters is not set (i.e. None)
    required_parameters = {
        "x_s": cfg.lst.params.x_s,
        "x_e": cfg.lst.params.x_e,
        "i_step": cfg.lst.params.i_step,
        "f_min": cfg.lst.params.f_min,
        "f_max": cfg.lst.params.f_max,
        "d_f": cfg.lst.params.d_f,
        "beta_s": cfg.lst.params.beta_s,
        "beta_e": cfg.lst.params.beta_e,
        "d_beta": cfg.lst.params.d_beta,
    }

    # collect all (if any) missing parameters and raise a ValueError with instructions
    missing = []
    for key, val in required_parameters.items():
        if val is None:
            missing.append(key)

    if missing:
        raise ValueError(
            "the following lst.params fields are required:\n"
            + "\n".join(f"  - {name}" for name in missing)
            + "\nSet them in the config file or rerun command with --auto-fill option."
        )

    # ensure out_path is a Path object
    out = Path(out_path)

    # initialize input deck as empty list of strings
    lines: list[str] = []

    # create an alias to the lines.append function for convenience
    add = lines.append

    # debug output for devs
    logger.debug("flow conditions")
    logger.debug(pformat(cfg.flow_conditions))

    logger.debug("solver configuration")
    logger.debug(pformat(cfg.lst.solver))

    # --------------------------------------------------
    # solver type and problem type
    # --------------------------------------------------

    # 1: Global Parallel Solver (no initial guess required)
    # 2: Tracking Solver (single initial guess required)

    add("")
    add("SOLVER TYPE AND PROBLEM TYPE")
    add("=" * 126)

    vals = f"{_fmt_i(cfg.lst.solver.type, 1)},{_fmt_bool_tf(cfg.lst.solver.is_simplified)},{_fmt_f(cfg.lst.solver.alpha_i_threshold, 20, 10)}"
    add(_fmt_with_comment(vals, "Solver type, is_simplified, alpha_i_threshold"))

    vals = f"{_fmt_i(cfg.lst.solver.generalized, 1)}"
    add(_fmt_with_comment(vals, "Generalized (1 -> Orthogonal curvilinear coordinates solver, 0 -> Conical coordinates solver)"))

    vals = f"{_fmt_i(cfg.lst.solver.spatial_temporal, 1)}"
    add(_fmt_with_comment(vals, "Spatial/Temporal problem (1 -> spatial, 0 -> temporal)"))

    vals = f"{_fmt_i(cfg.lst.solver.energy_formulation, 1)}"
    add(_fmt_with_comment(vals, "Formulation for energy equation (DpDtFormulation)"))

    # --------------------------------------------------
    # geometry parameters
    # --------------------------------------------------

    logger.debug("lst geometry options")
    logger.debug(pformat(cfg.lst.options))

    # coerce optional geometry fields — None means "not applicable" (flat plate)
    theta_deg = cfg.geometry.theta_deg
    if theta_deg is None:
        logger.warning("no cone half angle specified, defaulting to 0.0 degrees")
        theta_deg = 0.0

    geometry_switch = cfg.lst.options.geometry_switch
    if geometry_switch is None:
        geometry_switch = 0

    add("")
    add("GEOMETRY PARAMETERS")
    add("=" * 126)

    vals = f"{_fmt_i(geometry_switch, 1)}"
    add(_fmt_with_comment(vals, "Geometry switch (cone = 0 is for flat plate, cone = 1 is for conical geometry)"))

    vals = f"{_fmt_f(theta_deg, 20, 10)}"
    add(_fmt_with_comment(vals, "cone half angle in degree (theta_c_deg)"))

    vals = f"{_fmt_bool_01(cfg.lst.options.longitudinal_curvature)}"
    add(_fmt_with_comment(vals, "Longitudinal Curvature (0 -> no longitudinal curvature, 1 -> with longitudinal curvature)"))

    # --------------------------------------------------
    # stability parameters
    # --------------------------------------------------

    logger.debug("lst parameters")
    logger.debug(pformat(cfg.lst.params))

    add("")
    add("STABILITY PARAMETERS")
    add("=" * 126)
    add("")

    add("Stability grid parameters")
    add("-------------------------")

    vals = f"{_fmt_i(cfg.lst.params.ny, 4)}"
    add(_fmt_with_comment(vals, "Number of points in y-direction (ny)"))

    vals = f"{_fmt_f(cfg.lst.params.yl_in, 20, 10)}"
    add(_fmt_with_comment(vals, "Mapping parameter (yl_in)"))

    add("")
    add("Secant iteration parameters")
    add("---------------------------")

    vals = f"{_fmt_f(cfg.lst.params.tol_lst, 23, 10)}"
    add(_fmt_with_comment(vals, "Tolerance for Secant Iteration (tol_lst)"))

    vals = f"{_fmt_i(cfg.lst.params.max_iter, 6)}"
    add(_fmt_with_comment(vals, "Maximum number of iterations for Secant Iteration (max_iter)"))

    # space sweep parameters

    add("")
    add("Space sweep parameters")
    add("----------------------")

    vals = f"{_fmt_f(cfg.lst.params.x_s, 20, 10)}"
    add(_fmt_with_comment(vals, "Start x location for tracking (x_min)"))

    vals = f"{_fmt_f(cfg.lst.params.x_e, 20, 10)}"
    add(_fmt_with_comment(vals, "End x location for tracking (x_max)"))

    vals = f"{_fmt_i(cfg.lst.params.i_step, 6)}"
    add(_fmt_with_comment(vals, "Point increment between x_min and x_max (istep)"))

    # frequency sweep parameters

    add("")
    add("Frequency sweep parameters (Only Relevant for Spatial Problem)")
    add("--------------------------")

    vals = f"{_fmt_f(cfg.lst.params.f_min, 25, 10)}"
    add(_fmt_with_comment(vals, "Minimum frequency for tracking (fmin)"))

    vals = f"{_fmt_f(cfg.lst.params.f_max, 25, 10)}"
    add(_fmt_with_comment(vals, "Maximum frequency for tracking (fmax)"))

    vals = f"{_fmt_f(cfg.lst.params.d_f, 25, 10)}"
    add(_fmt_with_comment(vals, "Frequency increment (between fmin and fmax)"))

    vals = f"{_fmt_f(cfg.lst.params.f_init, 25, 10)}"
    add(_fmt_with_comment(vals, "Initial frequency (frequency of Initial guess/eigenvalue) (finit)"))

    # wavenumbers and initial guess/eigenvalue

    add("")
    add("Wavenumbers and initial guess/eigenvalue")
    add("----------------------------------------")

    # hardcoded line for temporal problem
    add("100.000                 : Longitudinal wavenumber (alphaTemp_in) ==== ONLY RELEVANT FOR TEMPORAL PROBLEM")

    vals = f"{_fmt_f(cfg.lst.params.beta_s, 20, 10)}, {_fmt_f(cfg.lst.params.beta_e, 20, 10)}, {_fmt_f(cfg.lst.params.d_beta, 20, 10)}, {_fmt_f(cfg.lst.params.beta_init, 20, 10)}"
    add(_fmt_with_comment(vals, "Spanwise/Azimuthal wavenumber (beta_in): beta_min, beta_max, beta_step, beta_init"))

    alpha_0 = cfg.lst.params.alpha_0
    vals = f"({alpha_0.real:.10f},{alpha_0.imag:.10f})"
    add(_fmt_with_comment(vals, "Complex Initial guess/eigenvalue (alpha0_init)"))

    # --------------------------------------------------
    # baseflow options and scaling
    # --------------------------------------------------

    add("")
    add("")
    add("BASEFLOW OPTIONS AND SCALING")
    add("=" * 126)
    add("")

    add("General parameters")
    add("------------------")

    vals = f"{_fmt_i(cfg.flow_conditions.visc_law, 1)}"
    add(_fmt_with_comment(vals, "Viscosity law type (0 -> sutherlands law, 1 -> sutherlands law with low tempearture correction)"))

    add("")
    add("Baseflow parameters")
    add("------------------")

    vals = f"{_fmt_f(cfg.flow_conditions.gamma, 20, 10)}"
    add(_fmt_with_comment(vals, "Specific heat ratio (gamma)"))

    vals = f"{_fmt_f(cfg.flow_conditions.cp, 20, 10)}"
    add(_fmt_with_comment(vals, "Specific heat at constant pressure (cp)"))

    # --------------------------------------------------
    # io options
    # --------------------------------------------------

    add("")
    add("INPUT/OUTPUT OPTIONS")
    add("=" * 126)

    vals = f"{cfg.lst.io.baseflow_input}"
    add(_fmt_with_comment(vals, "mean flow input file name"))

    vals = f"{cfg.lst.io.solution_output}"
    add(_fmt_with_comment(vals, "growth rate output file name"))

    # --------------------------------------------------
    # write output file
    # --------------------------------------------------

    # create output directory if it does not exist
    out.parent.mkdir(parents=True, exist_ok=True)

    # write the input deck
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return out

