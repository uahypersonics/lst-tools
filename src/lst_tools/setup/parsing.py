"""Set up a parsing case (global LST solve).

- read and validate config
- ensure output directory exists
- generate a runnable input deck (input.dat)
- generate an HPC run script (if a scheduler is detected)
"""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from lst_tools.config import write_config
from lst_tools.convert.lst_input import generate_lst_input_deck
from lst_tools.hpc import hpc_configure, script_build

from ._common import read_baseflow_profiles, read_baseflow_stations, resolve_config


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# estimate relevant frequency range from baseflow quantities
# --------------------------------------------------
def estimate_freq(
    samples: dict,
    mach: float,
    gamma: float = 1.4,
    uvel_inf: float = 1.0,
) -> float:
    r"""Estimate relevant frequency range from baseflow quantities.

    Uses the Bertin criterion to determine the boundary-layer thickness:
    the edge is located where the wall-normal gradient of total enthalpy
    vanishes (:math:`\partial h_0 / \partial \eta = 0`).

    Total enthalpy (non-dimensional):

    .. math::

        h_0 = \frac{T}{M^2 (\gamma - 1)}
              + \frac{1}{2}(u^2 + v^2 + w^2)

    The estimated maximum frequency is then:

    .. math::

        f_{\max} = \frac{0.9\, U_e}{n\, \delta}

    where *n* = 2 for M > 4 (second-mode dominant) and *n* = 10 for
    M ≤ 4 (first-mode dominant).  The median :math:`\delta` across
    sampled stations is used for robustness.

    Parameters
    ----------
    samples : dict
        Output of :func:`read_baseflow_profiles` with keys
        ``eta``, ``uvel``, ``vvel``, ``wvel``, ``temp``.
    mach : float
        Edge Mach number from config.
    gamma : float, optional
        Ratio of specific heats (default 1.4).
    uvel_inf : float, optional
        Freestream velocity used to redimensionalize the
        nondimensional edge velocity (default 1.0).

    Returns
    -------
    float
        Estimated f_max.
    """

    gm1 = gamma - 1.0

    delta_list: list[float] = []
    u_e_list: list[float] = []

    for eta, uvel, vvel, wvel, temp in zip(
        samples["eta"],
        samples["uvel"],
        samples["vvel"],
        samples["wvel"],
        samples["temp"],
    ):

        # skip if there are too few points to compute a gradient
        if len(eta) < 3:
            continue

        # edge velocity is the velocity at the last point (assuming profiles are ordered wall to edge)
        u_e = uvel[-1]
        #
        # if velocity is zero ad the edge skip the profile
        if u_e == 0.0:
            continue

        # total enthalpy (nondimensional form)
        # h0 = T / (M^2*(gamma-1)) + 0.5*(u^2 + v^2 + w^2)
        h0 = temp / (mach**2 * gm1) + 0.5 * (uvel**2 + vvel**2 + wvel**2)

        # BL edge: first eta where h0 reaches 99% of the freestream value
        # For near-adiabatic walls h0 is ~constant, so fall back to
        # 99% of the velocity profile when h0 gives no useful edge.
        h0_e = h0[-1]
        if h0_e != 0.0 and (h0[0] / h0_e) < 0.99:
            idx = np.searchsorted(h0 / h0_e, 0.99)
        else:
            # adiabatic-wall fallback: use velocity profile
            ratio = uvel / u_e
            idx = np.searchsorted(ratio, 0.99)

        if idx >= len(eta):
            idx = len(eta) - 1
        if eta[idx] == 0.0:
            continue

        delta_list.append(float(eta[idx]))
        u_e_list.append(float(u_e))

    if len(delta_list) == 0:
        logger.warning(
            "[auto-fill] could not compute BL thickness from any station"
        )
        # fallback to f_max = 100000 if boundary layer thickness cannot be estimated
        return 100000

    delta_median = float(np.median(delta_list))
    u_e_mean = float(np.mean(u_e_list)) * uvel_inf
    c_ph = 0.9 * u_e_mean

    n = 2.0 if mach > 4.0 else 10.0
    relaxation_factor = 0.5
    f_max = relaxation_factor*c_ph / (n * delta_median)

    # round up to nearest 10 kHz
    f_max = math.ceil(f_max / 10000) * 10000

    logger.info(
        "[auto-fill] f_max estimate (Bertin): median delta = %g, "
        "mean U_e = %g, c_ph = %g, n = %g -> f_max = %g",
        delta_median, u_e_mean, c_ph, n, f_max,
    )

    return f_max


# --------------------------------------------------
# auto-fill unset parsing parameters from meanflow data
# --------------------------------------------------
def auto_fill_parsing(cfg: Any, *, force: bool = False, cfg_path: str | Path | None = None) -> bool:
    """Derive sweep parameters from the meanflow binary for unset fields.

    Only fields that are currently ``None`` are filled unless *force* is
    ``True``, in which case all fields are overwritten.  If *cfg_path* is
    given the updated configuration is written back to disk so the user can
    inspect (and later tweak) what was chosen.

    **Space sweep** (``x_s``, ``x_e``, ``i_step``):
    Derived from the meanflow station coordinates.

    **Frequency sweep** (``f_min``, ``f_max``, ``d_f``):
    ``f_min`` defaults to 0.  ``f_max`` is estimated from the boundary
    layer thickness and edge velocity when possible.

    **Wavenumber sweep** (``beta_s``, ``beta_e``, ``d_beta``):
    Uses sensible non-dimensional defaults.

    Parameters
    ----------
    cfg : Config
        A validated configuration object (dataclass).
    force : bool, optional
        When ``True``, overwrite all parameters even if already set.
    cfg_path : str or Path, optional
        Path to the config file on disk.  When provided the config is
        written back with ``overwrite=True`` after filling.

    Returns
    -------
    bool
        ``True`` if at least one field was filled, ``False`` otherwise.
    """

    # alias for parameters
    params = cfg.lst.params

    # alias for meanflow path
    baseflow_path = Path(cfg.lst.io.baseflow_input)

    # initialize flag to track if any fields were auto filled so we know whether to write the updated config back to disk at the end
    changed = False

    # streamwise parameters: x_s, x_e, i_step
    x_params = [params.x_s, params.x_e, params.i_step]

    autofill_x = force or any(val is None for val in x_params)

    if autofill_x and not baseflow_path.exists():
        logger.warning(
            "[auto-fill] meanflow file '%s' not found — cannot autofill"
            " space sweep parameters (x_s, x_e, i_step)",
            baseflow_path,
        )
    elif autofill_x:

        x_stations = read_baseflow_stations(baseflow_path)
        n_stations = len(x_stations)

        if n_stations == 0:
            logger.warning(
                "[auto-fill] meanflow contains no stations — cannot autofill"
                " space sweep parameters"
            )
        else:

            if params.x_s is None or force:
                # set x_s to the first station location if it is not provided
                params.x_s = float(x_stations[0])
                logger.info("[auto-fill] x_s = %g  (first station)", params.x_s)
                changed = True

            if params.x_e is None or force:
                # set x_e to the last station location if it is not provided
                params.x_e = float(x_stations[-1])
                logger.info("[auto-fill] x_e = %g  (last station)", params.x_e)
                changed = True

            if params.i_step is None or force:
                # set i_step to ceil(n_stations / 100) to give ~100 points in the streamwise direction
                params.i_step = max(1, math.ceil(n_stations / 100))
                logger.info(
                    "[auto-fill] i_step = %d  (ceil(%d stations / 100))",
                    params.i_step,
                    n_stations,
                )
                changed = True

    # frequency parameters: f_min, f_max, d_f

    # set lower frequency to 0 if not provided
    if params.f_min is None or force:
        params.f_min = 0.0
        logger.info("[auto-fill] f_min = %g", params.f_min)
        changed = True

    # estimate upper frequency from baseflow if not provided
    if params.f_max is None or force:
        # get mach number
        mach = cfg.flow_conditions.mach

        if mach is None:
            logger.warning(
                "[auto-fill] flow_conditions.mach not set — using f_max = 100000"
            )
            params.f_max = 100000
        elif not baseflow_path.exists():
            logger.warning(
                "[auto-fill] meanflow file '%s' not found — using f_max = 100000",
                baseflow_path,
            )
            params.f_max = 100000
        else:
            # read base flow profiles
            profiles = read_baseflow_profiles(baseflow_path)

            # check if crossflow is presnet (if yes, esitmates are not good)
            has_crossflow = any(
                np.any(w != 0.0) for w in profiles["wvel"]
            )
            if has_crossflow:
                logger.warning(
                    "[auto-fill] crossflow detected (wvel != 0)"
                    " — cannot estimate f_max automatically, using 100000"
                )
                params.f_max = 100000
            else:
                # estimate f_max from boundary layer thickness and edge velocity
                gamma = cfg.flow_conditions.gamma
                uvel_inf = cfg.flow_conditions.uvel_inf or 1.0
                params.f_max = estimate_freq(profiles, mach, gamma, uvel_inf)

        logger.info("[auto-fill] f_max = %g", params.f_max)
        changed = True

    # set frequency step such that approximately 50 points are sampled between f_min and f_max if not provided
    if params.d_f is None or force:
        d_f_raw = (params.f_max - params.f_min) / 50
        params.d_f = max(1000, math.ceil(d_f_raw / 1000) * 1000)

        # adjust f_max so it is an exact multiple of d_f above f_min
        n_steps = math.ceil((params.f_max - params.f_min) / params.d_f)
        params.f_max = params.f_min + n_steps * params.d_f

        logger.info("[auto-fill] d_f = %g", params.d_f)
        logger.info("[auto-fill] f_max adjusted to %g", params.f_max)
        changed = True

    # wavenumber parameters: beta_s, beta_e, d_beta
    if params.beta_s is None or force:
        params.beta_s = 0.0
        logger.info("[auto-fill] beta_s = %g", params.beta_s)
        changed = True

    if params.beta_e is None or force:
        params.beta_e = 100.0
        logger.info("[auto-fill] beta_e = %g", params.beta_e)
        changed = True

    if params.d_beta is None or force:
        params.d_beta = 10.0
        logger.info("[auto-fill] d_beta = %g", params.d_beta)
        changed = True

    # if anything was changed with the autofill options write updated config file to to disk so the user can review and tweak
    if changed and cfg_path is not None:
        cfg_path = Path(cfg_path)
        write_config(
            path=cfg_path,
            overwrite=True,
            cfg_data=cfg.to_toml_dict(),
        )
        logger.info("[auto-fill] updated config written to %s", cfg_path)

    return changed


# --------------------------------------------------
# main routine to prepare a parsing case (input file for lst code)
# --------------------------------------------------
def parsing_setup(
    *,
    cfg: Mapping[str, Any] | None = None,
    out_dir: str | Path = ".",
    out_name: str = "lst_input_parsing.dat",
    auto_fill: bool = False,
    force: bool = False,
    cfg_path: str | Path | None = None,
) -> Path:

    """Create a minimal parsing case in *out_dir*.

    Parameters
    ----------
    cfg : Mapping[str, Any], optional
        Already-loaded configuration dictionary.
    out_dir : str | Path, optional
        Directory to write the input deck into. Created if missing.
    out_name : str, optional
        File name for the input deck (default: "lst_input_parsing.dat").
    auto_fill : bool, optional
        When True, derive ``x_s``, ``x_e``, and ``i_step`` from the
        meanflow binary for any fields that are still ``None``.
    cfg_path : str | Path, optional
        Path to the config file on disk.  Used by ``--auto-fill`` to
        write the updated config back so the user can inspect it.

    Returns
    -------
    Path
        Path to the written input deck.
    """

    logger.info("preparing lst code input file for parsing step")

    cfg = resolve_config(cfg)

    # validate parsing solver mode
    if not cfg.lst.solver.is_simplified:
        logger.warning(
            "parsing setup detected lst.solver.is_simplified = false; "
            "parsing runs are expected to use true"
        )

    # auto-fill unset fields for parsing step
    if auto_fill:
        auto_fill_parsing(cfg, force=force, cfg_path=cfg_path)

    # if output directory is None or empty string set to "." (current directory)
    if not out_dir:
        out_dir = "."

    # convert out_dir to Path object
    out_dir = Path(out_dir)

    # ensure out_dir exists and create (including parents) if not
    out_dir.mkdir(parents=True, exist_ok=True)

    # full output path: / is Pathlib join
    out_path = out_dir / out_name

    # delegate formatting of input deck for lst code to convert/lst_input.py module
    written = generate_lst_input_deck(out_path=out_path, cfg=cfg)

    logger.debug("wrote input deck: %s", written)

    # generate HPC run script alongside the input deck
    hpc_cfg = hpc_configure(cfg, set_defaults=True)

    if hpc_cfg.scheduler.lower() != "unknown":
        lst_exe = cfg.lst_exe if hasattr(cfg, "lst_exe") and cfg.lst_exe else "lst.x"
        script_path = script_build(
            hpc_cfg,
            out_dir,
            lst_exe=lst_exe,
            args=[out_name, ">run.log"],
            extra_env=cfg.hpc.extra_env,
        )
        logger.info("wrote run script: %s", script_path)
    else:
        logger.info("unknown scheduler; skipping HPC script generation")

    return written

