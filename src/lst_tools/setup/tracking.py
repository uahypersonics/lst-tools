"""Set up tracking (N-factor envelope) calculations.

Reads a parsing solution, identifies frequency peaks, and generates
per-station input decks and HPC job scripts for the tracking step.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lst_tools.convert import generate_lst_input_deck
from lst_tools.data_io import read_tecplot_ascii
from lst_tools.hpc import detect, hpc_configure, script_build

from lst_tools.config import write_config

from ._common import (
    read_baseflow_stations,
    resolve_config,
    scaffold_case_dir,
    write_launcher_script,
)

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# auto-fill unset tracking parameters
# --------------------------------------------------
def auto_fill_tracking(cfg: Any, *, force: bool = False, cfg_path: str | Path | None = None) -> bool:
    """Fill ``beta_s``, ``beta_e``, ``d_beta``, and ``i_step`` if unset or forced.

    Tracking overrides ``x_s``, ``x_e``, frequencies, and ``alpha_0``
    per-case from the parsing solution, so only the wavenumber sweep
    and ``i_step`` need to be present in the base config.

    Parameters
    ----------
    cfg : Config
        A validated configuration object (dataclass).
    cfg_path : str or Path, optional
        Path to the config file on disk.  When provided the config is
        written back with ``overwrite=True`` after filling.

    Returns
    -------
    bool
        ``True`` if at least one field was filled, ``False`` otherwise.
    """
    params = cfg.lst.params
    changed = False

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

    if params.i_step is None or force:
        params.i_step = 1
        logger.info("[auto-fill] i_step = %d", params.i_step)
        changed = True

    # persist to disk so the user can review and tweak
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
# 1d hampel filter
# --------------------------------------------------
def _hampel_1d(y: np.ndarray, win: int = 7, n_sig: float = 3.0) -> np.ndarray:

    """
    Robust 1-D outlier suppressor (Hampel filter).
    Replaces points deviating from local median by > n_sig * 1.4826 * MAD with the local median.
    """

    y = np.asarray(y, float)
    n = y.size

    if n == 0 or win < 3:

        return y.copy()

    k = (win - 1) // 2
    pad = np.pad(y, (k, k), mode="edge")
    out = y.copy()

    for i in range(n):
        w = pad[i : i + win]
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        scale = 1.4826 * (mad if mad > 0 else 1e-12)

        if abs(y[i] - med) > n_sig * scale:
            out[i] = med


    return out



# --------------------------------------------------
# set to zero any contiguous run where y>thresh whose length < k
# --------------------------------------------------
def _remove_spurious_peaks(y: np.ndarray, k: int, thresh: float = 0.0) -> np.ndarray:

    y = np.asarray(y, float)

    out = y.copy()

    pos = y > thresh

    # find run edges in boolean mask
    edges = np.flatnonzero(np.diff(np.r_[False, pos, False]))

    for s, e in edges.reshape(-1, 2):
        if (e - s) < k:
            out[s:e] = 0.0
    return out


# --------------------------------------------------
# centered rolling minimum (NumPy-only)
# --------------------------------------------------
def _rolling_min(a: np.ndarray, w: int) -> np.ndarray:

    """Centered rolling min with edge padding; w must be odd."""

    w = int(w) | 1  # force odd
    k = (w - 1) // 2
    pad = np.pad(a, (k, k), mode="edge")
    return sliding_window_view(pad, w).min(axis=1)


# --------------------------------------------------
# combined row cleaner:
# 1) Hampel de-spike
# 2) Zero short positive runs (< min_run)
# 3) Zero points with low prominence relative to rolling min
# --------------------------------------------------
def _clean_alpi_row(
    y: np.ndarray,
    *,
    tau: float = 0.0,
    min_run: int = 5,
    hampel_win: int = 7,
    hampel_sig: float = 3.0,
    min_prom: float = 5.0,
    prom_win: int = 21,
) -> np.ndarray:

    y = np.asarray(y, float)

    # demand persistence along x
    y = _remove_spurious_peaks(y, k=min_run, thresh=tau)

    # robust de-spike (Hampel)
    y = _hampel_1d(y, win=hampel_win, n_sig=hampel_sig)

    # drop small leftovers by prominence relative to rolling minimum
    base = _rolling_min(y, prom_win)
    prom = y - base
    y = np.where(prom >= min_prom, y, 0.0)

    return y



# --------------------------------------------------
# dynamic-programming ridge tracker (pure NumPy)
# --------------------------------------------------
def _track_ridge_dp(
    alpi_2d: np.ndarray, lam: float = 0.6, max_jump: int = 3
) -> np.ndarray:

    """
    Track a smooth ridge across x (columns) by dynamic programming.
    Maximizes accumulated alpha_i while penalizing frequency-index jumps |Δj|.
    Inputs:
      alpi_2d : array with shape (n_freq, n_x)
      lam     : smoothness penalty weight (larger => smoother path)
      max_jump: max allowed index jump per x-step (limits local search)
    Returns:
      j_star  : int array (length n_x) with the ridge index per x-column
    """

    A = np.asarray(alpi_2d, float)
    J, I = A.shape

    if J == 0 or I == 0:
        return np.empty(0, dtype=int)

    cost = np.empty((J, I), dtype=float)
    prev = np.full((J, I), -1, dtype=int)

    # initialize at the first x-column
    cost[:, 0] = A[:, 0]
    prev[:, 0] = -1

    # forward pass
    for i in range(1, I):
        for j in range(J):
            jL = max(0, j - max_jump)
            jR = min(J, j + max_jump + 1)
            js = np.arange(jL, jR)
            penalties = lam * np.abs(js - j)
            cands = cost[js, i - 1] + A[j, i] - penalties
            k = int(np.argmax(cands))
            cost[j, i] = cands[k]
            prev[j, i] = int(js[k])

    # backtrack best end state
    j_end = int(np.argmax(cost[:, -1]))
    j_star = np.empty(I, dtype=int)
    j_star[-1] = j_end
    for i in range(I - 1, 0, -1):
        j_star[i - 1] = prev[j_star[i], i]

    return j_star


# --------------------------------------------------
# build a keep mask around a ridge path (+- half_width bins)
# --------------------------------------------------
def _keep_mask_from_path(
    j_star: np.ndarray, n_freq: int, half_width: int = 3
) -> np.ndarray:
    keep = np.zeros((n_freq, j_star.size), dtype=bool)
    for i, j0 in enumerate(j_star):
        j0 = int(j0)
        jL = max(0, j0 - half_width)
        jR = min(n_freq, j0 + half_width + 1)
        keep[jL:jR, i] = True
    return keep


# --------------------------------------------------
# smooth a 2d contour field
# --------------------------------------------------
def smooth_contour_field(
    field_2d: np.ndarray, npasses: int = 1
) -> tuple[np.ndarray, np.ndarray]:

    if npasses == 0:
        return field_2d.copy(), np.ones(field_2d.shape, dtype=bool)

    for ipass in range(npasses):

        # --------------------------------------------------
        # track a coherent ridge across x and protect a narrow band around it
        # --------------------------------------------------

        # tuneable parameters for ridge protection
        RIDGE_DP_LAM = 0.6  # smoothness penalty (larger => smoother path)
        RIDGE_DP_MAX_JUMP = 3  # max frequency-index change per x-step
        RIDGE_HALF_WIDTH_BINS = 3  # protect ± this many freq bins around the path

        # track ridge on the original field (avoid biasing the path)
        j_star = _track_ridge_dp(field_2d, lam=RIDGE_DP_LAM, max_jump=RIDGE_DP_MAX_JUMP)

        keep_mask = _keep_mask_from_path(
            j_star, n_freq=field_2d.shape[0], half_width=RIDGE_HALF_WIDTH_BINS
        )

        # --------------------------------------------------
        # smooth alpi_2d
        # --------------------------------------------------

        # set smoothing parameters

        # alpha_i threshold for "unstable" points
        ALPI_POS_THRESH = 1.0

        # minimum number of consecutive points in frequency direction to keep a "peak"
        MIN_RUN_X = 4

        # hampel filter parameters

        # window size (odd integer)
        HAMPEL_WIN = 7
        # number of standard deviations for outlier detection
        HAMPEL_SIGMA = 3.0

        # additional prominence filter parameters
        MIN_PROM = 5.0
        PROM_WIN = 21

        field_2d_smooth = field_2d.copy()

        n_freq, _ = field_2d.shape

        for j in range(n_freq):

            row = field_2d_smooth[j, :]

            # first remove all peaks that do not have sustained support
            # (and lightly de-spike + enforce a minimum prominence)
            row_smoothed = _clean_alpi_row(
                row,
                tau=ALPI_POS_THRESH,
                min_run=MIN_RUN_X,
                hampel_win=HAMPEL_WIN,
                hampel_sig=HAMPEL_SIGMA,
                min_prom=MIN_PROM,
                prom_win=PROM_WIN,
            )

            field_2d_smooth[j, :] = np.where(keep_mask[j, :], row, row_smoothed)


        # --------------------------------------------------
        # update field_2d for next pass
        # --------------------------------------------------

        field_2d = field_2d_smooth


    return field_2d_smooth, keep_mask



# --------------------------------------------------
# read parsing solution (Tecplot ASCII)
# --------------------------------------------------
def _read_parsing_solution(fname: str | None) -> Any:
    """Resolve filename and read the parsing Tecplot solution."""

    # check if a filename was provided; if not, use the default and check it exists
    if fname is None:
        # default file name
        fname = "growth_rate_with_nfact_amps.dat"
        
        logger.info("no solution file provided, using default %s...", fname)

        # check if the default file exists; if not, raise an error
        if not Path(fname).is_file():
            logger.error(
                "default solution file %s does not exist"
                " and no other file was provided"
                " -> abort tracking setup",
                fname,
            )
            raise FileNotFoundError(
                f"parsing solution file not found: {fname}"
            )

    logger.info("read parsing solution")

    return read_tecplot_ascii(fname)


# --------------------------------------------------
# Phase 3: compute beta values for tracking
# --------------------------------------------------
def _resolve_beta_values(cfg: Any, betr_parsing: np.ndarray) -> np.ndarray:
    """Compute the set of beta values to track, filtered against available data."""
    beta_s = cfg.lst.params.beta_s
    beta_e = cfg.lst.params.beta_e
    d_beta = cfg.lst.params.d_beta

    n = int((beta_e - beta_s) / d_beta) + 1
    logger.info("number of betr values for tracking: %s", n)

    betr = np.linspace(beta_s, beta_e, n)
    tol = 1e-8
    mask = np.array(
        [np.any(np.isclose(val, betr_parsing, atol=tol)) for val in betr]
    )
    return betr[mask]


# --------------------------------------------------
# Phase 4: scaffold a case directory (files + executable)
# --------------------------------------------------
def _setup_case_directory(
    betr_loc: float, cfg: Any
) -> tuple[str, str | None]:
    """Create kc_* directory, copy meanflow.bin and LST executable.

    Returns
    -------
    dir_name : str
        Name of the created directory.
    lst_exe : str | None
        Path to the LST executable (from config), or None.
    """
    betr_str = f"{betr_loc:07.2f}".replace(".", "pt")
    dir_name = f"kc_{betr_str}"
    lst_exe = cfg.lst_exe
    scaffold_case_dir(dir_name, "meanflow.bin", lst_exe)
    return dir_name, lst_exe


# --------------------------------------------------
# Phase 5: find the initial guess for eigenvalue tracking
# --------------------------------------------------
def _resolve_freq_bound_start(
    f_s: float | None,
    freq_line: np.ndarray,
    f_min: float,
    f_max: float,
) -> int:
    """Return the start frequency index for the tracking search window."""
    if f_s is None:
        idf_s = 0
    elif f_s < f_min or f_s > f_max:
        logger.warning(
            "f_s = %s outside available range [%s, %s] -> reset to f_min",
            f_s,
            f_min,
            f_max,
        )
        idf_s = 0
    else:
        diff = freq_line - f_s
        valid_indices = diff >= 0
        if np.any(valid_indices):
            idf_s = int(np.argmin(np.where(valid_indices, diff, np.inf)))
        else:
            idf_s = freq_line.size - 1
    logger.info("f_s = %s, idf_s = %s", f_s, idf_s)
    return idf_s


# --------------------------------------------------
# get the founds in frequency space for the tracking search window
# --------------------------------------------------
def _resolve_freq_bound_end(
    f_e: float | None,
    freq_line: np.ndarray,
    f_min: float,
    f_max: float,
) -> int:
    """Return the end frequency index for the tracking search window."""
    if f_e is None:
        idf_e = freq_line.size - 1
    elif f_e < f_min or f_e > f_max:
        logger.warning(
            "f_e = %s outside available range [%s, %s] -> reset to f_max",
            f_e,
            f_min,
            f_max,
        )
        idf_e = freq_line.size - 1
    else:
        diff = freq_line - f_e
        valid_indices = diff <= 0
        if np.any(valid_indices):
            idf_e = int(
                np.argmax(np.where(valid_indices, freq_line, -np.inf))
            )
        else:
            idf_e = 0
    logger.info("f_e = %s, idf_e = %s", f_e, idf_e)
    return idf_e


# --------------------------------------------------
# find initial guess for tracking 
# --------------------------------------------------
def _find_initial_guess(
    tp: Any,
    x_ini: float,
    idx_betr: int,
    cfg: Any,
    debug_path: Path | str | None,
) -> dict[str, Any]:
    """Smooth the alpha_i contour, resolve frequency bounds, and walk
    along the tracking direction to find the most-unstable initial eigenvalue guess.
    """

    # get x locations from parsing solution
    x = tp.field("s")[0, 0, :]
    n_x = x.size
    # get frequencies from parsing solution
    freq_line = tp.field("freq")[0, :, 0]
    # get freestream velocity from config file
    uvel_inf = cfg.flow_conditions.uvel_inf
    # get beta value for this case from parsing solution
    betr_loc = float(tp.field("beta")[idx_betr, 0, 0])

    # extract 2-D fields for given beta index
    alpi_2d = tp.field("alpi")[idx_betr, :, :]
    alpr_2d = tp.field("alpr")[idx_betr, :, :]

    # smooth the contour field
    alpi_2d_smoothed, keep_mask = smooth_contour_field(alpi_2d, npasses=5)

    # optional debug output
    if debug_path is not None:
        dbg_dir = Path(debug_path)
        dbg_dir.mkdir(parents=True, exist_ok=True)
        nj, ni = alpi_2d.shape
        freq_1d = tp.field("freq")[0, :, 0]
        x_2d = np.broadcast_to(x[None, :], (nj, ni)).copy()
        freq_2d = np.broadcast_to(freq_1d[:, None], (nj, ni)).copy()
        km_2d = keep_mask.astype(float)

        # compute non-dimensional phase speed: c_ph,x / U_inf = omega / (alpha_r * U_inf)
        eps = 1e-12
        cphx_2d = (
            (2.0 * np.pi * freq_line[:, None])
            / np.clip(alpr_2d, eps, None)
            / uvel_inf
        )

        from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii

        write_tecplot_ascii(
            dbg_dir / "alpi_debug.dat",
            {
                "x": x_2d,
                "freq": freq_2d,
                "alpi_orig": alpi_2d,
                "alpi_smooth": alpi_2d_smoothed,
                "cphx": cphx_2d,
                "keep_mask": km_2d,
            },
            zone=f"beta={betr_loc:.6f}",
            fmt=".6e",
        )

    # clamp x_ini to available range
    if x_ini > x[-1]:
        logger.warning(
            "x_ini = %s exceeds maximum available x = %s -> reset to x_max",
            x_ini,
            x[-1],
        )
        x_ini = float(x[-1])
    
    if x_ini < x[0]:
        logger.warning(
            "x_ini = %s below minimum available x = %s -> reset to x_min",
            x_ini,
            x[0],
        )
        x_ini = float(x[0])

    # find index of first x station >= x_ini
    idx_ini = int(np.searchsorted(x, x_ini))

    # resolve frequency search range
    freq_min = float(freq_line.min())
    freq_max = float(freq_line.max())
    idf_s = _resolve_freq_bound_start(
        cfg.lst.params.f_min, freq_line, freq_min, freq_max
    )
    idf_e = _resolve_freq_bound_end(
        cfg.lst.params.f_max, freq_line, freq_min, freq_max
    )

    # walk along tracking direction to find an unstable initial guess
    idx_x = idx_ini
    step = 5

    if cfg.lst.params.tracking_dir == 0:
        # downstream: walk forward
        step_dir = step
    else:
        # upstream (default): walk backward
        step_dir = -step
    
    initial_guess: dict[str, Any] = {
        "idx_x": None,
        "idx_f": None,
        "alpi": None,
        "alpr": None,
        "freq": None,
    }

    # keep modyfing idx_x within bounds of available data until we find a positive alpi value (unstable) or exhaust the search range
    while 0 <= idx_x < n_x:
        alpi_line = alpi_2d[idf_s:idf_e, idx_x]
        j_idx_local = int(np.argmax(alpi_line))
        j_idx_global = idf_s + j_idx_local
        if alpi_line[j_idx_local] > 0.0:
            initial_guess["idx_x"] = idx_x
            initial_guess["idx_f"] = j_idx_global
            initial_guess["alpi"] = float(alpi_2d[j_idx_global, idx_x])
            initial_guess["alpr"] = float(alpr_2d[j_idx_global, idx_x])
            initial_guess["freq"] = float(freq_line[j_idx_global])
            break
        else:
            idx_x += step_dir

    # fallback: use the initial station
    if initial_guess["idx_x"] is None:
        idx_x = idx_ini
        alpi_line = alpi_2d[idf_s:idf_e, idx_x]
        j_idx_local = int(np.argmax(alpi_line))
        j_idx_global = idf_s + j_idx_local
        initial_guess["idx_x"] = idx_x
        initial_guess["idx_f"] = j_idx_global
        initial_guess["alpi"] = float(alpi_2d[j_idx_global, idx_x])
        initial_guess["alpr"] = float(alpr_2d[j_idx_global, idx_x])
        initial_guess["freq"] = float(freq_line[j_idx_global])

    logger.info(
        "initial guess at x-index %s (x = %.6f)",
        initial_guess["idx_x"],
        float(x[initial_guess["idx_x"]]),
    )
    logger.info(
        "beta=%.6f, f=%.6f, alpr=%.6e, alpi=%.6e",
        float(betr_loc),
        initial_guess["freq"],
        initial_guess["alpr"],
        initial_guess["alpi"],
    )

    return initial_guess


# --------------------------------------------------
# build tracking config, input deck, and HPC script
# --------------------------------------------------
def _build_and_write_case(
    dir_name: str,
    cfg: Any,
    initial_guess: dict[str, Any],
    betr_loc: float,
    tp: Any,
    x_baseflow: np.ndarray,
    lst_exe: str | None,
) -> Any:
    """Deep-copy config with tracking overrides, write input deck and HPC
    script.  Returns the resolved HPC config.
    """
    x = tp.field("s")[0, 0, :]

    cfg_tracking = copy.deepcopy(cfg)

    # initial conditions
    cfg_tracking.lst.params.f_init = initial_guess["freq"]
    cfg_tracking.lst.params.alpha_0 = complex(initial_guess["alpr"], -initial_guess["alpi"])

    # solver settings
    cfg_tracking.lst.solver.type = 2
    cfg_tracking.lst.solver.is_simplified = False

    # get start and end location for tracking from config file or default to baseflow range if not provided
    if cfg.lst.params.x_s is None:
        x_s = float(x_baseflow.min())
        logger.warning("x_s not set in config => default to x_min (meanflow.bin) = %s", x_s)
    else:
        x_s = cfg.lst.params.x_s

    if cfg.lst.params.x_e is None:
        x_e = float(x_baseflow.max())
        logger.warning("x_e not set in config => default to x_max (meanflow.bin) = %s", x_e)
    else:
        x_e = cfg.lst.params.x_e

    # compute closest index in x_baseflow for x_s and x_e (tracking will be performed between these indices)
    idx_s = int(np.argmin(np.abs(x_baseflow - x_s)))
    idx_e = int(np.argmin(np.abs(x_baseflow - x_e)))

    # if i_step for tracking is not set default to 1 (compute all stations between x_s and x_e)
    if cfg.lst.params.i_step is None:
        i_step = 1
        logger.warning("i_step not set in config => default to %s", i_step)
    else:
        i_step = cfg.lst.params.i_step

    # log user info for tracking range (only printed if --verbose/-v or higher)
    logger.info(
        "tracking x-range: x_s = %s (index %s) -> x_e = %s (index %s)",
        x_s, idx_s, x_e, idx_e,
    )

    logger.info("skip index for tracking i_step = %s", i_step)

    # compute number of stations that will be computed for tracking
    n_stations_tracking = int((abs(idx_s - idx_e) + 1) / i_step)

    # sanity check: ensure at least one station will be computed for tracking
    if n_stations_tracking < 1:
        raise ValueError(
            f"computed 0 tracking stations (idx_s={idx_s}, idx_e={idx_e},"
            f" i_step={i_step}) — check x_s/x_e and i_step in config"
        )

    # user info: ouptut the number of stations to be computed for tracking (only printed if --verbose/-v or higher)
    logger.info(
        "number of stations to be computed for tracking = %s",
        n_stations_tracking,
    )

    # wavenumber range (single beta)
    cfg_tracking.lst.params.beta_s = betr_loc
    cfg_tracking.lst.params.beta_e = betr_loc
    cfg_tracking.lst.params.d_beta = 0
    cfg_tracking.lst.params.beta_init = betr_loc

    # set tracking x-range: one end keeps the user value, the other is
    # overridden to the initial-guess location
    x_ig = float(x[initial_guess["idx_x"]])
    if cfg.lst.params.tracking_dir == 0:
        # downstream: initial guess at x_s, track toward x_e
        cfg_tracking.lst.params.x_s = x_ig
        cfg_tracking.lst.params.x_e = x_e
    else:
        # upstream (default): initial guess at x_e, track toward x_s
        cfg_tracking.lst.params.x_s = x_s
        cfg_tracking.lst.params.x_e = x_ig

    # generate LST input deck (renderer handles the x_s/x_e -> x_min/x_max mapping)
    out_path = Path(dir_name) / "lst_input.dat"

    generate_lst_input_deck(out_path=out_path, cfg=cfg_tracking)

    # generate HPC run script
    env = detect()
    ntasks_per_node = env.cpus_per_node or 1
    nodes_optimal = max(1, n_stations_tracking // ntasks_per_node)

    # user info (only printed if --verbose/-v or higher)
    logger.info("number of nodes requested by user config: %s", cfg.hpc.nodes)

    user_hpc = (
        cfg.to_dict().get("hpc", {})
        if hasattr(cfg, "to_dict")
        else cfg.get("hpc", {})
    )
    user_nodes = user_hpc.get("nodes")
    user_time = user_hpc.get("time")

    if user_nodes is None:
        logger.info(
            "number of nodes not set => compute 'optimal' number of nodes"
        )
        nodes_final = nodes_optimal
        logger.info(
            "number of nodes computed based on lst stations: %s",
            nodes_final,
        )
    else:
        if user_nodes * ntasks_per_node > n_stations_tracking:
            logger.warning(
                "number of requested processors %s > number of locations %s",
                user_nodes * ntasks_per_node,
                n_stations_tracking,
            )
            logger.warning(
                "reset to optimal number of nodes: %s",
                nodes_optimal,
            )
            nodes_final = nodes_optimal
        else:
            nodes_final = user_nodes

    if user_time is None:
        time_final = 0.5
        logger.info(
            "time for scheduler not set => set to default time = %s",
            time_final,
        )
    else:
        time_final = user_time

    hpc_cfg = hpc_configure(
        cfg,
        set_defaults=False,
        nodes_override=nodes_final,
        time_override=time_final,
    )
    logger.debug(
        "hpc scheduler: %s, host: %s",
        hpc_cfg.scheduler,
        hpc_cfg.hostname,
    )

    script_build(
        hpc_cfg,
        Path(dir_name),
        lst_exe=lst_exe,
        args=["lst_input.dat", ">run.log"],
        extra_env=cfg.hpc.extra_env,
    )

    return hpc_cfg


# --------------------------------------------------
# main function: orchestrate the tracking setup pipeline
# --------------------------------------------------
def tracking_setup(
    *,
    cfg: Mapping[str, Any] | None = None,
    fname_parsing: str | None = None,
    debug_path: Path | str | None = None,
    auto_fill: bool = False,
    force: bool = False,
    cfg_path: str | Path | None = None,
) -> Path:
    """Set up eigenvalue tracking cases for all requested beta values.

    Pipeline
    --------
    - Resolve and validate configuration
    - Auto-fill unset sweep parameters (if requested)
    - Read parsing solution and baseflow data
    - Determine beta values to track
    - For each beta: scaffold directory, find initial guess,
      write input deck and HPC script
    - Write a launcher script to submit all jobs
    """

    # info output (only printed if --verbose/-v or higher)
    logger.info("setting up tracking step ...")

    # load and validate config file
    cfg = resolve_config(cfg)

    # theta_deg is a required input in the lst_input.dat file -> set to 0 if not needed
    if cfg.geometry.theta_deg is None:
        cfg.geometry.theta_deg = 0.0

    # auto-fill any missing tracking parameters and write to config file (if requested)
    if auto_fill:
        auto_fill_tracking(cfg, force=force, cfg_path=cfg_path)

    # read parsing solution
    tp = _read_parsing_solution(fname_parsing)

    # get baseflow file name (default meanflow.bin)
    fname_baseflow = cfg.lst.io.baseflow_input

    # initialize baseflow stations to None; if the file exists, read it and extract the station locations in x
    x_baseflow = None
    if fname_baseflow is not None and Path(str(fname_baseflow)).is_file():
        x_baseflow = read_baseflow_stations(fname_baseflow)

    betr_parsing = tp.field("beta")[:, 0, 0]
    betr = _resolve_beta_values(cfg, betr_parsing)

    # determine initial guess location based on tracking direction
    x_parsing = tp.field("s")[0, 0, :]
    if cfg.lst.params.tracking_dir == 0:
        # downstream: start search from x_s (or parsing x_min as fallback)
        x_ini = cfg.lst.params.x_s if cfg.lst.params.x_s is not None else float(x_parsing.min())
    else:
        # upstream (default): start search from x_e (or parsing x_max as fallback)
        x_ini = cfg.lst.params.x_e if cfg.lst.params.x_e is not None else float(x_parsing.max())

    logger.info("begin tracking setup...")

    # initialize a list to keep track of created directories for the launcher script
    created_dirs: list[str] = []

    # initialize hpc_cfg to None; it will be set in the loop and returned at the end (assuming at least one beta value is processed)
    hpc_cfg = None

    # loop over beta values, set up case directories, find initial guesses, and write input decks and HPC scripts
    for betr_loc in betr:

        # get the closest index in the parsing solution to the current beta value; this is the index of the 2D alpi and alpr fields that will be used for the tracking initial guess
        idx_betr = int(np.argmin(np.abs(betr_parsing - betr_loc)))

        # scaffold case directory and copy files
        # - get the path to the LST executable from the config (if set) for use in the HPC script generation
        # - store the created directory name for the launcher script
        dir_name, lst_exe = _setup_case_directory(betr_loc, cfg)

        # store the created directory name for the launcher script
        created_dirs.append(dir_name)

        # find the initial guess for tracking using the following logic:
        # - smoothe the alpha_i contour
        # - resolve frequency bounds
        # - walk upstream to find the most unstable point
        # This returns a dictionary with the initial_guess idx_x, idx_f, alpi, alpr, and freq
        initial_guess = _find_initial_guess(tp, x_ini, idx_betr, cfg, debug_path)

        # build the tracking config for this beta, write the input deck, and generate the HPC script for this case
        hpc_cfg = _build_and_write_case(
            dir_name, cfg, initial_guess, betr_loc, tp, x_baseflow, lst_exe,
        )

    if hpc_cfg is None:
        raise RuntimeError(
            "No beta values to process; cannot build tracking jobs."
        )

    submit_cmd = {"pbs": "qsub", "slurm": "sbatch"}.get(hpc_cfg.scheduler)

    script_path = write_launcher_script(
        created_dirs,
        script_name="run_jobs.sh",
        submit_cmd=submit_cmd,
        fname_run_script=hpc_cfg.fname_run_script,
    )

    logger.info(
        "wrote launcher script %s"
        " (make sure $LST_EXE points to your solver"
        " if not named 'lst_solver').",
        script_path,
    )

    return script_path
