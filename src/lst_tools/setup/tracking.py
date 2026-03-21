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
from typing import Any, Mapping, NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from lst_tools.convert import generate_lst_input_deck
from lst_tools.data_io import read_tecplot_ascii
from lst_tools.geometry import GeometryKind
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


class _TrackingData(NamedTuple):
    """Bundle of arrays loaded from the parsing solution and baseflow."""

    tp: Any                         # TecplotData object
    x: np.ndarray                   # streamwise coordinates (1-D)
    freq_line: np.ndarray           # frequency array (1-D)
    betr_parsing: np.ndarray        # available beta values (1-D)
    x_baseflow: np.ndarray | None   # baseflow station x-locations (1-D)
    uvel_inf: float | None          # freestream velocity
    x_min: float                    # min streamwise coordinate
    x_max: float                    # max streamwise coordinate
    f_min: float                    # min frequency
    f_max: float                    # max frequency



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
# build a keep mask around a ridge path (± half_width bins)
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
# Phase 1: resolve and validate configuration
# --------------------------------------------------


def _resolve_config(cfg: Mapping[str, Any] | None) -> Any:
    """Load config, run consistency checks, validate geometry."""
    cfg = resolve_config(cfg)

    # tracking-specific: validate geometry parameters
    theta_deg = cfg.geometry.theta_deg
    geometry_type = cfg.geometry.type

    if theta_deg is None:
        logger.warning("theta_deg not provided in config file")
        if geometry_type == GeometryKind.CONE:
            logger.error(
                "theta_deg is required for cone geometries"
                " => update lst configuration file and rerun tracking setup"
            )
            raise KeyError
        else:
            logger.info(
                "theta_deg not required for geometry type %s"
                " -> set theta_deg = 0 and continue",
                geometry_type,
            )
            cfg.geometry.theta_deg = 0.0

    return cfg


# --------------------------------------------------
# Phase 2: read all input data (parsing solution + baseflow)
# --------------------------------------------------


def _read_input_data(
    cfg: Any,
    fname_parsing: str | None,
) -> _TrackingData:
    """Read the parsing Tecplot solution and baseflow binary; return bundled arrays."""
    uvel_inf = cfg.flow_conditions.uvel_inf
    logger.info("uvel_inf = %s", uvel_inf)

    # resolve parsing solution filename
    logger.info("get parsing solution file name")
    if fname_parsing is None:
        fname_parsing = "growth_rate_with_nfact_amps.dat"
        logger.info(
            "no solution file provided, using default %s...",
            fname_parsing,
        )
        if not Path(fname_parsing).is_file():
            logger.error(
                "default solution file %s does not exist"
                " and no other file was provided"
                " -> abort tracking setup",
                fname_parsing,
            )
            raise FileNotFoundError(
                f"parsing solution file not found: {fname_parsing}"
            )

    # read baseflow binary (if available)
    baseflow_input = cfg.lst.io.baseflow_input
    x_baseflow: np.ndarray | None = None

    if baseflow_input is not None:
        baseflow_input = Path(str(baseflow_input))

    if baseflow_input is not None and baseflow_input.is_file():
        x_baseflow = read_baseflow_stations(baseflow_input)

    # read Tecplot parsing solution
    logger.info("read parsing solution")
    tp = read_tecplot_ascii(fname_parsing)

    # extract coordinate arrays
    x_field = tp.field("s")
    x_min = float(x_field.min())
    x_max = float(x_field.max())
    logger.info("x_min = %s, x_max = %s", x_min, x_max)
    x = x_field[0, 0, :]

    freq_line = tp.field("freq")[0, :, 0]
    f_min = float(freq_line.min())
    f_max = float(freq_line.max())
    logger.info("f_min = %s, f_max = %s", f_min, f_max)

    betr_field = tp.field("beta")
    logger.info(
        "betr_min = %s, betr_max = %s", betr_field.min(), betr_field.max()
    )
    betr_parsing = betr_field[:, 0, 0]
    logger.info("available betr values from parsing: ")
    for val in betr_parsing:
        logger.info("  %10.4f", val)

    return _TrackingData(
        tp=tp,
        x=x,
        freq_line=freq_line,
        betr_parsing=betr_parsing,
        x_baseflow=x_baseflow,
        uvel_inf=uvel_inf,
        x_min=x_min,
        x_max=x_max,
        f_min=f_min,
        f_max=f_max,
    )


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


def _find_initial_guess(
    data: _TrackingData,
    idx_betr: int,
    cfg: Any,
    betr_loc: float,
    debug_path: Path | str | None,
) -> dict[str, Any]:
    """Smooth the alpha_i contour, resolve frequency bounds, and walk
    upstream to find the most-unstable initial eigenvalue guess.
    """
    tp = data.tp
    x = data.x
    freq_line = data.freq_line

    # extract 2-D fields for this beta index
    alpi_2d = tp.field("alpi")[idx_betr, :, :]
    alpr_2d = tp.field("alpr")[idx_betr, :, :]

    eps = 1e-12
    cphx_2d = (
        (2.0 * np.pi * freq_line[:, None])
        / np.clip(alpr_2d, eps, None)
        / data.uvel_inf
    )

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

    # resolve initial x-location
    if cfg.lst.params.x_e is None:
        logger.warning("no x_e location provided for tracking")
        logger.warning("set x_e to x_max = %s", data.x_max)
        x_ini = data.x_max
    else:
        x_ini = cfg.lst.params.x_e

    if x_ini > x[-1]:
        logger.warning(
            "x_ini = %s exceeds maximum available x = %s -> reset to x_max",
            x_ini,
            x[-1],
        )
        x_ini = float(x[-1])

    idx_ini = int((x >= x_ini).nonzero()[0][0])

    # resolve frequency search range
    idf_s = _resolve_freq_bound_start(
        cfg.lst.params.f_min, freq_line, data.f_min, data.f_max
    )
    idf_e = _resolve_freq_bound_end(
        cfg.lst.params.f_max, freq_line, data.f_min, data.f_max
    )

    # walk upstream to find an unstable initial guess
    idx_x = idx_ini
    back_step = 5
    best: dict[str, Any] = {
        "idx_x": None,
        "idx_f": None,
        "alpi": None,
        "alpr": None,
        "freq": None,
    }

    while idx_x >= 0:
        alpi_line = alpi_2d[idf_s:idf_e, idx_x]
        j_idx_local = int(np.argmax(alpi_line))
        j_idx_global = idf_s + j_idx_local
        if alpi_line[j_idx_local] > 0.0:
            best["idx_x"] = idx_x
            best["idx_f"] = j_idx_global
            best["alpi"] = float(alpi_2d[j_idx_global, idx_x])
            best["alpr"] = float(alpr_2d[j_idx_global, idx_x])
            best["freq"] = float(freq_line[j_idx_global])
            break
        else:
            idx_x -= back_step

    # fallback: use the initial station
    if best["idx_x"] is None:
        idx_x = idx_ini
        alpi_line = alpi_2d[idf_s:idf_e, idx_x]
        j_idx_local = int(np.argmax(alpi_line))
        j_idx_global = idf_s + j_idx_local
        best["idx_x"] = idx_x
        best["idx_f"] = j_idx_global
        best["alpi"] = float(alpi_2d[j_idx_global, idx_x])
        best["alpr"] = float(alpr_2d[j_idx_global, idx_x])
        best["freq"] = float(freq_line[j_idx_global])

    logger.info(
        "initial guess at x-index %s (x = %.6f)",
        best["idx_x"],
        float(x[best["idx_x"]]),
    )
    logger.info(
        "beta=%.6f, f=%.6f, alpr=%.6e, alpi=%.6e",
        float(betr_loc),
        best["freq"],
        best["alpr"],
        best["alpi"],
    )

    return best


# --------------------------------------------------
# Phase 6: build tracking config, input deck, and HPC script
# --------------------------------------------------


def _build_and_write_case(
    dir_name: str,
    cfg: Any,
    best: dict[str, Any],
    betr_loc: float,
    data: _TrackingData,
    lst_exe: str | None,
) -> Any:
    """Deep-copy config with tracking overrides, write input deck and HPC
    script.  Returns the resolved HPC config.
    """
    x = data.x

    cfg_tracking = copy.deepcopy(cfg)

    # initial conditions
    cfg_tracking.lst.params.f_init = best["freq"]
    cfg_tracking.lst.params.alpha_0 = complex(best["alpr"], -best["alpi"])

    # solver settings
    cfg_tracking.lst.solver.type = 2
    cfg_tracking.lst.solver.is_simplified = False

    # x-range
    cfg_tracking.lst.params.x_s = float(x[best["idx_x"]])
    if cfg.lst.params.x_s is None:
        cfg_tracking.lst.params.x_e = data.x_min
    else:
        cfg_tracking.lst.params.x_e = cfg.lst.params.x_s

    # compute number of tracking stations from baseflow grid
    x_s = cfg_tracking.lst.params.x_s
    x_e = cfg_tracking.lst.params.x_e
    idx_s = int(np.argmin(np.abs(data.x_baseflow - x_s)))
    idx_e = int(np.argmin(np.abs(data.x_baseflow - x_e)))

    if cfg_tracking.lst.params.i_step is None:
        logger.warning(
            "lst.params.i_step not set in config file => default to 1"
        )
        cfg_tracking.lst.params.i_step = 1

    i_step = cfg_tracking.lst.params.i_step
    logger.info(
        "start location for tracking x_s = %s - index = %s", x_s, idx_s
    )
    logger.info(
        "end location for tracking x_e = %s - index = %s", x_e, idx_e
    )
    logger.info("skip index for tracking i_step = %s", i_step)

    n_stations_tracking = int((abs(idx_s - idx_e) + 1) / i_step)
    if n_stations_tracking < 1:
        logger.warning(
            "computed 0 tracking stations (idx_s=%s, idx_e=%s, i_step=%s) "
            "— check x_s/x_e in config; defaulting to 1 node",
            idx_s, idx_e, i_step,
        )
        n_stations_tracking = max(n_stations_tracking, 1)
    logger.info(
        "number of stations to be computed for tracking = %s",
        n_stations_tracking,
    )

    # wavenumber range (single beta)
    cfg_tracking.lst.params.beta_s = betr_loc
    cfg_tracking.lst.params.beta_e = betr_loc
    cfg_tracking.lst.params.d_beta = 0
    cfg_tracking.lst.params.beta_init = betr_loc

    # generate LST input deck
    out_path = Path(dir_name) / "lst_input.dat"
    generate_lst_input_deck(out_path=out_path, cfg=cfg_tracking)

    # generate HPC run script
    env = detect()
    ntasks_per_node = env.cpus_per_node or 1
    nodes_optimal = max(1, n_stations_tracking // ntasks_per_node)

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
    1. Resolve and validate configuration
    2. Auto-fill unset sweep parameters (if requested)
    3. Read parsing solution and baseflow data
    4. Determine beta values to track
    5. For each beta: scaffold directory, find initial guess,
       write input deck and HPC script
    6. Write a launcher script to submit all jobs
    """
    logger.info("setting up tracking step ...")

    cfg = _resolve_config(cfg)

    if auto_fill:
        auto_fill_tracking(cfg, force=force, cfg_path=cfg_path)

    data = _read_input_data(cfg, fname_parsing)
    betr = _resolve_beta_values(cfg, data.betr_parsing)

    logger.info("begin tracking setup...")

    created_dirs: list[str] = []
    hpc_cfg = None

    for betr_loc in betr:
        idx_betr = int(np.argmin(np.abs(data.betr_parsing - betr_loc)))
        dir_name, lst_exe = _setup_case_directory(betr_loc, cfg)
        created_dirs.append(dir_name)
        best = _find_initial_guess(data, idx_betr, cfg, betr_loc, debug_path)
        hpc_cfg = _build_and_write_case(
            dir_name, cfg, best, betr_loc, data, lst_exe,
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
