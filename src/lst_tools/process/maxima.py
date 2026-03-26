"""Maxima extraction from 2D LST tracking data.

Finds local maxima (peaks) in the frequency direction at each streamwise
station, then tracks them across stations using the Hungarian algorithm
to produce continuous ridge lines (instability modes).

The ridge tracker (_track_ridges) is a pure index-based algorithm that
only returns peak locations.  The extract_maxima() function uses those
locations to gather full variable data and write output files.
"""


# --------------------------------------------------
# imports necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import argrelextrema

from lst_tools.data_io import read_tecplot_ascii, write_tecplot_ascii

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# constants
# --------------------------------------------------

# default gate tolerance for ridge matching (relative frequency difference)
_DEFAULT_GATE_TOL = 0.10

# minimum number of valid points for a mode to be considered real
_MIN_VALID_POINTS = 40

# peak finding neighborhood size (order parameter for argrelextrema)
_PEAK_ORDER = 1

# fill value for missing data
_MISSING = -99.0


# --------------------------------------------------
# data structures
# --------------------------------------------------
@dataclass
class Ridge:
    """A single tracked ridge defined by index locations.

    Attributes:
        indices: list of (i_station, j_freq) tuples marking where this
                 ridge has a peak in the 2-D (nf, nx) data.
    """

    indices: list[tuple[int, int | float]] = field(default_factory=list)


# --------------------------------------------------
# peak finding
# --------------------------------------------------
def _find_peaks(values: np.ndarray, order: int = _PEAK_ORDER) -> np.ndarray:
    """Find indices of local maxima in a 1-D array.

    Args:
        values: 1-D array of values (e.g. growth rate along frequency axis).
        order: number of neighbors on each side to compare.

    Returns:
        1-D integer array of peak indices (may be empty).
    """

    # find local maxima using scipy
    peak_indices = argrelextrema(values, np.greater, order=order)[0]

    # generate filter mask for positive values at the detected peaks
    positive_value_mask = values[peak_indices] > 0.0

    # filter to positive values only (unstable modes have alpha_i > 0 or nfac > 0)
    peak_indices = peak_indices[positive_value_mask]

    return peak_indices


def _find_peaks_parabolic_interpolation(
    values: np.ndarray, order: int = _PEAK_ORDER
) -> tuple[np.ndarray, np.ndarray]:
    """Find peaks with sub-grid parabolic refinement.

    First finds integer peak indices using argrelextrema (same as
    _find_peaks), then refines each peak position by fitting a parabola
    through the peak and its two neighbors.  The vertex of the parabola
    gives a fractional index closer to the true maximum.

    The computational cost is negligible — just arithmetic on three
    values per peak.

    Args:
        values: 1-D array of values (e.g. growth rate along frequency axis).
        order: number of neighbors on each side to compare.

    Returns:
        Tuple of (integer_indices, fractional_indices).
        integer_indices: 1-D int array of discrete peak locations.
        fractional_indices: 1-D float array of parabolic-refined positions.
    """

    # find discrete peaks (positive values only)
    int_peaks = _find_peaks(values, order=order)

    # no peaks — return empty arrays
    if len(int_peaks) == 0:
        return int_peaks, np.array([], dtype=float)

    nf = len(values)
    frac_peaks = np.empty(len(int_peaks), dtype=float)

    for k, j in enumerate(int_peaks):

        # boundary check: can't fit parabola at edges
        if j == 0 or j == nf - 1:
            frac_peaks[k] = float(j)
            continue

        # values at (j-1, j, j+1)
        a = values[j - 1]
        b = values[j]
        c = values[j + 1]

        # second difference (denominator of vertex formula)
        denom = a - 2.0 * b + c

        # if denominator is zero or positive, parabola is not concave down
        # fall back to integer index
        if denom >= 0.0:
            frac_peaks[k] = float(j)
            continue

        # parabolic vertex offset from j
        offset = 0.5 * (a - c) / denom

        # clamp offset to [-0.5, 0.5] for safety
        offset = max(-0.5, min(0.5, offset))

        frac_peaks[k] = j + offset

    return int_peaks, frac_peaks


# --------------------------------------------------
# ridge tracking (Hungarian algorithm with gating)
# --------------------------------------------------
def _track_ridges(
    target_2d: np.ndarray,
    freq_2d: np.ndarray,
    *,
    gate_tol: float = _DEFAULT_GATE_TOL,
    peak_order: int = _PEAK_ORDER,
    interpolate: bool = False,
) -> list[Ridge]:
    """Track ridges across streamwise stations.

    At each x-station, find peaks of the target variable along the
    frequency axis.  Match peaks to existing ridges using the Hungarian
    algorithm (optimal assignment) with a frequency-distance gate.

    This function only tracks peak locations — it does not access or
    store any variable data beyond the target and frequency arrays.

    Args:
        target_2d: array of shape (nf, nx) — the variable to find peaks in
                   (e.g. alpha_i or nfac).
        freq_2d: array of shape (nf, nx) — the frequency variable.
        gate_tol: maximum relative frequency difference for a valid match.
        peak_order: neighborhood size for peak detection.

    Returns:
        List of Ridge objects (index locations only), sorted by average
        frequency.
    """
    nf, nx = target_2d.shape

    # active ridges being built
    ridges: list[Ridge] = []

    # last known frequency for each ridge (parallel to ridges list)
    last_freqs: list[float] = []

    # walk downstream station by station
    for i in range(nx):

        # extract the frequency and target variable at this station
        freq_slice = freq_2d[:, i]
        target_slice = target_2d[:, i]

        # find peaks in the target variable
        if interpolate:
            peak_idx, frac_idx = _find_peaks_parabolic_interpolation(
                target_slice, order=peak_order
            )
        else:
            peak_idx = _find_peaks(target_slice, order=peak_order)
            frac_idx = None

        # skip station if no peaks found
        if len(peak_idx) == 0:
            continue

        # frequencies at the detected peaks
        peak_freqs = freq_slice[peak_idx]

        # --------------------------------------------------
        # match peaks to existing ridges
        # --------------------------------------------------
        if len(ridges) == 0:
            # first station with peaks — each peak starts a new ridge
            for k, j_peak in enumerate(peak_idx):
                j_store = float(frac_idx[k]) if frac_idx is not None else int(j_peak)
                ridges.append(Ridge(indices=[(i, j_store)]))
                last_freqs.append(float(freq_slice[j_peak]))
            continue

        # build cost matrix: |last_freq - peak_freq|
        prev_arr = np.array(last_freqs)
        cost = np.abs(prev_arr[:, None] - peak_freqs[None, :])

        # solve optimal assignment
        row_idx, col_idx = linear_sum_assignment(cost)

        # track which peaks were matched
        matched_peaks: set[int] = set()

        for r_i, p_i in zip(row_idx, col_idx):

            # gate check: reject if frequency distance is too large
            freq_diff = cost[r_i, p_i]
            ref_freq = prev_arr[r_i]

            if ref_freq > 0 and (freq_diff / ref_freq) > gate_tol:
                continue

            # assign peak to existing ridge
            j_peak = int(peak_idx[p_i])
            j_store = float(frac_idx[p_i]) if frac_idx is not None else j_peak
            ridges[r_i].indices.append((i, j_store))
            last_freqs[r_i] = float(freq_slice[j_peak])
            matched_peaks.add(p_i)

        # unmatched peaks start new ridges
        for p_i in range(len(peak_freqs)):
            if p_i not in matched_peaks:
                j_peak = int(peak_idx[p_i])
                j_store = float(frac_idx[p_i]) if frac_idx is not None else j_peak
                ridges.append(Ridge(indices=[(i, j_store)]))
                last_freqs.append(float(freq_slice[j_peak]))

    # sort ridges by average frequency (low to high)
    def _avg_freq(r: Ridge) -> float:
        if not r.indices:
            return np.inf
        freqs = [float(freq_2d[int(round(j)), i]) for i, j in r.indices]
        return float(np.mean(freqs))

    ridges.sort(key=_avg_freq)

    return ridges


# --------------------------------------------------
# public API: extract maxima from one case directory
# --------------------------------------------------
def extract_maxima(
    dir_name: str | Path,
    fname: str = "growth_rate_with_nfact_amps.dat",
    *,
    gate_tol: float = _DEFAULT_GATE_TOL,
    min_valid: int = _MIN_VALID_POINTS,
    interpolate: bool = False,
) -> list[Path]:
    """Extract ridge-line maxima from a single tracking case directory.

    Reads the solution file (Tecplot ASCII), runs ridge tracking for
    both growth rate (alpha_i) and N-factor, and writes one file per
    valid mode.

    Args:
        dir_name: path to the kc_* directory.
        fname: name of the solution file to read.
        gate_tol: relative frequency gate for ridge matching.
        min_valid: minimum number of valid stations for a mode to be kept.

    Returns:
        List of paths to the written output files.
    """

    # initialize list to collect paths of written files
    written_files: list[Path] = []

    # ensure dir_name is a Path object
    dir_name = Path(dir_name)

    # assemble full path to the solution file
    fpath = dir_name / fname

    # check if the solution file exists before trying to read
    if not fpath.exists():
        logger.warning("solution file not found: %s", fpath)
        return written_files

    # read tecplot data
    logger.info("reading %s", fpath)
    tp = read_tecplot_ascii(fpath)

    # data shape is (K, J, I, nvars) —> K=1 for tracking data
    # squeeze K dimension to get (nf, nx, nvars)
    data_2d = tp.data[0, :, :, :]

    # extract freq, alpi, nfac arrays via public .field() API
    # shape is (K=1, nf, nx) — squeeze to (nf, nx)
    freq_2d = tp.field("freq")[0, :, :]
    alpi_2d = tp.field("alpi")[0, :, :]
    nfac_2d = tp.field("nfac")[0, :, :]

    # debug output for devs
    logger.debug("data shape (nf, nx, nvars): %s", data_2d.shape)

    # --------------------------------------------------
    # extract ridges for growth rate (alpha_i)
    # --------------------------------------------------
    logger.info("extracting growth rate ridges...")

    alpi_ridges = _track_ridges(alpi_2d, freq_2d, gate_tol=gate_tol, interpolate=interpolate)

    logger.info("found %d growth rate ridge(s) in %s", len(alpi_ridges), dir_name.name)

    alpi_files = _write_ridge_files(
        ridges=alpi_ridges,
        data_2d=data_2d,
        prefix="alpi_max_mode",
        variables=tp.variables,
        dir_name=dir_name,
        min_valid=min_valid,
    )
    written_files.extend(alpi_files)

    # --------------------------------------------------
    # extract ridges for N-factor
    # --------------------------------------------------
    logger.info("extracting N-factor ridges...")

    nfac_ridges = _track_ridges(nfac_2d, freq_2d, gate_tol=gate_tol, interpolate=interpolate)

    logger.info("found %d N-factor ridge(s) in %s", len(nfac_ridges), dir_name.name)

    nfac_files = _write_ridge_files(
        ridges=nfac_ridges,
        data_2d=data_2d,
        prefix="nfac_max_mode",
        variables=tp.variables,
        dir_name=dir_name,
        min_valid=min_valid,
    )

    written_files.extend(nfac_files)

    logger.info(
        "wrote %d alpi mode(s) and %d nfac mode(s) for %s",
        len(alpi_files), len(nfac_files), dir_name.name,
    )

    return written_files


# --------------------------------------------------
# write ridge data to Tecplot ASCII files
# --------------------------------------------------
def _write_ridge_files(
    ridges: list[Ridge],
    data_2d: np.ndarray,
    prefix: str,
    variables: list[str],
    dir_name: Path,
    min_valid: int,
) -> list[Path]:
    """Write one Tecplot file per valid ridge.

    Uses the index locations from each Ridge to gather the full variable
    data from data_2d.

    Args:
        ridges: list of Ridge objects (index locations only).
        data_2d: array of shape (nf, nx, nvars) — all variables.
        prefix: filename prefix (e.g. "alpi_max_mode").
        variables: list of variable names from the source file.
        dir_name: directory to write output files.
        min_valid: minimum number of valid points to keep a mode.

    Returns:
        List of paths to written files.
    """
    written: list[Path] = []
    mode_num = 1

    for ridge in ridges:

        # skip modes with too few valid data points
        if len(ridge.indices) < min_valid:
            logger.debug(
                "skipping mode with %d valid points (< %d)",
                len(ridge.indices), min_valid,
            )
            continue

        # gather data at each ridge location from the full dataset
        rows = []
        for i, j in ridge.indices:
            if isinstance(j, float) and j % 1 != 0:
                # interpolate between adjacent frequency grid points
                j_lo = int(np.floor(j))
                j_hi = min(j_lo + 1, data_2d.shape[0] - 1)
                t = j - j_lo
                row = data_2d[j_lo, i, :] * (1.0 - t) + data_2d[j_hi, i, :] * t
            else:
                # direct indexing at integer grid point
                row = data_2d[int(j), i, :]
            rows.append(row)
        rows = np.array(rows)

        # build variable dict for writer
        mode_str = f"{mode_num:03d}"
        var_dict = {
            name: rows[:, col]
            for col, name in enumerate(variables)
        }

        # write file
        fname = f"{prefix}_{mode_str}.dat"
        fpath = dir_name / fname

        write_tecplot_ascii(
            fpath,
            var_dict,
            title=f"{prefix}_{mode_str}",
            zone=f"{prefix}_{mode_str}",
        )

        logger.info("wrote %s (%d points)", fname, len(rows))
        written.append(fpath)
        mode_num += 1

    return written
