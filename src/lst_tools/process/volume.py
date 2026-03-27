"""Assemble 2-D tracking slices into a 3-D volume.

Each completed kc_* directory holds a 2-D Tecplot file
(I=x-stations, J=frequencies).  This module reads every completed
slice, interpolates all variables onto a common (x, f) grid, and
writes a single 3-D Tecplot ASCII volume file (I=nx, J=nf, K=n_kc).
"""


# --------------------------------------------------
# imports
# --------------------------------------------------
from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import typer

from lst_tools.data_io import read_tecplot_ascii, write_tecplot_ascii
from lst_tools.utils.progress import progress
from ._discover import discover_pattern_dirs

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# constants
# --------------------------------------------------

# number of points in x on the common grid
_NX_COMMON = 500

# frequency spacing for the common grid (Hz)
_FREQ_SPACING = 1000.0

# fill value for out-of-bounds interpolation
_FILL_VALUE = -99.0

# solution file name
_SOLUTION_FNAME = "growth_rate_with_nfact_amps.dat"


# --------------------------------------------------
# public API
# --------------------------------------------------
def assemble_volume(
    parent_dir: str | Path,
    solution_fname: str = _SOLUTION_FNAME,
    output_fname: str = "lst_vol.dat",
    plain_output: bool = False,
    dir_pattern: str = "kc_*",
) -> Path | None:
    """Combine 2-D tracking slices from matching case dirs into a 3-D volume.

    Args:
        parent_dir: directory containing tracking case subdirectories.
        solution_fname: name of the solution file inside each kc_* dir.
        output_fname: name of the 3-D output file.
        plain_output: if True, print plain text directory progress instead of
            showing a rich progress bar.
        dir_pattern: glob pattern used to discover case directories.

    Returns:
        Path to the written volume file, or None if no valid slices found.
    """
    parent_dir = Path(parent_dir)

    # --------------------------------------------------
    # discover and sort matching case directories
    # --------------------------------------------------
    kc_dirs = discover_pattern_dirs(parent_dir, dir_pattern)

    if not kc_dirs:
        logger.warning(
            "no case directories found in %s for pattern %s",
            parent_dir,
            dir_pattern,
        )
        return None

    logger.info("found %d case directories for pattern %s", len(kc_dirs), dir_pattern)

    # --------------------------------------------------
    # first pass: read all slices, determine global x/f range
    # --------------------------------------------------
    slices: list[_SliceData] = []

    # process each kc directory and collect valid slices
    def _process_one_slice(kc_dir: Path) -> None:
        sol_path = kc_dir / solution_fname

        if not sol_path.exists():
            logger.info("skipping %s (no %s)", kc_dir.name, solution_fname)
            return

        # read tecplot data
        tp = read_tecplot_ascii(sol_path)

        # extract arrays — shape (K=1, J=nf, I=nx)
        s_arr = tp.field("s")[0, :, :]      # (nf, nx)
        f_arr = tp.field("freq")[0, :, :]   # (nf, nx)
        data_3d = tp.data[0, :, :, :]       # (nf, nx, nvars)

        # extract beta (kc) value from directory name
        kc_val = _parse_kc_value(kc_dir.name)

        slices.append(_SliceData(
            kc_dir=kc_dir,
            kc_val=kc_val,
            s_arr=s_arr,
            f_arr=f_arr,
            data=data_3d,
            variables=tp.variables,
            var_index=tp.var_index,
        ))

        logger.info(
            "read %s: shape (%d freq, %d x-stations)",
            kc_dir.name, data_3d.shape[0], data_3d.shape[1],
        )

    # choose output style based on user preference
    if plain_output:
        for kc_dir in kc_dirs:
            typer.echo(f"- {kc_dir.name}")
            _process_one_slice(kc_dir)
    else:
        with progress(
            total=len(kc_dirs),
            description="process.volume.read_slices",
            persist=True,
        ) as advance:
            for kc_dir in kc_dirs:
                _process_one_slice(kc_dir)
                advance()

    if not slices:
        logger.warning("no completed slices found")
        return None

    # take variable names from the first slice
    variables = slices[0].variables

    # --------------------------------------------------
    # compute global x and f range
    # --------------------------------------------------
    x_min = min(s.s_arr.min() for s in slices)
    x_max = max(s.s_arr.max() for s in slices)
    f_min = min(s.f_arr.min() for s in slices)
    f_max = max(s.f_arr.max() for s in slices)

    # round ranges outward
    x_min = math.floor(x_min * 1000.0) / 1000.0
    x_max = math.ceil(x_max * 1000.0) / 1000.0
    f_min = math.floor(f_min / _FREQ_SPACING) * _FREQ_SPACING
    f_max = math.ceil(f_max / _FREQ_SPACING) * _FREQ_SPACING

    # build common grid
    nx = _NX_COMMON
    nf = round((f_max - f_min) / _FREQ_SPACING) + 1

    x_new = np.linspace(x_min, x_max, nx)
    f_new = np.linspace(f_min, f_max, nf)

    logger.info(
        "common grid: x=[%.3f, %.3f] nx=%d, f=[%.0f, %.0f] nf=%d, nk=%d",
        x_min, x_max, nx, f_min, f_max, nf, len(slices),
    )

    # --------------------------------------------------
    # resolve column indices for s, f, beta
    # --------------------------------------------------
    s_col = slices[0].var_index.get("s", slices[0].var_index.get("x", -1))
    f_col = slices[0].var_index.get("freq.", slices[0].var_index.get("frequency", -1))
    b_col = slices[0].var_index.get("beta", slices[0].var_index.get("kc", -1))

    # resolve via the alias system on the first slice's TecplotData
    # (we already have the field arrays, just need column indices)
    for col_name in variables:
        low = col_name.lower().strip()
        if low in ("s", "x"):
            s_col = variables.index(col_name)
        elif low in ("freq.", "frequency"):
            f_col = variables.index(col_name)
        elif low in ("beta", "kc"):
            b_col = variables.index(col_name)

    # --------------------------------------------------
    # second pass: interpolate each slice onto common grid
    # --------------------------------------------------
    nvars = len(variables)
    n_kc = len(slices)

    # output array: (nx, nf, nk, nvars) — Tecplot point ordering: i fastest
    vol = np.full((nx, nf, n_kc, nvars), _FILL_VALUE)

    for k, slc in enumerate(slices):

        nf_orig = slc.data.shape[0]

        # reshape slice data to (nx_orig, nf_orig, nvars) for x-first indexing
        # our data is (nf_orig, nx_orig, nvars) from TecplotData
        data_ij = np.transpose(slc.data, (1, 0, 2))  # (nx_orig, nf_orig, nvars)

        # get original x and f vectors
        # x values are the same for each frequency line
        x_old = data_ij[:, 0, s_col]

        # --------------------------------------------------
        # step 1: interpolate in x (for each frequency line)
        # --------------------------------------------------
        arr_int_x = np.zeros((nx, nf_orig, nvars))

        for j in range(nf_orig):
            for ivar in range(nvars):
                if ivar == s_col:
                    arr_int_x[:, j, ivar] = x_new
                    continue

                # frequency is constant across x-stations, no interpolation needed
                if ivar == f_col:
                    arr_int_x[:, j, ivar] = data_ij[0, j, f_col]
                    continue

                val_old = data_ij[:, j, ivar]
                interp_func = interp1d(
                    x_old, val_old,
                    kind="linear",
                    fill_value=_FILL_VALUE,
                    bounds_error=False,
                )
                arr_int_x[:, j, ivar] = interp_func(x_new)

        # --------------------------------------------------
        # step 2: interpolate in f (for each x station)
        # --------------------------------------------------
        # f values are the same for each x station (use original, not x-interpolated)
        f_old = data_ij[0, :, f_col]

        for i in range(nx):
            for ivar in range(nvars):
                if ivar == f_col:
                    vol[i, :, k, ivar] = f_new
                    continue

                if ivar == s_col:
                    vol[i, :, k, ivar] = x_new[i]
                    continue

                if ivar == b_col:
                    vol[i, :, k, ivar] = slc.kc_val
                    continue

                val_old = arr_int_x[i, :, ivar]
                interp_func = interp1d(
                    f_old, val_old,
                    kind="linear",
                    fill_value=_FILL_VALUE,
                    bounds_error=False,
                )
                vol[i, :, k, ivar] = interp_func(f_new)

        logger.info("interpolated slice %d/%d (%s)", k + 1, n_kc, slc.kc_dir.name)

    # --------------------------------------------------
    # write 3-D Tecplot ASCII file
    # --------------------------------------------------
    output_path = parent_dir / output_fname

    # build variable dict for the shared writer
    # vol is (nx, nf, n_kc, nvars) — transpose to (n_kc, nf, nx) per variable
    # so that I=nx, J=nf, K=n_kc in the Tecplot file
    var_dict = {
        name: np.transpose(vol[:, :, :, col], (2, 1, 0))
        for col, name in enumerate(variables)
    }

    # write output with explicit progress so users see activity during large file I/O
    if plain_output:
        typer.echo("writing volume data...")
        write_tecplot_ascii(
            output_path,
            var_dict,
            title="lst_vol",
            zone="lst_vol",
        )
    else:
        total_points = nx * nf * n_kc
        with progress(
            total=total_points,
            description="process.volume.write_data",
            persist=True,
        ) as advance:
            write_tecplot_ascii(
                output_path,
                var_dict,
                title="lst_vol",
                zone="lst_vol",
                progress_callback=advance,
            )

    logger.info("wrote volume: %s (I=%d, J=%d, K=%d)", output_path, nx, nf, n_kc)

    return output_path


# --------------------------------------------------
# internal helpers
# --------------------------------------------------
class _SliceData:
    """Container for data read from a single kc_* slice."""

    __slots__ = ("kc_dir", "kc_val", "s_arr", "f_arr", "data", "variables", "var_index")

    def __init__(
        self,
        kc_dir: Path,
        kc_val: float,
        s_arr: np.ndarray,
        f_arr: np.ndarray,
        data: np.ndarray,
        variables: list[str],
        var_index: dict[str, int],
    ) -> None:
        self.kc_dir = kc_dir
        self.kc_val = kc_val
        self.s_arr = s_arr
        self.f_arr = f_arr
        self.data = data
        self.variables = variables
        self.var_index = var_index


def _parse_kc_value(dirname: str) -> float:
    """Extract the numeric case value from directory names ending in ``NNNptMM``.

    Returns:
        The kc value as a float (e.g. 45.0).
    """
    # parse the final numeric token so names like kc_0045pt00 and
    # slice_0045pt00 both resolve to 45.0
    match = re.search(r"(\d+pt\d+)$", dirname)
    if match is None:
        raise ValueError(f"cannot parse case value from directory name: {dirname}")

    # replace the in-name decimal token and convert
    num_str = match.group(1).replace("pt", ".")
    return float(num_str)
