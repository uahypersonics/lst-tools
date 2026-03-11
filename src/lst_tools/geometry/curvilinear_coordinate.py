"""Curvilinear (streamwise) coordinate computation along a grid line."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# function to compute curvilinear coordinate (simple streamwise s)
# --------------------------------------------------
def curvilinear_coordinate(
    x: Any,
    y: Any,
    j: int = 0,
    debug_path: Path | str | None = None,
) -> np.ndarray:
    """Return a curvilinear coordinate array *s* for a given grid line.

    This minimal version mirrors the original Fortran logic::

        s(i) = x(i, 1)

    i.e., use the x-coordinates along a fixed j-line as the streamwise
    coordinate.

    Args:
        x: 2-D grid x-coordinates with shape (ny, nx), or 1-D array.
        y: 2-D grid y-coordinates with shape (ny, nx), or 1-D array.
        j: Row index (0..ny-1) along which to extract *s*.
        debug_path: Directory for Tecplot ASCII debug output.

    Returns:
        1-D array of the curvilinear coordinate along the selected j-line.

    Note:
        If 1-D arrays are provided, the function simply returns a copy of *x*.
        This is intentionally simple.  If you later want arc-length or more
        sophisticated definitions of *s*, extend this routine while keeping
        the same call signature.
    """

    X = np.asarray(x)
    Y = np.asarray(y)

    if X.shape != Y.shape:
        raise ValueError(
            f"x and y must have the same shape, got {X.shape} vs {Y.shape}"
        )

    if X.ndim == 1:
        # Degenerate case: 1-D, just return x
        s = X.astype(float).copy()
    elif X.ndim == 2:
        ny, nx = X.shape
        if not (0 <= j < ny):
            raise IndexError(
                f"j index {j} out of range for shape (ny={ny}, nx={nx})"
            )
        s = X[j, :].astype(float).copy()
    else:
        raise ValueError(
            f"expected 1-D or 2-D arrays; got ndim={X.ndim}"
        )

    logger.debug("shape(x,y): %s", X.shape)
    if X.ndim == 2:
        logger.debug("using j-line j=%d; returned s.shape=%s", j, s.shape)
    else:
        logger.debug("1-D input; returned s.shape=%s", s.shape)

    # write tecplot-readable debug file if requested
    if debug_path is not None:
        dbg_dir = Path(debug_path)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        if X.ndim == 2:
            zone_name = f"j={j}"
            x_out = X[j, :]
            y_out = Y[j, :]
        else:
            zone_name = "1D"
            x_out = X
            y_out = Y

        from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii

        write_tecplot_ascii(
            dbg_dir / "curvilinear_coordinate_debug.dat",
            {"x": x_out, "y": y_out, "s": s},
            title="curvilinear coordinate debug",
            zone=zone_name,
        )

    return s
