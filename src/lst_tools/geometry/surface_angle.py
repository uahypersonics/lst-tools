"""Surface-angle computation along a wall j-line."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from lst_tools.core.grid import Grid


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# function to compute the surface angle at the wall using a first order method (fwd/backward differences)
# --------------------------------------------------
def _first_order(
    grid: Grid,
    debug_path: Path | str | None = None,
) -> np.ndarray:
    """Compute surface tangent angle along j=0 using first-order differences.

    Forward differences are used for all interior points.  The last point
    repeats the backward difference so the output length matches the input.

    Args:
        grid: Structured grid (shape ny x nx).
        debug_path: Directory for Tecplot ASCII debug output.

    Returns:
        Surface tangent angle phi (radians) at each streamwise station.
    """

    # select j location at which surface angle is computed (default j = 0 -> wall line)
    j = 0

    # get x and y coordinates for specified j location to compute surface angle
    x = np.asarray(grid.x[j, :], dtype=float)
    y = np.asarray(grid.y[j, :], dtype=float)

    # compute derivatives (forward differences, last point repeated as backward difference)
    dx = np.diff(x)
    dy = np.diff(y)
    dx = np.append(dx, x[-1] - x[-2])
    dy = np.append(dy, y[-1] - y[-2])

    # compute angle of tangent
    phi = np.arctan2(dy, dx)

    # debug output (tecplot readable ascii format) if specified
    if debug_path is not None:
        dbg_dir = Path(debug_path)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii

        write_tecplot_ascii(
            dbg_dir / "surface_angle_first_order.dat",
            {"x": x, "y": y, "phi_rad": phi, "phi_deg": np.degrees(phi)},
            title="debug",
            zone="debug",
        )

    # return surface angle as an array
    return phi


# --------------------------------------------------
# function to compute the surface angle at the wall using a second order method (gradient function)
# --------------------------------------------------
def _second_order(
    grid: Grid,
    debug_path: Path | str | None = None,
) -> np.ndarray:
    """Compute surface tangent angle along j=0 using second-order central differences.

    Uses ``np.gradient`` which provides second-order accuracy at interior
    points and first-order at the boundaries.

    Args:
        grid: Structured grid (shape ny x nx).
        debug_path: Directory for Tecplot ASCII debug output.

    Returns:
        Surface tangent angle phi (radians) at each streamwise station.
    """

    # select j location at which surface angle is computed (set to j = 0)
    j = 0

    # get x and y coordinates for specified j location to compute surface angle (default j = 0)
    x = np.asarray(grid.x[j, :], dtype=float)
    y = np.asarray(grid.y[j, :], dtype=float)

    # compute derivatives and tangent angle
    dx = np.gradient(x)
    dy = np.gradient(y)

    # compute robust angle of tangent (handles dx ~ 0)
    phi = np.arctan2(dy, dx)

    # debug output (tecplot readable ascii format) if specified
    if debug_path is not None:
        dbg_dir = Path(debug_path)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii

        write_tecplot_ascii(
            dbg_dir / "surface_angle_second_order.dat",
            {"x": x, "y": y, "phi_rad": phi, "phi_deg": np.degrees(phi)},
            title="debug",
            zone="debug",
        )

    # return surface angle as an array
    return phi


# --------------------------------------------------
# method registry
# --------------------------------------------------
_METHODS: dict[str, Callable[..., np.ndarray]] = {
    "first_order": _first_order,
    "second_order": _second_order,
}


# --------------------------------------------------
# dispatcher (this picks the right routine based on user/default inputs)
# --------------------------------------------------
def surface_angle(
    grid: Grid,
    *,
    method: str = "first_order",
    debug_path: Path | str | None = None,
) -> np.ndarray:
    """Compute surface tangent angle phi along the wall j-line.

    Args:
        grid: Structured grid (shape ny x nx).
        method: Differentiation scheme — ``"first_order"`` (forward/backward
            differences) or ``"second_order"`` (``np.gradient``).
        debug_path: Directory for Tecplot ASCII debug output.

    Returns:
        Surface tangent angle phi (radians) at each streamwise station.

    Raises:
        ValueError: If *method* is not a recognised scheme.
    """

    # try to link method to function name; raise error if not possible
    try:
        fn = _METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown method='{method}'; choose one of {list(_METHODS)}"
        )

    return fn(grid, debug_path=debug_path)
