"""Local body radius computation for different geometry kinds."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from lst_tools.core import Grid
from lst_tools.geometry.kinds import GeometryKind


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# main function: radius
# --------------------------------------------------
def radius(
    grid: Grid,
    cfg: dict,
    *,
    debug_path: Path | str | None = None,
) -> np.ndarray:
    """Compute the local body radius array for the given geometry kind.

    Args:
        grid: Computational grid with shape (ny, nx).
        cfg: Configuration object with ``geometry.type`` and related params.
        debug_path: Directory for Tecplot ASCII debug output.

    Returns:
        1-D radius array along the wall (j = 0).
    """

    # --------------------------------------------------
    # get x and y coordinates for specified j location to compute surface angle (default j = 0)
    # --------------------------------------------------

    j = 0

    x = np.asarray(grid.x[j, :], dtype=float)
    y = np.asarray(grid.y[j, :], dtype=float)

    # --------------------------------------------------
    # inialize radius array with zeros
    # --------------------------------------------------

    r = np.zeros_like(x, dtype=float)

    # --------------------------------------------------
    # get geometry type from configuration
    # --------------------------------------------------

    geom_type = cfg.geometry.type

    # --------------------------------------------------
    # determine local radius based on GeometryKind
    # --------------------------------------------------

    if geom_type == GeometryKind.FLAT_PLATE:

        # --------------------------------------------------
        # flat plate: no radius -> nothing to be done, radius already set to zero
        # --------------------------------------------------

        pass

    elif geom_type == GeometryKind.CYLINDER:

        # --------------------------------------------------
        # cylinder: constant radius given by r_cyl in configuration file
        # NOTE: r_cyl lives in grid.cfg (raw dict), not in the schema dataclass
        # --------------------------------------------------

        r_cyl = grid.cfg.get("geometry", {}).get("r_cyl")

        r[:] = r_cyl

    elif geom_type == GeometryKind.CONE:

        # --------------------------------------------------
        # cone: increasing radius given by r_nose and theta_deg in configuration file (if body fitted grid was provided)
        # --------------------------------------------------

        r_nose = cfg.geometry.r_nose
        theta_deg = cfg.geometry.theta_deg
        is_body_fitted = cfg.geometry.is_body_fitted

        if is_body_fitted:

            # convert degrees to radians
            theta_rad = np.radians(theta_deg)

            # compute radius based on local x coordinate and angle theta
            r = x * np.sin(theta_rad) + r_nose * np.cos(theta_rad)

        else:

            # if body fitted grid is not provided, use y coordinate as radius (assumes x-y grid)

            r = y

    elif geom_type == GeometryKind.GENERALIZED_AXISYMMETRIC:

        # --------------------------------------------------
        # generalized axisymmetric: radius is computed from local y coordinate at wall
        # --------------------------------------------------

        r = y

    else:

        logger.error("unsupported geometry type '%s'", geom_type)
        logger.error("supported geometry types are:")

        for g in GeometryKind:

            logger.error("  %s: %s", g.value, g.name)

        raise ValueError(f"unsupported geometry type '{geom_type}'")

    # debug output
    if debug_path is not None:

        # normalize to a Path and ensure directory exists
        dbg_dir = Path(debug_path)
        dbg_dir.mkdir(parents=True, exist_ok=True)

        # write tecplot readable ascii file
        from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii

        write_tecplot_ascii(
            dbg_dir / "radius.dat",
            {"x": x, "y": y, "r": r},
            title="radius debug",
            zone="wall",
        )

    return r

