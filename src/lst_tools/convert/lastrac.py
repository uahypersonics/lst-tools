"""Convert HDF5 base-flow data to LASTRAC meanflow format.

Handles curvilinear coordinate computation, surface angle rotation,
local body radius, curvature, and writes the LASTRAC binary (or
ASCII) meanflow file.
"""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from lst_tools.config import Config
from lst_tools.core import Grid, Flow
from lst_tools.geometry import (
    curvature,
    curvilinear_coordinate,
    surface_angle,
    radius,
    GeometryKind,
)
from lst_tools.data_io import LastracWriter
from lst_tools.geometry.kinds import list_geometry_kinds
from lst_tools.utils import progress


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# convert grid and flow (hdf5) to lastrac format
# --------------------------------------------------
def convert_meanflow(
    grid: Grid,
    flow: Flow,
    out: str | Path,
    *,
    cfg: Config,
    format: str = "binary",
    debug_path: Path | str | None = None,
) -> Path:
    """Convert grid and flow fields to LASTRAC meanflow format.

    Steps performed:
      1. Compute curvilinear coordinate *s*
      2. Compute surface angle and (optionally) rotate velocity profiles
      3. Compute local body radius and curvature
      4. Write meanflow file (Fortran binary or ASCII)

    Args:
        grid (Grid): Computational grid (lst_tools.core).
        flow (Flow): Flow field data defined on *grid* (lst_tools.core).
        out (str | Path): Output file path for the LASTRAC meanflow file.
        cfg (dict): Configuration object (typically ``Config`` from
            config/schema.py).
        format (str): Output format: ``"binary"`` (default) or ``"ascii"``.
        debug_path (Path | str | None): If given, write diagnostic files
            to this directory.

    Returns:
        Path: The resolved output file path.

    Raises:
        ValueError: If the configuration contains invalid station ranges
            or missing geometry type.
    """

    # get geometry type from configuration
    if cfg is None:
        raise ValueError("requires a configuration dictionary")

    geometry_type = cfg.geometry.type

    # sanity check: geometry type has to be specified in the config for meaningful radius and curvature computation
    if geometry_type is None:
        # no geometry tpye specified -> print error and list available geometry types for user to choose from
        logger.error("geometry type not specified in configuration file")

        geo_list = list_geometry_kinds()

        logger.error("%-5s %-25s %-50s", "id", "name", "explanation")

        for geo_id, description in geo_list.items():

            name, explanation = description.split(" — ", 1)

            logger.error("%-5s %-25s %s", geo_id, name, explanation)

        raise ValueError(
            "geometry type must be specified in configuration file (can be provided as id or name)"
        )


    # check if grid is body fitted
    is_body_fitted = cfg.geometry.is_body_fitted

    # debug output for devs
    logger.debug("start lastrac format conversion")

    # compute curvilinear coordinate along wall
    s = curvilinear_coordinate(grid.x, grid.y)

    # compute surface angle
    phi = surface_angle(grid, debug_path=debug_path)

    # check if there is a meaningful surface angle

    # set threshold for detecting meaningful surface angle
    tol = 1e-4

    # set rotation flag to true if surface angle exceed threshold anywhere along the wall
    do_rotate = np.any(np.abs(phi) > tol)

    if do_rotate:
        # output for user
        logger.info("non negligible surface inclination -> rotate u/v profiles")

        # get velocity components from flow field array
        uvel = flow.field("uvel")
        vvel = flow.field("vvel")

        # note: flow arrays are (ny, nx); phi is (nx,)

        # this operation adds a leading dimension so c and s become shape (1,nx) -> enables elementwise multiplication downstream
        cphi = np.cos(phi)[None, :]
        sphi = np.sin(phi)[None, :]

        urot = uvel * cphi + vvel * sphi
        vrot = -uvel * sphi + vvel * cphi

    else:
        # no rotation needed -> print output for user
        logger.info("no rotation needed")

        if not is_body_fitted:
            logger.error("surface angle below threshold => negligible")
            logger.error("grid seems to be body fitted but is_body_fitted is set to False in config file")
            logger.error("update configuration file and run conversion again")

            raise ValueError(
                "grid seems to be body fitted but is_body_fitted is set to False in config file"
            )

        urot = flow.field("uvel")
        vrot = flow.field("vvel")

    # --------------------------------------------------
    # set v to 0
    # --------------------------------------------------
    if cfg.meanflow_conversion.set_v_zero:
        logger.info("set vvel to zero")

        vrot = np.zeros_like(urot, float)

    # --------------------------------------------------
    # compute local geometry radius along x (in geometry.radius)
    # --------------------------------------------------
    logger.info("compute local radius...")

    r = radius(grid, cfg, debug_path=debug_path)

    # --------------------------------------------------
    # compute drdx (change of radius along the axis of the body)
    # --------------------------------------------------
    drdx = np.zeros_like(phi, float)
    drdx = np.sin(phi)

    # --------------------------------------------------
    # compute curvature along wall
    # --------------------------------------------------
    logger.info("compute curvature along wall...")

    if is_body_fitted and geometry_type == GeometryKind.CONE:
        logger.info("body fitted cone geometry -> set curvature to zero")

        kappa = np.zeros_like(r, float)

    else:
        kappa = curvature(grid, method="spline", debug_path=debug_path)

    # --------------------------------------------------
    # set number of stations (Python range semantics)
    # values come from cfg["meanflow_conversion"] if present, otherwise defaults
    # --------------------------------------------------

    # start index (default 0)
    i_s = cfg.meanflow_conversion.i_s

    # end index (default full x-length, exclusive)
    i_e = cfg.meanflow_conversion.i_e

    if i_e is None:
        i_e = grid.x.shape[1]

        logger.info("end index i_e is None -> using full x-length i_e=%d", i_e)

    # stride (default 1)
    d_i = cfg.meanflow_conversion.d_i

    # sanity checks
    if d_i <= 0:
        raise ValueError("stride d_i must be a positive integer")

    nx = int(grid.x.shape[1])

    # runtime checks that validator cannot catch:
    if not (0 <= i_s <= nx):
        raise ValueError(f"i_s out of range for grid: i_s={i_s}, nx={nx}")
    if not (1 <= d_i):
        raise ValueError(f"d_i must be positive: d_i={d_i}")
    if not (i_s < i_e <= nx):
        raise ValueError(f"invalid station span: i_s={i_s}, i_e={i_e}, nx={nx}")

    # number of stations visited by range(i_s, i_e, d_i)
    n_i = (i_e - i_s + d_i - 1) // d_i

    # output for user
    logger.info("meanflow conversion configuration:")

    for key, value in cfg.meanflow_conversion.to_dict().items():
        logger.info("%s = %s", key, value)

    # --------------------------------------------------
    # initialize fortran binary writer class -> opens file
    # --------------------------------------------------
    logger.info("initialize binary output...")

    # initiate output file
    w = LastracWriter(
        out, endianness="<", int_dtype=np.int32, real_dtype=np.float64
    )

    # write file header
    w.write_header(
        title="lst_meanflow",
        n_station=n_i,
        igas=1,
        iunit=1,
        Pr=0.71,
        stat_pres=cfg.flow_conditions.pres_inf or 101325.0,
        nsp=0,
    )

    # --------------------------------------------------
    # number of wall-normal points (used repeatedly below)
    # --------------------------------------------------
    n_eta = grid.y.shape[0]

    # --------------------------------------------------
    # validate required flow conditions before entering the write loop
    # --------------------------------------------------
    required_flow = {
        "re1": cfg.flow_conditions.re1,
        "temp_inf": cfg.flow_conditions.temp_inf,
        "uvel_inf": cfg.flow_conditions.uvel_inf,
        "dens_inf": cfg.flow_conditions.dens_inf,
    }
    missing = [k for k, v in required_flow.items() if v is None]
    if missing:
        raise ValueError(
            f"flow_conditions fields required for meanflow conversion "
            f"are None: {', '.join(missing)}"
        )

    # --------------------------------------------------
    # generate rich progress bar and loop over stations to write to lastrac meanflow file
    # --------------------------------------------------
    with progress(total=n_i, description="convert.lastrac.convert_meanflow", persist=True) as advance:

        for i_glb, i_loc in enumerate(range(i_s, i_e, d_i)):

            # write station header: note i_loc+1 because lastrac starts counting at 1 (fortran indexing vs python indexing)
            w.write_station_header(
                i_loc=i_loc + 1,
                n_eta=grid.y.shape[0],
                s=s[i_loc],
                lref=cfg.geometry.l_ref,
                re1=cfg.flow_conditions.re1,
                kappa=kappa[i_loc],
                rloc=r[i_loc],
                drdx=drdx[i_loc],
                stat_temp=cfg.flow_conditions.temp_inf,
                stat_uvel=cfg.flow_conditions.uvel_inf,
                stat_dens=cfg.flow_conditions.dens_inf,
            )

            # compute wall normal coordinate at i_loc and write to lastrac meanflow file
            x_loc = grid.x[:, i_loc]
            y_loc = grid.y[:, i_loc]

            eta = np.hypot(x_loc - x_loc[0], y_loc - y_loc[0])

            w.write_station_vector(eta)

            # write streamwise velocity (rotated if surface angle is meaningful, otherwise original uvel)
            w.write_station_vector(urot[:, i_loc])

            # write wall-normal velocity (rotated if surface angle is meaningful, otherwise original vvel)
            w.write_station_vector(vrot[:, i_loc])

            # write spanwise velocity (optional) — write zeros if field missing

            try:
                has_w = hasattr(flow, "fields") and ("wvel" in flow.fields)
            except Exception:
                has_w = False
            if has_w:
                wvel_i = flow.field("wvel")[:, i_loc]
            else:
                wvel_i = np.zeros(n_eta, dtype=float)

            w.write_station_vector(wvel_i)

            # write temperature
            w.write_station_vector(flow.field("temp")[:, i_loc])

            # write pressure
            w.write_station_vector(flow.field("pres")[:, i_loc])

            # advance progress bar
            advance()

    # close file
    w.close()

    return Path(out)
