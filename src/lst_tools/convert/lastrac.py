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
# helper functions for LASTRAC normalization
# --------------------------------------------------
def _infer_unit_reynolds(
    flow_attrs: dict[str, float],
    cfg: Config,
    dens_inf: float,
    uvel_inf: float,
) -> float:
    """Return the freestream unit Reynolds number in 1/m.

    Args:
        flow_attrs: Root-level HDF5 attributes.
        cfg: Parsed lst-tools configuration.
        dens_inf: Freestream density.
        uvel_inf: Freestream velocity.

    Returns:
        Unit Reynolds number.

    Raises:
        ValueError: If the unit Reynolds number cannot be inferred.
    """

    # check config first
    config_re1 = cfg.flow_conditions.re1
    if config_re1 is not None:
        return float(config_re1)

    # check whether the extractor stored a freestream viscosity attribute
    mu_inf_attr = flow_attrs.get("freestream viscosity")
    if mu_inf_attr is None:
        raise ValueError(
            "flow_conditions.re1 is missing and HDF5 attribute 'freestream viscosity' "
            "is not available to infer the LASTRAC unit Reynolds number"
        )

    mu_inf = float(mu_inf_attr)
    if mu_inf <= 0.0:
        raise ValueError(
            f"freestream viscosity must be positive to infer re1, got {mu_inf:.6e}"
        )

    inferred_re1 = dens_inf * uvel_inf / mu_inf

    logger.info("re1 missing in config -> using HDF5 freestream viscosity to infer re1 %.6e", inferred_re1)

    return float(inferred_re1)


def _compute_station_reference_length(
    s_value: float,
    unit_reynolds: float,
    fallback_lref: float | None,
) -> float:
    """Compute the local LASTRAC reference length for one station.

    Args:
        s_value: Dimensional streamwise station coordinate.
        unit_reynolds: Freestream unit Reynolds number.
        fallback_lref: Optional fallback reference length from config.

    Returns:
        Local reference length used to normalize eta.

    Raises:
        ValueError: If no positive reference length can be produced.
    """

    # compute the local viscous length from the streamwise station coordinate
    if s_value > 0.0:
        local_lref = np.sqrt(s_value / unit_reynolds)
        return float(local_lref)

    # fall back to the user-provided reference length only for degenerate s=0 stations
    if fallback_lref is not None and fallback_lref > 0.0:
        logger.warning(
            "station s <= 0 encountered (s=%.6e) -> falling back to geometry.l_ref %.6e",
            s_value,
            fallback_lref,
        )
        return float(fallback_lref)

    raise ValueError(
        "cannot compute LASTRAC local reference length for station with non-positive "
        f"s={s_value:.6e}; provide geometry.l_ref or skip this station"
    )


def _extract_station_edge_scales(
    flow: Flow,
    urot: np.ndarray,
    i_loc: int,
    *,
    rgas: float,
) -> tuple[float, float, float]:
    """Return station-local edge scales for LASTRAC normalization.

    Args:
        flow: Flow-field container.
        urot: Tangential velocity array after optional rotation.
        i_loc: Streamwise station index.
        rgas: Gas constant used for density fallback.

    Returns:
        Tuple of ``(t_edge, u_edge, rho_edge)``.

    Raises:
        ValueError: If any required local scale is not positive.
    """

    # read full station profiles so the edge scale can fall back to the last valid sample
    temp_profile = np.asarray(flow.field("temp")[:, i_loc], dtype=float)
    uvel_profile = np.abs(np.asarray(urot[:, i_loc], dtype=float))

    # read local edge density directly when available
    try:
        dens_profile = np.asarray(flow.field("dens")[:, i_loc], dtype=float)
    except KeyError:
        pres_profile = np.asarray(flow.field("pres")[:, i_loc], dtype=float)
        dens_profile = pres_profile / (rgas * temp_profile)

    # select the last usable outer-edge sample instead of assuming the final entry is valid
    valid_mask = (temp_profile > 0.0) & (uvel_profile > 0.0) & (dens_profile > 0.0)
    valid_indices = np.flatnonzero(valid_mask)

    if valid_indices.size == 0:
        raise ValueError(
            "station profile does not contain any valid positive edge state for LASTRAC normalization"
        )

    edge_index = int(valid_indices[-1])
    temp_edge = float(temp_profile[edge_index])
    uvel_edge = float(uvel_profile[edge_index])
    dens_edge = float(dens_profile[edge_index])

    # validate the station-local scales before normalization
    if temp_edge <= 0.0:
        raise ValueError(f"station edge temperature must be positive, got {temp_edge:.6e}")
    if uvel_edge <= 0.0:
        raise ValueError(f"station edge velocity must be positive, got {uvel_edge:.6e}")
    if dens_edge <= 0.0:
        raise ValueError(f"station edge density must be positive, got {dens_edge:.6e}")

    return temp_edge, uvel_edge, dens_edge


# --------------------------------------------------
# convert grid and flow (hdf5) to lastrac format
# --------------------------------------------------
def convert_meanflow(
    grid: Grid,
    flow: Flow,
    out: str | Path,
    *,
    cfg: Config,
    debug_path: Path | str | None = None,
) -> Path:
    """Convert grid and flow fields to LASTRAC meanflow format.

    Steps performed:
      1. Compute curvilinear coordinate *s*
      2. Compute surface angle and (optionally) rotate velocity profiles
      3. Compute local body radius and curvature
      4. Write meanflow file (Fortran binary)

    Args:
        grid (Grid): Computational grid (lst_tools.core).
        flow (Flow): Flow field data defined on *grid* (lst_tools.core).
        out (str | Path): Output file path for the LASTRAC meanflow file.
        cfg (dict): Configuration object (typically ``Config`` from
            config/schema.py).
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
    # infer missing freestream fields from HDF5 attrs or edge-state values
    # --------------------------------------------------
    flow_attrs = dict(getattr(flow, "attrs", {}) or {})

    temp_inf = cfg.flow_conditions.temp_inf
    if temp_inf is None:
        temp_attr = flow_attrs.get("static temperature")
        if temp_attr is not None:
            temp_inf = float(temp_attr)
            logger.info("temp_inf missing in config -> using HDF5 static temperature %.6e", temp_inf)
        else:
            temp_inf = float(np.mean(flow.field("temp")[-1, :]))
            logger.info("temp_inf missing in config -> using edge mean temp %.6e", temp_inf)

    uvel_inf = cfg.flow_conditions.uvel_inf
    if uvel_inf is None:
        uvel_inf = float(np.mean(flow.field("uvel")[-1, :]))
        logger.info("uvel_inf missing in config -> using edge mean uvel %.6e", uvel_inf)

    dens_inf = cfg.flow_conditions.dens_inf
    if dens_inf is None:
        dens_attr = flow_attrs.get("static density")
        if dens_attr is not None:
            dens_inf = float(dens_attr)
            logger.info("dens_inf missing in config -> using HDF5 static density %.6e", dens_inf)
        else:
            dens_inf = float(np.mean(flow.field("dens")[-1, :]))
            logger.info("dens_inf missing in config -> using edge mean dens %.6e", dens_inf)

    # --------------------------------------------------
    # validate required flow conditions before entering the write loop
    # --------------------------------------------------
    unit_reynolds = _infer_unit_reynolds(flow_attrs, cfg, dens_inf, uvel_inf)

    required_flow = {
        "temp_inf": temp_inf,
        "uvel_inf": uvel_inf,
        "dens_inf": dens_inf,
        "re1": unit_reynolds,
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

            # build the physical wall-normal coordinate before normalization
            x_loc = grid.x[:, i_loc]
            y_loc = grid.y[:, i_loc]
            eta_dim = np.hypot(x_loc - x_loc[0], y_loc - y_loc[0])

            # build station-local LASTRAC scales
            s_value = float(s[i_loc])
            local_lref = _compute_station_reference_length(
                s_value,
                unit_reynolds,
                cfg.geometry.l_ref,
            )
            local_re1 = unit_reynolds * local_lref

            temp_edge, uvel_edge, dens_edge = _extract_station_edge_scales(
                flow,
                urot,
                i_loc,
                rgas=cfg.flow_conditions.rgas,
            )

            # normalize the station vectors to LASTRAC convention
            eta_norm = eta_dim / local_lref
            uvel_norm = urot[:, i_loc] / uvel_edge
            vvel_norm = vrot[:, i_loc] / uvel_edge

            # write station header: note i_loc+1 because lastrac starts counting at 1 (fortran indexing vs python indexing)
            w.write_station_header(
                i_loc=i_loc + 1,
                n_eta=grid.y.shape[0],
                s=s_value,
                lref=local_lref,
                re1=local_re1,
                kappa=kappa[i_loc],
                rloc=r[i_loc],
                drdx=drdx[i_loc],
                stat_temp=temp_edge,
                stat_uvel=uvel_edge,
                stat_dens=dens_edge,
            )

            # write normalized wall-normal coordinate
            w.write_station_vector(eta_norm)

            # write streamwise velocity (rotated if surface angle is meaningful, otherwise original uvel)
            w.write_station_vector(uvel_norm)

            # write wall-normal velocity (rotated if surface angle is meaningful, otherwise original vvel)
            w.write_station_vector(vvel_norm)

            # write spanwise velocity (optional) — write zeros if field missing

            try:
                has_w = hasattr(flow, "fields") and ("wvel" in flow.fields)
            except Exception:
                has_w = False
            if has_w:
                wvel_i = flow.field("wvel")[:, i_loc]
            else:
                wvel_i = np.zeros(n_eta, dtype=float)

            w.write_station_vector(wvel_i / uvel_edge)

            # write temperature
            w.write_station_vector(flow.field("temp")[:, i_loc] / temp_edge)

            # write pressure
            pres_scale = dens_edge * uvel_edge**2.0
            w.write_station_vector(flow.field("pres")[:, i_loc] / pres_scale)

            # advance progress bar
            advance()

    # close file
    w.close()

    return Path(out)
