"""Dimensional detection and normalization for extracted profiles.

Heuristic detection
-------------------
Non-dimensional profiles have edge velocity u/u_e ~ 1 and edge temperature
T/T_e ~ 1.  Dimensional profiles have u_e in hundreds to thousands of m/s and
T_e in tens to hundreds of K.

The detection threshold for velocity is 5.0: if the mean edge velocity across
all stations exceeds this, the data is treated as dimensional.  This comfortably
separates physical velocities (typically > 100 m/s for supersonic flows) from
non-dimensional ratios (always in [0, ~1.5]).

Normalization
-------------
Profiles are normalized by the local edge value at each station (outermost
wall-normal point).  Edge values are stored as separate arrays for traceability.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging

import numpy as np

from lst_tools.extract._types import SampledProfiles

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# velocity threshold separating dimensional (m/s) from non-dimensional (u/u_e)
_DIMENSIONAL_VELOCITY_THRESHOLD = 5.0


# --------------------------------------------------
# detect_dimensional: heuristic check
# --------------------------------------------------
def detect_dimensional(profiles: SampledProfiles) -> bool:
    """Return True if the profiles appear to contain dimensional data.

    Heuristic: the mean edge velocity (outermost wall-normal point) is compared
    against a threshold of 5 m/s.  Non-dimensional profiles always have u/u_e
    close to 1; dimensional profiles have velocities typically > 100 m/s.

    Args:
        profiles: Sampled wall-normal profiles.

    Returns:
        True when the profiles look dimensional; False when they look
        non-dimensional.
    """
    # edge velocity at each station = last wall-normal point
    u_edge = profiles.uvel[:, -1]
    mean_u_edge = float(np.mean(np.abs(u_edge)))

    is_dimensional = mean_u_edge > _DIMENSIONAL_VELOCITY_THRESHOLD

    # debug output for devs
    logger.debug(
        "dimensional detection: mean |u_edge| = %.4g  threshold = %.1f  -> %s",
        mean_u_edge,
        _DIMENSIONAL_VELOCITY_THRESHOLD,
        "dimensional" if is_dimensional else "non-dimensional",
    )

    return is_dimensional


# --------------------------------------------------
# normalize_profiles: divide by edge values per station
# --------------------------------------------------
def normalize_profiles(
    profiles: SampledProfiles,
) -> tuple[SampledProfiles, dict[str, np.ndarray]]:
    """Normalize dimensional profiles by their local edge values.

    Each profile is divided by the value at the outermost wall-normal point
    (the edge condition), giving ratios u/u_e, T/T_e, rho/rho_e, p/(rho_e*u_e^2).

    The edge values are returned separately so they can be stored as HDF5
    attributes for downstream reconstruction.

    Args:
        profiles: Sampled wall-normal profiles (assumed dimensional).

    Returns:
        Tuple of (normalized profiles, edge_values dict).
        edge_values has keys: "uvel_e", "temp_e", "rho_e", "pres_e" each with
        shape (n_stations,).

    Raises:
        ValueError: If any edge value is zero or negative.
    """
    # extract edge (outermost) values per station
    uvel_edge   = profiles.uvel[:, -1].copy()
    temp_edge   = profiles.temp[:, -1].copy()
    rho_edge = profiles.rho[:, -1].copy()
    pres_edge   = profiles.pres[:, -1].copy()

    # validate: edge values must be positive
    for name, arr in [("uvel_e", uvel_edge), ("temp_e", temp_edge), ("rho_e", rho_edge), ("pres_e", pres_edge)]:
        if np.any(arr <= 0.0):
            raise ValueError(
                f"edge value {name} has non-positive entries: "
                f"min = {arr.min():.4g}.  Cannot normalize."
            )

    # broadcast edge values for division: shape (n_stations,) -> (n_stations, 1)
    u_e_2d   = uvel_edge[:, np.newaxis]
    T_e_2d   = temp_edge[:, np.newaxis]
    rho_e_2d = rho_edge[:, np.newaxis]

    # normalize profiles: divide each station profile by its own edge value
    uvel_nd = profiles.uvel / u_e_2d
    vvel_nd = profiles.vvel / u_e_2d
    wvel_nd = profiles.wvel / u_e_2d
    temp_nd = profiles.temp / T_e_2d
    rho_nd  = profiles.rho  / rho_e_2d
    # pressure: normalize by rho_e * u_e^2 (dynamic pressure scale)
    pres_nd = profiles.pres / (rho_e_2d * u_e_2d**2)

    # build normalized profiles container
    normalized = SampledProfiles(
        station_x=profiles.station_x,
        station_y=profiles.station_y,
        station_s=profiles.station_s,
        eta=profiles.eta,
        sample_x=profiles.sample_x,
        sample_y=profiles.sample_y,
        uvel=uvel_nd,
        vvel=vvel_nd,
        wvel=wvel_nd,
        temp=temp_nd,
        pres=pres_nd,
        rho=rho_nd,
    )

    # collect edge values for storage as HDF5 attributes
    edge_values = {
        "uvel_e": uvel_edge,
        "temp_e": temp_edge,
        "rho_e":  rho_edge,
        "pres_e": pres_edge,
    }

    logger.debug(
        "normalized %d stations: u_e=[%.2f,%.2f] T_e=[%.2f,%.2f]",
        len(uvel_edge),
        uvel_edge.min(), uvel_edge.max(),
        temp_edge.min(), temp_edge.max(),
    )

    return normalized, edge_values
