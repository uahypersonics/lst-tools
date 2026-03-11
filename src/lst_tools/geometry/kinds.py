"""Geometry kind identifiers and metadata.

Maps integer geometry codes to human-readable names and lists the
parameters each geometry kind requires.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations
from enum import IntEnum
import logging
from typing import Mapping


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# ids for geometry kinds
# --------------------------------------------------
class GeometryKind(IntEnum):
    """Integer identifiers for supported geometry types."""
    # flat plate: no radius, no curvature
    FLAT_PLATE = 0
    # cylinder: constant radius, no curvature
    CYLINDER = 1
    # straight cone: increasing radius, no curvature
    CONE = 2
    # generalized axisymmetric geometry: variable radius, variable curvature (e.g. ogive, flared cone, ...)
    GENERALIZED_AXISYMMETRIC = 3


# --------------------------------------------------
# geometry kind explanations
# --------------------------------------------------
GEOMETRY_KIND_EXPLANATIONS: dict[GeometryKind, str] = {
    GeometryKind.FLAT_PLATE: "Flat plate",
    GeometryKind.CYLINDER: "Cylinder with constant radius",
    GeometryKind.CONE: "Straight cone (constant half-angle)",
    GeometryKind.GENERALIZED_AXISYMMETRIC: "Generalized axisymmetric geometry (e.g. ogive, flared cone, ...)",
}


# --------------------------------------------------
# expected parameter names per kind (guidance only)
#
# These lists are advisory: they help validation and cli prompts
#
# Conventions:
#   - length units in meters
#   - angles in degrees unless otherwise stated
# --------------------------------------------------
REQUIRED_PARAMS: dict[GeometryKind, tuple[str, ...]] = {
    # required parameters for flat plate
    GeometryKind.FLAT_PLATE: ("r_nose",),
    # required parameters for cylinder
    GeometryKind.CYLINDER: ("r_nose", "r_cyl"),
    # required parameters for cone
    GeometryKind.CONE: ("r_nose", "theta_deg"),
    # required parameters for generalized axisymmetric geometry
    GeometryKind.GENERALIZED_AXISYMMETRIC: tuple(),
}


# --------------------------------------------------
# helper function to get geometry type from either an integer or string input
# --------------------------------------------------
def coerce_kind(value: int | str | GeometryKind) -> GeometryKind:
    """
    Convert various user inputs into a GeometryKind enum.

    Accepts:
      - an existing GeometryKind
      - an int (e.g., 2)
      - a string (case-insensitive), either the integer value "2" or the enum name "cone"

    Raises:
      ValueError if the value cannot be interpreted.
    """
    if isinstance(value, GeometryKind):
        return value

    # int-like string or raw int
    try:
        iv = int(value)  # works for "2" and 2
        return GeometryKind(iv)
    except Exception:
        pass

    # name-like string
    if isinstance(value, str):
        name = value.strip().upper()
        try:
            return GeometryKind[name]
        except KeyError:
            pass

    raise ValueError(f"unknown geometry kind: {value!r}")



# --------------------------------------------------
# function to get the geometry kind description with safety net to raise error when kind is unknown
# --------------------------------------------------
def describe_geometry_kind(kind: GeometryKind | int | str) -> str:
    """
    return description for a given kind
    """
    k = coerce_kind(kind)

    # .get function -> return explanation or print error
    return GEOMETRY_KIND_EXPLANATIONS.get(
        k, f"unknown geometry kind ({int(k)})"
    )



# --------------------------------------------------
# function to list all geometry kinds and map into a dictionary
# --------------------------------------------------
def list_geometry_kinds() -> Mapping[int, str]:
    """
    Return a mapping {integer_id: 'NAME — explanation'} for display/CLI.
    """

    # generate a dictionary mapping integer ids to their names and explanations
    out: dict[int, str] = {}

    for k in GeometryKind:
        out[int(k)] = f"{k.name} — {GEOMETRY_KIND_EXPLANATIONS[k]}"
    return out



# --------------------------------------------------
# function to get the required geometry parameters
# --------------------------------------------------
def required_geometry_parameters(kind: GeometryKind | int | str) -> tuple[str, ...]:
    """
    Return the tuple of required parameter names for the given kind.
    """
    k = coerce_kind(kind)

    return REQUIRED_PARAMS.get(k, tuple())



