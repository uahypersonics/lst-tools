"""Geometry public API."""

from .curvature import curvature
from .curvilinear_coordinate import curvilinear_coordinate
from .kinds import (
    GEOMETRY_KIND_EXPLANATIONS,
    REQUIRED_PARAMS,
    GeometryKind,
    coerce_kind,
    describe_geometry_kind,
    list_geometry_kinds,
    required_geometry_parameters,
)
from .radius import radius
from .surface_angle import surface_angle

__all__ = [
    # Functions
    "curvature",
    "curvilinear_coordinate",
    "radius",
    "surface_angle",
    # Kinds and utilities
    "GeometryKind",
    "GEOMETRY_KIND_EXPLANATIONS",
    "REQUIRED_PARAMS",
    "coerce_kind",
    "describe_geometry_kind",
    "list_geometry_kinds",
    "required_geometry_parameters",
]
