"""Geometry presets for LST configuration files."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from enum import Enum


# --------------------------------------------------
# geometry presets for lst-tools init --geometry
# --------------------------------------------------
class GeometryPreset(str, Enum):
    """Supported geometry templates for ``lst-tools init --geometry``."""

    cone = "cone"
    ogive = "ogive"
    flat_plate = "flat-plate"
    cylinder = "cylinder"


# --------------------------------------------------
# geometry templates for each preset (append as needed)
# --------------------------------------------------
GEOMETRY_TEMPLATES: dict[GeometryPreset, dict] = {
    GeometryPreset.cone: {
        "geometry": {
            "type": 2,
            "theta_deg": 7.0,
            "r_nose": 5e-5,
            "is_body_fitted": True,
        },
        "lst": {
            "solver": {"generalized": 0},
            "options": {"geometry_switch": 1, "longitudinal_curvature": 0},
        },
    },
    GeometryPreset.ogive: {
        "geometry": {
            "type": 3,
            "is_body_fitted": False,
        },
        "lst": {
            "solver": {"generalized": 1},
            "options": {"geometry_switch": 1, "longitudinal_curvature": 1},
        },
    },
    GeometryPreset.flat_plate: {
        "geometry": {
            "type": 0,
            "is_body_fitted": False,
        },
        "lst": {
            "solver": {"generalized": 1},
            "options": {"geometry_switch": 0, "longitudinal_curvature": 0},
        },
    },
    GeometryPreset.cylinder: {
        "geometry": {
            "type": 1,
            "is_body_fitted": False,
        },
        "lst": {
            "solver": {"generalized": 0},
            "options": {"geometry_switch": 0, "longitudinal_curvature": 0},
        },
    },
}
