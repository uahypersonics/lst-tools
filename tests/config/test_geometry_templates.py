"""Tests for geometry preset templates used by ``lst-tools init -g``."""

from __future__ import annotations

from lst_tools.config.geometry import GEOMETRY_TEMPLATES, GeometryPreset


class TestBodyFittedDefaults:
    """is_body_fitted defaults must match physical reality of each preset."""

    def test_flat_plate_is_body_fitted_by_default(self):
        # flat-plate grids have the wall at y=0 with j-lines along y -> body-fitted by construction
        template = GEOMETRY_TEMPLATES[GeometryPreset.flat_plate]
        assert template["geometry"]["is_body_fitted"] is True

    def test_cone_is_body_fitted_by_default(self):
        # the canonical cone workflow uses a body-fitted curvilinear grid
        template = GEOMETRY_TEMPLATES[GeometryPreset.cone]
        assert template["geometry"]["is_body_fitted"] is True


class TestGeometryTypes:
    """type codes encode body kind: 0=flat-plate, 1=cylinder, 2=cone, 3=ogive (generalized axisymmetric)."""

    def test_flat_plate_type_code(self):
        assert GEOMETRY_TEMPLATES[GeometryPreset.flat_plate]["geometry"]["type"] == 0

    def test_cylinder_type_code(self):
        assert GEOMETRY_TEMPLATES[GeometryPreset.cylinder]["geometry"]["type"] == 1

    def test_cone_type_code(self):
        assert GEOMETRY_TEMPLATES[GeometryPreset.cone]["geometry"]["type"] == 2

    def test_ogive_type_code(self):
        assert GEOMETRY_TEMPLATES[GeometryPreset.ogive]["geometry"]["type"] == 3
