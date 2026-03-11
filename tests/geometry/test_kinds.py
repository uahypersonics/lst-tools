import pytest
from lst_tools.geometry.kinds import (
    GeometryKind,
    coerce_kind,
    describe_geometry_kind,
    list_geometry_kinds,
    required_geometry_parameters,
)


class TestGeometryKinds:
    def test_coerce_kind_with_enum(self):
        """Test that coerce_kind returns the same enum when passed an enum."""
        assert coerce_kind(GeometryKind.FLAT_PLATE) == GeometryKind.FLAT_PLATE
        assert coerce_kind(GeometryKind.CYLINDER) == GeometryKind.CYLINDER
        assert coerce_kind(GeometryKind.CONE) == GeometryKind.CONE
        assert (
            coerce_kind(GeometryKind.GENERALIZED_AXISYMMETRIC)
            == GeometryKind.GENERALIZED_AXISYMMETRIC
        )

    def test_coerce_kind_with_int(self):
        """Test that coerce_kind correctly converts integers to enums."""
        assert coerce_kind(0) == GeometryKind.FLAT_PLATE
        assert coerce_kind(1) == GeometryKind.CYLINDER
        assert coerce_kind(2) == GeometryKind.CONE
        assert coerce_kind(3) == GeometryKind.GENERALIZED_AXISYMMETRIC

    def test_coerce_kind_with_string_int(self):
        """Test that coerce_kind correctly converts string integers to enums."""
        assert coerce_kind("0") == GeometryKind.FLAT_PLATE
        assert coerce_kind("1") == GeometryKind.CYLINDER
        assert coerce_kind("2") == GeometryKind.CONE
        assert coerce_kind("3") == GeometryKind.GENERALIZED_AXISYMMETRIC

    def test_coerce_kind_with_enum_name(self):
        """Test that coerce_kind correctly converts enum names to enums."""
        assert coerce_kind("FLAT_PLATE") == GeometryKind.FLAT_PLATE
        assert coerce_kind("CYLINDER") == GeometryKind.CYLINDER
        assert coerce_kind("CONE") == GeometryKind.CONE
        assert (
            coerce_kind("GENERALIZED_AXISYMMETRIC")
            == GeometryKind.GENERALIZED_AXISYMMETRIC
        )

    def test_coerce_kind_with_mixed_case_string(self):
        """Test that coerce_kind handles mixed case enum names."""
        assert coerce_kind("flat_plate") == GeometryKind.FLAT_PLATE
        assert coerce_kind("Cylinder") == GeometryKind.CYLINDER
        assert coerce_kind("cOnE") == GeometryKind.CONE
        assert (
            coerce_kind("generalized_axisymmetric")
            == GeometryKind.GENERALIZED_AXISYMMETRIC
        )

    def test_coerce_kind_invalid_value(self):
        """Test that coerce_kind raises ValueError for invalid inputs."""
        with pytest.raises(ValueError):
            coerce_kind("invalid")
        with pytest.raises(ValueError):
            coerce_kind(999)
        with pytest.raises(ValueError):
            coerce_kind(-1)

    def test_describe_geometry_kind(self):
        """Test that describe_geometry_kind returns correct descriptions."""
        assert describe_geometry_kind(0) == "Flat plate"
        assert (
            describe_geometry_kind(GeometryKind.CYLINDER)
            == "Cylinder with constant radius"
        )
        assert describe_geometry_kind("cone") == "Straight cone (constant half-angle)"
        assert (
            describe_geometry_kind("3")
            == "Generalized axisymmetric geometry (e.g. ogive, flared cone, ...)"
        )

    def test_describe_geometry_kind_invalid_value(self):
        """Test that describe_geometry_kind raises ValueError for invalid kinds."""
        with pytest.raises(ValueError):
            describe_geometry_kind(999)

    def test_list_geometry_kinds(self):
        """Test that list_geometry_kinds returns expected mapping."""
        expected = {
            0: "FLAT_PLATE — Flat plate",
            1: "CYLINDER — Cylinder with constant radius",
            2: "CONE — Straight cone (constant half-angle)",
            3: "GENERALIZED_AXISYMMETRIC — Generalized axisymmetric geometry (e.g. ogive, flared cone, ...)",
        }
        result = list_geometry_kinds()
        assert result == expected

    def test_required_geometry_parameters(self):
        """Test that required_geometry_parameters returns correct tuples."""
        assert required_geometry_parameters(0) == ("r_nose",)
        assert required_geometry_parameters(GeometryKind.CYLINDER) == (
            "r_nose",
            "r_cyl",
        )
        assert required_geometry_parameters("cone") == ("r_nose", "theta_deg")
        assert required_geometry_parameters("3") == tuple()

    def test_required_geometry_parameters_invalid_value(self):
        """Test that required_geometry_parameters raises ValueError for invalid kinds."""
        with pytest.raises(ValueError):
            required_geometry_parameters(999)
