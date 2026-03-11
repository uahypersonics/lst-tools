# --------------------------------------------------
# test cases for local radius computation
# --------------------------------------------------
#
import numpy as np
import pytest
from pathlib import Path

from lst_tools.core import Grid
from lst_tools.geometry.radius import radius
from lst_tools.geometry.kinds import GeometryKind
from lst_tools.config.schema import Config


class TestRadius:
    """Test cases for the radius computation function."""

    def test_flat_plate(self):
        """Test radius computation for flat plate geometry."""
        # Create mock grid data
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        y = np.array([[0.0, 0.0, 0.0, 0.0]])

        grid = Grid(x=x, y=y)
        cfg = Config.from_dict({"geometry": {"type": GeometryKind.FLAT_PLATE}})

        result = radius(grid, cfg)

        # For flat plate, all radii should be zero
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_cylinder(self):
        """Test radius computation for cylinder geometry."""
        # Create mock grid data
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        y = np.array([[0.0, 0.0, 0.0, 0.0]])
        r_cyl = 5.0

        grid = Grid(x=x, y=y, cfg={"geometry": {"r_cyl": r_cyl}})
        cfg = Config.from_dict({"geometry": {"type": GeometryKind.CYLINDER}})

        result = radius(grid, cfg)

        # For cylinder, all radii should be equal to r_cyl
        expected = np.array([r_cyl, r_cyl, r_cyl, r_cyl])
        np.testing.assert_array_equal(result, expected)

    def test_cone_body_fitted(self):
        """Test radius computation for body-fitted cone geometry."""
        # Create mock grid data
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        y = np.array([[1.0, 1.5, 2.0, 2.5]])  # Not used for body fitted cone
        r_nose = 1.0
        theta_deg = 30.0

        grid = Grid(x=x, y=y)
        cfg = Config.from_dict({
            "geometry": {
                "type": GeometryKind.CONE,
                "r_nose": r_nose,
                "theta_deg": theta_deg,
                "is_body_fitted": True,
            }
        })

        result = radius(grid, cfg)

        # For body fitted cone: r = x * sin(theta) + r_nose * cos(theta)
        theta_rad = np.radians(theta_deg)
        expected = x[0] * np.sin(theta_rad) + r_nose * np.cos(theta_rad)
        np.testing.assert_array_almost_equal(result, expected)

    def test_cone_not_body_fitted(self):
        """Test radius computation for non-body-fitted cone geometry."""
        # Create mock grid data
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        y = np.array([[1.0, 1.5, 2.0, 2.5]])

        grid = Grid(x=x, y=y)
        cfg = Config.from_dict({"geometry": {"type": GeometryKind.CONE, "is_body_fitted": False}})

        result = radius(grid, cfg)

        # For non-body fitted cone, radius should equal y coordinates
        expected = y[0]
        np.testing.assert_array_equal(result, expected)

    def test_generalized_axisymmetric(self):
        """Test radius computation for generalized axisymmetric geometry."""
        # Create mock grid data
        x = np.array([[0.0, 1.0, 2.0, 3.0]])
        y = np.array([[1.0, 2.0, 3.0, 4.0]])

        grid = Grid(x=x, y=y)
        cfg = Config.from_dict({"geometry": {"type": GeometryKind.GENERALIZED_AXISYMMETRIC}})

        result = radius(grid, cfg)

        # For generalized axisymmetric, radius should equal y coordinates
        expected = y[0]
        np.testing.assert_array_equal(result, expected)

    def test_unsupported_geometry(self):
        """Test error handling for unsupported geometry types."""
        x = np.array([[0.0, 1.0, 2.0]])
        y = np.array([[0.0, 0.0, 0.0]])

        grid = Grid(x=x, y=y)
        # Use an invalid geometry type
        cfg = Config.from_dict({"geometry": {"type": -1}})

        # Should raise ValueError for unsupported geometry
        with pytest.raises(ValueError, match="unsupported geometry type"):
            radius(grid, cfg)

    def test_with_debug_path(self):
        """Test radius computation with debug_path parameter."""
        x = np.array([[0.0, 1.0, 2.0]])
        y = np.array([[0.0, 0.0, 0.0]])

        grid = Grid(x=x, y=y)
        cfg = Config.from_dict({"geometry": {"type": GeometryKind.FLAT_PLATE}})
        debug_path = Path("/tmp")

        result = radius(grid, cfg, debug_path=debug_path)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)
