import numpy as np
import pytest
from lst_tools.core.grid import Grid


class TestGrid:
    def test_grid_creation_with_2d_coordinates(self):
        """Test creating a Grid with 2D x and y coordinates"""
        x = np.array([[0, 1], [0, 1]])
        y = np.array([[0, 0], [1, 1]])

        grid = Grid(x=x, y=y)

        assert np.array_equal(grid.x, x)
        assert np.array_equal(grid.y, y)
        assert grid.z is None
        assert grid.attrs is None
        assert grid.shape == (2, 2)

    def test_grid_creation_with_3d_coordinates(self):
        """Test creating a Grid with 3D coordinates including z"""
        x = np.array([[[0, 1], [0, 1]], [[0, 1], [0, 1]]])
        y = np.array([[[0, 0], [1, 1]], [[0, 0], [1, 1]]])
        z = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]]])

        grid = Grid(x=x, y=y, z=z)

        assert np.array_equal(grid.x, x)
        assert np.array_equal(grid.y, y)
        assert np.array_equal(grid.z, z)
        assert grid.shape == (2, 2, 2)

    def test_grid_creation_with_attributes(self):
        """Test creating a Grid with attributes"""
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        attrs = {"spacing": 0.1, "units": "meters", "version": 1}

        grid = Grid(x=x, y=y, attrs=attrs)

        assert np.array_equal(grid.x, x)
        assert np.array_equal(grid.y, y)
        assert grid.attrs == attrs
        assert grid.shape == (3,)

    def test_grid_shape_property(self):
        """Test that the shape property returns the correct shape"""
        # Test with 1D arrays
        x = np.array([0, 1, 2, 3])
        y = np.array([0, 1, 2, 3])
        grid = Grid(x=x, y=y)
        assert grid.shape == (4,)

        # Test with 2D arrays
        x = np.array([[0, 1, 2], [0, 1, 2]])
        y = np.array([[0, 0, 0], [1, 1, 1]])
        grid = Grid(x=x, y=y)
        assert grid.shape == (2, 3)

        # Test with 3D arrays
        x = np.zeros((2, 3, 4))
        y = np.zeros((2, 3, 4))
        z = np.zeros((2, 3, 4))
        grid = Grid(x=x, y=y, z=z)
        assert grid.shape == (2, 3, 4)

    def test_grid_immutable(self):
        """Test that Grid is immutable (frozen)"""
        x = np.array([0, 1])
        y = np.array([0, 1])
        grid = Grid(x=x, y=y)

        # Attempting to modify should raise an error
        with pytest.raises(AttributeError):
            grid.x = np.array([2, 3])

        with pytest.raises(AttributeError):
            grid.new_attr = "value"

    def test_grid_creation_with_mismatched_shapes(self):
        """Test that Grid can be created with mismatched array shapes"""
        # This should be allowed since the Grid doesn't enforce shape matching
        x = np.array([0, 1])
        y = np.array([[0, 0], [1, 1]])  # Different shape

        grid = Grid(x=x, y=y)
        assert np.array_equal(grid.x, x)
        assert np.array_equal(grid.y, y)
        # Shape reflects the shape of the x array
        assert grid.shape == (2,)
