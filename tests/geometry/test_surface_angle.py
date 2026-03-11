# --------------------------------------------------
# Tests for surface_angle module
# --------------------------------------------------
#
import numpy as np
import pytest
from lst_tools.geometry.surface_angle import surface_angle, _first_order, _second_order
from lst_tools.core.grid import Grid


@pytest.fixture
def simple_grid():
    """Create a simple 2D grid for testing."""
    # Create a simple grid with a straight line at j=0
    x = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    y = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    return Grid(x=x, y=y)


@pytest.fixture
def diagonal_grid():
    """Create a grid with a diagonal line at j=0."""
    # Create a grid with a diagonal line (45 degrees)
    x = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    y = np.array([[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])
    return Grid(x=x, y=y)


def test_first_order_horizontal_line(simple_grid, tmp_path):
    """Test first order method with a horizontal line."""
    result = _first_order(simple_grid, debug_path=tmp_path)

    # For a horizontal line, all angles should be 0
    expected = np.zeros(simple_grid.x.shape[1])
    np.testing.assert_array_almost_equal(result, expected)

    # Check that debug file was created
    debug_file = tmp_path / "surface_angle_first_order.dat"
    assert debug_file.exists()


def test_second_order_horizontal_line(simple_grid, tmp_path):
    """Test second order method with a horizontal line."""
    result = _second_order(simple_grid, debug_path=tmp_path)

    # For a horizontal line, all angles should be 0
    expected = np.zeros(simple_grid.x.shape[1])
    np.testing.assert_array_almost_equal(result, expected)

    # Check that debug file was created
    debug_file = tmp_path / "surface_angle_second_order.dat"
    assert debug_file.exists()


def test_first_order_diagonal_line(diagonal_grid, tmp_path):
    """Test first order method with a diagonal line."""
    result = _first_order(diagonal_grid, debug_path=tmp_path)

    # For a diagonal line, all angles should be π/4 (45 degrees)
    expected = np.full(diagonal_grid.x.shape[1], np.pi / 4)
    np.testing.assert_array_almost_equal(result, expected)


def test_second_order_diagonal_line(diagonal_grid, tmp_path):
    """Test second order method with a diagonal line."""
    result = _second_order(diagonal_grid, debug_path=tmp_path)

    # For a diagonal line, all angles should be π/4 (45 degrees)
    expected = np.full(diagonal_grid.x.shape[1], np.pi / 4)
    np.testing.assert_array_almost_equal(result, expected)


def test_surface_angle_method_dispatch(simple_grid):
    """Test that the dispatcher correctly selects methods."""
    # Test first_order method
    result1 = surface_angle(simple_grid, method="first_order")
    expected = np.zeros(simple_grid.x.shape[1])
    np.testing.assert_array_almost_equal(result1, expected)

    # Test second_order method
    result2 = surface_angle(simple_grid, method="second_order")
    np.testing.assert_array_almost_equal(result2, expected)


def test_surface_angle_invalid_method(simple_grid):
    """Test that an invalid method raises an error."""
    with pytest.raises(ValueError, match="unknown method='invalid_method'"):
        surface_angle(simple_grid, method="invalid_method")


def test_mismatched_array_shapes():
    """Test that mismatched x and y shapes raise an error."""
    x = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0]])

    grid = Grid(x=x, y=y)

    # numpy raises its own broadcast / shape error when the arrays differ
    with pytest.raises((ValueError, IndexError)):
        _first_order(grid)

    with pytest.raises((ValueError, IndexError)):
        _second_order(grid)
