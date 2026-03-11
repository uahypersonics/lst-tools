import numpy as np
import pytest
import os
from lst_tools.geometry.curvilinear_coordinate import curvilinear_coordinate


class TestCurvilinearCoordinate:
    """Test suite for curvilinear_coordinate function"""

    def test_curvilinear_coordinate_1d_arrays(self):
        """Test with 1D arrays - should return copy of x array"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0.1, 0.2, 0.3, 0.4])

        result = curvilinear_coordinate(x, y)

        # Should return copy of x array
        assert result.shape == x.shape
        assert np.allclose(result, x)
        # Should be a copy, not the same object
        assert result is not x

    def test_curvilinear_coordinate_2d_arrays_default_j(self):
        """Test with 2D arrays using default j=0"""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        result = curvilinear_coordinate(x, y)

        # Should return first row of x
        expected = np.array([1.0, 2.0, 3.0])
        assert result.shape == (3,)
        assert np.allclose(result, expected)

    def test_curvilinear_coordinate_2d_arrays_custom_j(self):
        """Test with 2D arrays using custom j"""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])

        result = curvilinear_coordinate(x, y, j=1)

        # Should return second row of x
        expected = np.array([4.0, 5.0, 6.0])
        assert result.shape == (3,)
        assert np.allclose(result, expected)

    def test_curvilinear_coordinate_invalid_shapes(self):
        """Test with arrays of different shapes - should raise ValueError"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([[0.1, 0.2], [0.3, 0.4]])  # Different shape

        with pytest.raises(ValueError, match="x and y must have the same shape"):
            curvilinear_coordinate(x, y)

    def test_curvilinear_coordinate_invalid_ndim(self):
        """Test with 3D arrays - should raise ValueError"""
        x = np.array([[[1.0, 2.0], [3.0, 4.0]]])  # 3D array
        y = np.array([[[0.1, 0.2], [0.3, 0.4]]])  # 3D array

        with pytest.raises(ValueError, match="expected 1-D or 2-D arrays"):
            curvilinear_coordinate(x, y)

    def test_curvilinear_coordinate_invalid_j_index(self):
        """Test with invalid j index - should raise IndexError"""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        with pytest.raises(IndexError, match="j index .* out of range"):
            curvilinear_coordinate(x, y, j=5)  # j=5 is out of range for 2 rows

    def test_curvilinear_coordinate_debug_1d(self, tmpdir):
        """Test debug output for 1D arrays"""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([0.1, 0.2, 0.3])

        curvilinear_coordinate(x, y, debug_path=str(tmpdir))

        debug_file = os.path.join(str(tmpdir), "curvilinear_coordinate_debug.dat")
        assert os.path.exists(debug_file)

        with open(debug_file, "r") as f:
            lines = f.readlines()

        assert lines[0].strip() == 'TITLE = "curvilinear coordinate debug"'
        assert lines[1].strip() == 'VARIABLES = "x" "y" "s"'
        assert lines[2].strip() == 'ZONE T="1D", I=3'
        assert len(lines) == 6
        for i in range(3):
            expected_line = f"{x[i]:.10e} {y[i]:.10e} {x[i]:.10e}\n"
            assert lines[3 + i] == expected_line

    def test_curvilinear_coordinate_debug_2d(self, tmpdir):
        """Test debug output for 2D arrays"""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        curvilinear_coordinate(x, y, j=1, debug_path=str(tmpdir))

        debug_file = os.path.join(str(tmpdir), "curvilinear_coordinate_debug.dat")
        assert os.path.exists(debug_file)

        with open(debug_file, "r") as f:
            lines = f.readlines()

        assert lines[0].strip() == 'TITLE = "curvilinear coordinate debug"'
        assert lines[1].strip() == 'VARIABLES = "x" "y" "s"'
        assert lines[2].strip() == 'ZONE T="j=1", I=3'
        assert len(lines) == 6
        expected_s = np.array([4.0, 5.0, 6.0])
        for i in range(3):
            expected_line = f"{x[1, i]:.10e} {y[1, i]:.10e} {expected_s[i]:.10e}\n"
            assert lines[3 + i] == expected_line
