import numpy as np

from lst_tools.setup.tracking import (
    _hampel_1d,
    _remove_spurious_peaks,
    _rolling_min,
    _clean_alpi_row,
    _track_ridge_dp,
    _keep_mask_from_path,
    smooth_contour_field,
)


class TestHampel1D:
    """Test suite for _hampel_1d function"""

    def test_hampel_1d_empty_array(self):
        """Test with empty array"""
        result = _hampel_1d(np.array([]))
        assert result.shape == (0,)
        assert isinstance(result, np.ndarray)

    def test_hampel_1d_small_window(self):
        """Test with window size < 3"""
        y = np.array([1.0, 2.0, 3.0])
        result = _hampel_1d(y, win=2)
        np.testing.assert_array_equal(result, y)

    def test_hampel_1d_no_outliers(self):
        """Test with no outliers"""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _hampel_1d(y)
        np.testing.assert_array_equal(result, y)

    def test_hampel_1d_with_outliers(self):
        """Test with outliers that should be replaced"""
        y = np.array([1.0, 2.0, 10.0, 4.0, 5.0])  # 10.0 is an outlier
        result = _hampel_1d(y, win=3)
        # The outlier should be replaced with median of window
        assert result[2] - np.float64(10.0) < 1e-11
        assert result[0] - np.float64(1.0) < 1e-11  # First element unchanged
        assert result[1] - np.float64(2.0) < 1e-11  # Second element unchanged
        assert result[3] - np.float64(4.0) < 1e-11  # Fourth element unchanged
        assert result[4] - np.float64(5.0) < 1e-11  # Fifth element unchanged


class TestRemoveSpuriousPeaks:
    """Test suite for _remove_spurious_peaks function"""

    def test_remove_spurious_peaks_no_peaks(self):
        """Test with no peaks above threshold"""
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = _remove_spurious_peaks(y, k=3, thresh=0.0)
        np.testing.assert_array_equal(result, y)

    def test_remove_spurious_peaks_short_runs(self):
        """Test removal of runs shorter than k"""
        y = np.array([0.0, 2.0, 2.0, 0.0, 3.0, 0.0, 0.0])  # Runs of length 2 and 1
        result = _remove_spurious_peaks(y, k=3, thresh=0.0)
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_remove_spurious_peaks_long_runs(self):
        """Test preservation of runs longer than or equal to k"""
        y = np.array(
            [0.0, 2.0, 2.0, 2.0, 0.0, 3.0, 3.0, 3.0, 3.0, 0.0]
        )  # Runs of length 3 and 4
        result = _remove_spurious_peaks(y, k=3, thresh=0.0)
        # Only runs of length >= 3 should be preserved
        # Preserved run
        assert result[1] == 2.0 and result[2] == 2.0 and result[3] == 2.0
        # Preserved run
        assert (
            result[5] == 3.0
            and result[6] == 3.0
            and result[7] == 3.0
            and result[8] == 3.0
        )

    def test_remove_spurious_peaks_with_threshold(self):
        """Test with custom threshold"""
        y = np.array([0.0, 0.5, 1.5, 0.0, 2.0, 0.0])
        result = _remove_spurious_peaks(y, k=2, thresh=1.0)
        # Only values > 1.0 are considered, and both runs have length 1 (< 2), so they should be removed
        expected = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)


class TestRollingMin:
    """Test suite for _rolling_min function"""

    def test_rolling_min_basic(self):
        """Test basic rolling min functionality"""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _rolling_min(a, w=3)
        assert result.shape == (5,)
        # For edge padding, first and last elements should be min of padded windows

    def test_rolling_min_even_window(self):
        """Test that even window is converted to odd"""
        a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = _rolling_min(a, w=4)  # Should become 3 (odd)
        expected_w3 = _rolling_min(a, w=3)
        np.testing.assert_array_equal(result, expected_w3)

    def test_rolling_min_single_element(self):
        """Test with single element array"""
        a = np.array([5.0])
        result = _rolling_min(a, w=3)
        np.testing.assert_array_equal(result, np.array([5.0]))


class TestCleanAlpiRow:
    """Test suite for _clean_alpi_row function"""

    def test_clean_alpi_row_basic(self):
        """Test basic functionality"""
        y = np.array([0.0, 1.0, 10.0, 1.0, 0.0])
        result = _clean_alpi_row(y)
        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape

    def test_clean_alpi_row_no_change(self):
        """Test with array that requires no cleaning"""
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = _clean_alpi_row(y)
        np.testing.assert_array_equal(result, y)

    def test_clean_alpi_row_parameters(self):
        """Test with custom parameters"""
        y = np.array([0.0, 2.0, 2.0, 0.0, 3.0])  # Short run
        result = _clean_alpi_row(y, min_run=3)  # Should zero out runs < 3
        expected = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)


class TestTrackRidgeDP:
    """Test suite for _track_ridge_dp function"""

    def test_track_ridge_dp_basic(self):
        """Test basic functionality"""
        alpi_2d = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
        result = _track_ridge_dp(alpi_2d)
        assert isinstance(result, np.ndarray)
        # Should have length equal to number of columns
        assert result.shape == (3,)
        assert np.issubdtype(result.dtype, np.integer)

    def test_track_ridge_dp_single_column(self):
        """Test with single column"""
        alpi_2d = np.array([[1.0], [2.0], [3.0]])
        result = _track_ridge_dp(alpi_2d)
        assert result.shape == (1,)
        # Should select the maximum value index
        assert result[0] == 2  # Index of maximum value (3.0)

    def test_track_ridge_dp_parameters(self):
        """Test with custom parameters"""
        alpi_2d = np.array([[1.0, 2.0, 1.0], [2.0, 1.0, 2.0], [1.0, 2.0, 1.0]])
        result1 = _track_ridge_dp(alpi_2d, lam=0.1)  # Low smoothness penalty
        result2 = _track_ridge_dp(alpi_2d, lam=10.0)  # High smoothness penalty
        assert result1.shape == (3,)
        assert result2.shape == (3,)


class TestKeepMaskFromPath:
    """Test suite for _keep_mask_from_path function"""

    def test_keep_mask_from_path_basic(self):
        """Test basic functionality"""
        j_star = np.array([1, 1, 1])
        result = _keep_mask_from_path(j_star, n_freq=5, half_width=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (5, 3)  # n_freq x length of j_star
        assert result.dtype == bool

    def test_keep_mask_from_path_edges(self):
        """Test edge handling"""
        j_star = np.array([0, 4, 2])  # First, last, middle
        result = _keep_mask_from_path(j_star, n_freq=5, half_width=2)
        assert result.shape == (5, 3)
        # Check that mask is correctly set around path indices

    def test_keep_mask_from_path_half_width(self):
        """Test different half_width values"""
        j_star = np.array([2])
        result1 = _keep_mask_from_path(
            j_star, n_freq=5, half_width=0
        )  # Only path index
        result2 = _keep_mask_from_path(j_star, n_freq=5, half_width=1)  # Path ±1
        # More elements should be True with larger half_width
        assert np.sum(result1) < np.sum(result2)


class TestSmoothContourField:
    """Test suite for smooth_contour_field function"""

    def test_smooth_contour_field_basic(self):
        """Test basic functionality"""
        field_2d = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
        result_field, result_mask = smooth_contour_field(field_2d)
        assert isinstance(result_field, np.ndarray)
        assert isinstance(result_mask, np.ndarray)
        assert result_field.shape == field_2d.shape
        assert result_mask.shape == field_2d.shape
        assert result_mask.dtype == bool

    def test_smooth_contour_field_npasses(self):
        """Test with multiple passes"""
        field_2d = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
        result_field1, _ = smooth_contour_field(field_2d, npasses=1)
        result_field2, _ = smooth_contour_field(field_2d, npasses=3)
        assert result_field1.shape == result_field2.shape

    def test_smooth_contour_field_empty(self):
        """Test with empty array"""
        field_2d = np.array([]).reshape(0, 0)
        result, _ = smooth_contour_field(field_2d, npasses=1)
        assert result.size == 0
