import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from lst_tools.core import Grid, Flow
from lst_tools.geometry.kinds import GeometryKind
from lst_tools.convert.lastrac import convert_meanflow
from lst_tools.config.schema import Config


class TestConvertMeanflow:
    """Test suite for convert_meanflow function."""

    @pytest.fixture
    def mock_grid(self):
        """Create a mock Grid object for testing."""
        x = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        y = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        return Grid(x=x, y=y, z=None, cfg={})

    @pytest.fixture
    def mock_flow(self, mock_grid):
        """Create a mock Flow object for testing."""
        fields = {
            "uvel": np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
            "vvel": np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            "temp": np.array([[300.0, 301.0, 302.0], [299.0, 300.0, 301.0]]),
            "pres": np.array(
                [[101325.0, 101325.0, 101325.0], [101325.0, 101325.0, 101325.0]]
            ),
        }
        attrs = {}
        return Flow(grid=mock_grid, fields=fields, attrs=attrs)

    @pytest.fixture
    def base_config(self):
        """Create a base configuration dictionary."""
        return Config.from_dict({
            "geometry": {
                "type": GeometryKind.FLAT_PLATE,
                "is_body_fitted": True,
                "l_ref": 1.0,
                "r_nose": 0.0,
            },
            "flow_conditions": {
                "re1": 1e6,
                "pres_inf": 101325.0,
                "temp_inf": 300.0,
                "uvel_inf": 100.0,
                "dens_inf": 1.225,
            },
            "meanflow_conversion": {
                "set_v_zero": False,
                "i_s": 0,
                "i_e": None,
                "d_i": 1,
            },
        })

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_basic_conversion(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test basic meanflow conversion with minimal configuration."""
        # Setup mocks
        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        # Mock writer instance
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        # Mock progress context manager
        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion - the function doesn't return a value on success
        output_path = Path("test_output.bin")
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out=output_path,
            cfg=base_config,
            format="binary",
        )

        assert result == output_path

        # Verify the conversion process was executed correctly
        mock_curvilinear.assert_called_once()
        mock_surface_angle.assert_called_once()
        mock_curvature.assert_called_once()
        mock_radius.assert_called_once()
        mock_writer.write_header.assert_called_once()
        assert mock_writer.write_station_header.call_count == 3  # 3 stations
        assert (
            mock_writer.write_station_vector.call_count == 18
        )  # 6 vectors × 3 stations
        mock_writer.close.assert_called_once()

    def test_missing_config_raises_error(self, mock_grid, mock_flow):
        """Test that missing configuration raises ValueError."""
        with pytest.raises(ValueError, match="requires a configuration dictionary"):
            convert_meanflow(grid=mock_grid, flow=mock_flow, out="output.bin", cfg=None)

    def test_missing_geometry_type(self, mock_grid, mock_flow):
        """Test that missing geometry type raises appropriate error."""
        bad_config = Config.from_dict({"geometry": {}, "flow_conditions": {}, "meanflow_conversion": {}})

        with pytest.raises(ValueError, match="geometry type must be specified"):
            convert_meanflow(
                grid=mock_grid, flow=mock_flow, out="output.bin", cfg=bad_config
            )

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_rotation_with_surface_angle(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test velocity rotation when surface angle is non-negligible."""
        # Setup mocks with non-zero surface angle
        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.1, 0.2, 0.3])  # Non-zero angles
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out="test.bin",
            cfg=base_config,
            format="binary",
        )

        assert result == Path("test.bin")
        # Check that rotation was applied (writer gets called with rotated velocities)
        assert mock_writer.write_station_vector.called

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_body_fitted_error(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test error when grid appears body-fitted but config says otherwise."""
        # Setup mocks with zero surface angle but is_body_fitted = False
        base_config.geometry.is_body_fitted = False

        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])  # Zero angles
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion and expect ValueError
        with pytest.raises(ValueError, match="body fitted"):
            convert_meanflow(
                grid=mock_grid,
                flow=mock_flow,
                out="test.bin",
                cfg=base_config,
                format="binary",
            )

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_set_v_zero_option(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test that set_v_zero option correctly zeros v-velocity."""
        # Enable set_v_zero
        base_config.meanflow_conversion.set_v_zero = True

        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out="test.bin",
            cfg=base_config,
            format="binary",
        )

        assert result == Path("test.bin")
        # Check that v-velocity vectors written are zeros
        # We expect write_station_vector to be called with zeros for v-velocity
        calls = mock_writer.write_station_vector.call_args_list
        # Every 3rd call starting from index 2 should be v-velocity (pattern: eta, u, v, w, T, p)
        for i in range(2, len(calls), 6):
            v_array = calls[i][0][0]
            np.testing.assert_array_equal(v_array, np.zeros_like(v_array))

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_cone_geometry_zero_curvature(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test that body-fitted cone geometry sets curvature to zero."""
        # Configure for cone geometry
        base_config.geometry.type = GeometryKind.CONE
        base_config.geometry.is_body_fitted = True

        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.1, 0.2, 0.3])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out="test.bin",
            cfg=base_config,
            format="binary",
        )

        assert result == Path("test.bin")
        # Curvature function should not be called for body-fitted cone
        mock_curvature.assert_not_called()

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_station_indexing(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test custom station indexing with i_s, i_e, d_i."""
        # Custom indexing: start=1, end=3, stride=2 (should give station 1 only from [0,1,2])
        base_config.meanflow_conversion.i_s = 1
        base_config.meanflow_conversion.i_e = 3
        base_config.meanflow_conversion.d_i = 2

        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out="test.bin",
            cfg=base_config,
            format="binary",
        )

        assert result == Path("test.bin")
        # Should only process 1 station (index 1) with stride 2
        assert mock_writer.write_station_header.call_count == 1
        # Check that station header was called with correct i_loc (fortran indexing: i_loc+1)
        mock_writer.write_station_header.assert_called_with(
            i_loc=2,  # Fortran index = Python index + 1
            n_eta=2,
            s=1.0,
            lref=1.0,
            re1=1e6,
            kappa=0.0,
            rloc=0.0,
            drdx=0.0,
            stat_temp=300.0,
            stat_uvel=100.0,
            stat_dens=1.225,
        )

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_with_wvel_field(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        base_config,
    ):
        """Test handling of optional wvel field when present."""
        # Create grid and flow with wvel field
        x = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        y = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
        grid = Grid(x=x, y=y)

        fields = {
            "uvel": np.array([[1.0, 1.1, 1.2], [0.9, 1.0, 1.1]]),
            "vvel": np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
            # Add wvel
            "wvel": np.array([[0.05, 0.05, 0.05], [0.05, 0.05, 0.05]]),
            "temp": np.array([[300.0, 301.0, 302.0], [299.0, 300.0, 301.0]]),
            "pres": np.array(
                [[101325.0, 101325.0, 101325.0], [101325.0, 101325.0, 101325.0]]
            ),
        }
        flow = Flow(grid=grid, fields=fields, attrs={})

        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run conversion
        result = convert_meanflow(
            grid=grid, flow=flow, out="test.bin", cfg=base_config, format="binary"
        )

        assert result == Path("test.bin")
        # Check that wvel was written (not zeros)
        calls = mock_writer.write_station_vector.call_args_list
        # Every 4th call starting from index 3 should be wvel (pattern: eta, u, v, w, T, p)
        for i in range(3, len(calls), 6):
            w_array = calls[i][0][0]
            # Should be the actual wvel values, not zeros
            assert not np.all(w_array == 0)

    def test_invalid_stride(self, mock_grid, mock_flow, base_config):
        """Test that invalid stride raises error."""
        base_config.meanflow_conversion.d_i = 0  # Invalid stride

        with pytest.raises(ValueError, match="stride d_i must be a positive integer"):
            convert_meanflow(
                grid=mock_grid, flow=mock_flow, out="test.bin", cfg=base_config
            )

    def test_invalid_station_range(self, mock_grid, mock_flow, base_config):
        """Test that invalid station range raises error."""
        base_config.meanflow_conversion.i_s = 5  # Out of range

        with pytest.raises(ValueError, match="i_s out of range"):
            convert_meanflow(
                grid=mock_grid, flow=mock_flow, out="test.bin", cfg=base_config
            )

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_debug_mode(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
        capsys,
    ):
        """Test debug output is produced when debug_path is set."""
        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Run with debug_path set
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out="test.bin",
            cfg=base_config,
            format="binary",
            debug_path="./debug",
        )

        assert result == Path("test.bin")
        # Check debug output was logged
        with patch("lst_tools.convert.lastrac.logger") as mock_logger:
            convert_meanflow(
                grid=mock_grid,
                flow=mock_flow,
                out="test2.bin",
                cfg=base_config,
                format="binary",
                debug_path="./debug",
            )
            messages = " ".join(
                str(c) for c in mock_logger.debug.call_args_list
            )
            assert "start lastrac format conversion" in messages

    @patch("lst_tools.convert.lastrac.LastracWriter")
    @patch("lst_tools.convert.lastrac.radius")
    @patch("lst_tools.convert.lastrac.curvature")
    @patch("lst_tools.convert.lastrac.surface_angle")
    @patch("lst_tools.convert.lastrac.curvilinear_coordinate")
    @patch("lst_tools.convert.lastrac.progress")
    def test_pathlib_path_output(
        self,
        mock_progress,
        mock_curvilinear,
        mock_surface_angle,
        mock_curvature,
        mock_radius,
        mock_writer_class,
        mock_grid,
        mock_flow,
        base_config,
    ):
        """Test that Path objects are handled correctly for output."""
        mock_curvilinear.return_value = np.array([0.0, 1.0, 2.0])
        mock_surface_angle.return_value = np.array([0.0, 0.0, 0.0])
        mock_curvature.return_value = np.array([0.0, 0.0, 0.0])
        mock_radius.return_value = np.array([0.0, 0.0, 0.0])

        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        mock_progress_ctx = MagicMock()
        mock_progress_ctx.__enter__.return_value = lambda: None
        mock_progress.return_value = mock_progress_ctx

        # Use Path object for output
        output_path = Path("test_output.bin")
        result = convert_meanflow(
            grid=mock_grid,
            flow=mock_flow,
            out=output_path,
            cfg=base_config,
            format="binary",
        )

        assert result == output_path
        # Verify writer was initialized with the path
        mock_writer_class.assert_called_once_with(
            output_path, endianness="<", int_dtype=np.int32, real_dtype=np.float64
        )
