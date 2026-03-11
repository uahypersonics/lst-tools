import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from lst_tools.convert.lst_input import generate_lst_input_deck
from lst_tools.config.schema import Config


def _complete_config(**overrides) -> Config:
    """Build a Config with all fields populated for the formatter.

    The pure formatter requires every field to have a concrete value.
    This helper fills in the nullable fields (geometry_switch, theta_deg,
    x_s, x_e, i_step, f_min, f_max, d_f, beta_s, beta_e, d_beta) that
    the schema intentionally leaves as None (they are auto-filled by
    setup code).
    """
    base = {
        "lst": {
            "solver": {},
            "options": {"geometry_switch": 0},
            "params": {
                "x_s": 0.0,
                "x_e": 1.0,
                "i_step": 1,
                "f_min": 0.0,
                "f_max": 100.0,
                "d_f": 10.0,
                "beta_s": 0.0,
                "beta_e": 100.0,
                "d_beta": 10.0,
            },
            "io": {},
        },
        "flow_conditions": {},
        "geometry": {"theta_deg": 0.0},
    }

    # apply any overrides
    for key, val in overrides.items():
        # support dot-separated keys like "lst.solver.type"
        parts = key.split(".")
        target = base
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = val

    return Config.from_dict(base)


class TestLstInput:
    """Test suite for generate_lst_input_deck function"""

    def test_generate_lst_input_deck_minimal_config(self):
        """Test with minimal configuration - should use all default values"""
        config = _complete_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            result_path = generate_lst_input_deck(
                cfg=config, out_path=output_path
            )

            # Check that file was created
            assert result_path.exists()
            assert result_path == output_path

            # Read the content and verify basic structure
            content = result_path.read_text()
            content.strip().split("\n")

            # Check that header sections exist
            assert "SOLVER TYPE AND PROBLEM TYPE" in content
            assert "GEOMETRY PARAMETERS" in content
            assert "STABILITY PARAMETERS" in content
            assert "BASEFLOW OPTIONS AND SCALING" in content
            assert "INPUT/OUTPUT OPTIONS" in content

            # Check some default values are present - corrected format
            assert (
                "1005.0250000000                                    : Specific heat at constant pressure (cp)"
                in content
            )
            # Default alpha_i_threshold
            assert (
                "0.0000000000        , 100.0000000000      , 10.0000000000       , 0.0000000000         : Spanwise/Azimuthal wavenumber (beta_in): beta_min, beta_max, beta_step, beta_init"
                in content
            )

    def test_generate_lst_input_deck_full_config(self):
        """Test with full configuration - should use provided values"""
        config = Config.from_dict({
            "lst": {
                "solver": {
                    "type": 2,
                    "is_simplified": False,
                    "alpha_i_threshold": 0.001,
                    "generalized": 1,
                    "spatial_temporal": 0,
                    "energy_formulation": 2,
                },
                "options": {"geometry_switch": 1, "longitudinal_curvature": True},
                "params": {
                    "ny": 200,
                    "yl_in": 0.5,
                    "tol_lst": 1e-6,
                    "max_iter": 20,
                    "x_s": 0.1,
                    "x_e": 2.0,
                    "i_step": 2,
                    "f_min": 0.1,
                    "f_max": 1.5,
                    "d_f": 0.05,
                    "f_init": 0.5,
                    "beta_s": 1.0,
                    "beta_e": 50.0,
                    "d_beta": 5.0,
                    "beta_init": 2.0,
                    "alpha_0": complex(0.1, 0.02),
                },
                "io": {
                    "baseflow_input": "test_meanflow.dat",
                    "solution_output": "test_growth_rate.dat",
                },
            },
            "flow_conditions": {"visc_law": 1, "gamma": 1.3, "cp": 1000.0},
            "geometry": {"theta_deg": 5.0},
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            result_path = generate_lst_input_deck(cfg=config, out_path=output_path)

            # Check that file was created
            assert result_path.exists()

            # Read the content and verify values
            content = result_path.read_text()

            # Check solver configuration
            assert "2,F" in content  # Solver type and is_simplified
            assert "0.0010000000" in content  # alpha_i_threshold
            assert "1" in content  # Generalized solver
            assert "0" in content  # Spatial/temporal problem

            # Check geometry parameters
            assert "1" in content  # Geometry switch
            assert "5.0000000000" in content  # Theta degrees
            assert "1" in content  # Longitudinal curvature (true = 1)

            # Check stability parameters
            assert "200" in content  # ny
            assert "0.5000000000" in content  # yl_in
            assert "0.0000010000" in content  # tol_lst
            assert "20" in content  # max_iter

            # Check IO options
            assert "test_meanflow.dat" in content
            assert "test_growth_rate.dat" in content

    def test_generate_lst_input_deck_missing_config_sections(self):
        """Test that bare Config() raises ValueError for missing required fields.

        Config() leaves nullable sweep fields (x_s, x_e, i_step, etc.)
        as None.  The validation guard should raise a ValueError listing
        the missing fields.
        """
        config = Config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            with pytest.raises(ValueError, match="--auto-fill"):
                generate_lst_input_deck(cfg=config, out_path=output_path)

    def test_generate_lst_input_deck_none_config(self):
        """Test that cfg=None falls back to Config() which raises ValueError.

        Config() has None nullable sweep fields, so the validation guard
        raises ValueError before formatting is attempted.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            with pytest.raises(ValueError, match="--auto-fill"):
                generate_lst_input_deck(cfg=None, out_path=output_path)

    def test_generate_lst_input_deck_debug_mode(self):
        """Test debug mode outputs configuration information"""
        config = _complete_config(**{
            "lst.solver.type": 1,
            "flow_conditions.gamma": 1.4,
            "geometry.theta_deg": 0.0,
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            with patch("lst_tools.convert.lst_input.logger") as mock_logger:
                generate_lst_input_deck(cfg=config, out_path=output_path)

                # Check debug output was logged
                messages = " ".join(
                    str(c) for c in mock_logger.debug.call_args_list
                )
                assert "flow conditions" in messages.lower()
                assert "solver configuration" in messages.lower()

    def test_generate_lst_input_deck_complex_alpha_0_formatting(self):
        """Test complex alpha_0 value formatting"""
        config = _complete_config(**{
            "lst.params.alpha_0": complex(0.123456789, -0.987654321),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            result_path = generate_lst_input_deck(cfg=config, out_path=output_path)

            content = result_path.read_text()
            # Check that complex number is formatted with 10 decimal places
            assert "(0.1234567890,-0.9876543210)" in content

    def test_generate_lst_input_deck_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        config = _complete_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a nested path that doesn't exist
            output_path = Path(tmpdir) / "nested" / "subdir" / "test_input.dat"
            result_path = generate_lst_input_deck(cfg=config, out_path=output_path)

            # Directory should be created and file should exist
            assert result_path.exists()
            assert result_path.parent.exists()

    def test_generate_lst_input_deck_return_path_type(self):
        """Test that function returns Path object"""
        config = _complete_config()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_input.dat"
            result_path = generate_lst_input_deck(cfg=config, out_path=output_path)

            # Should return Path object
            assert isinstance(result_path, Path)
            assert result_path == output_path
