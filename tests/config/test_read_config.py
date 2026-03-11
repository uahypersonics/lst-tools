import sys
import tempfile
import os

import pytest
from unittest import mock
import tomli_w as toml_w

from lst_tools.config.schema import Config
from lst_tools.config.read_config import read_config

DEFAULTS = Config()
DEFAULTS_DICT = DEFAULTS.to_dict()


class TestReadConfig:
    def test_read_config_defaults_only(self):
        """Test read_config with defaults only (no config file)."""
        config = read_config()
        assert config == DEFAULTS

    def test_read_config_with_valid_toml_file(self, tmp_path):
        """Test read_config with a valid TOML configuration file."""
        # Create a temporary TOML config file
        config_data = {
            "lst_exe": "custom_lst.x",
            "flow_conditions": {"mach": 0.9, "re1": 2e6},
            "lst": {"params": {"ny": 200, "tol_lst": 1e-06}},
        }
        config_file = tmp_path / "test_config.toml"
        with open(config_file, "wb") as f:
            toml_w.dump(config_data, f)

        config = read_config(path=str(config_file))

        # Check that defaults are overridden by config file
        assert config.lst_exe == "custom_lst.x"
        assert config.flow_conditions.mach == 0.9
        assert config.flow_conditions.re1 == 2e6
        assert config.lst.params.ny == 200
        assert config.lst.params.tol_lst == 1e-06

        # Check that other defaults remain unchanged
        assert config.hpc.account == DEFAULTS.hpc.account

    def test_read_config_defaults_no_file(self):
        with mock.patch.dict(os.environ, {}):
            result = read_config()
            assert result == DEFAULTS

    def test_read_config_missing_file_with_path(self):
        """Test that read_config raises FileNotFoundError when explicit path is missing."""
        with pytest.raises(FileNotFoundError, match="configuration file not found"):
            read_config(path="/non/existent/config.toml")

    def test_read_config_with_mock_config_file(self):
        """Test read_config with the provided mock_config.cfg converted to TOML."""
        # Convert cfg format to TOML for testing
        mock_toml = {
            "input_file": "mock.hdf5",
            "debug": False,
            "lst_exe": "lst.x",
            "flow_conditions": {
                "mach": 0.8,
                "re1": 1e6,
                "gamma": 1.4,
                "cp": 1005.025,
                "cv": 717.875,
                "rgas": 287.15,
                "pres_0": 101325.0,
                "temp_0": 288.15,
                "pres_inf": 101325.0,
                "temp_inf": 288.15,
                "dens_inf": 1.225,
                "uvel_inf": 0.0,
                "visc_law": 0,
            },
            "geometry": {
                "type": 3,
                "theta_deg": 5.0,
                "r_nose": 1.25e-3,
                "l_ref": 1.0,
                "is_body_fitted": False,
            },
            "meanflow_conversion": {"i_s": 0, "i_e": 150, "d_i": 1, "set_v_zero": True},
            "lst": {
                "solver": {
                    "type": 1,
                    "is_simplified": True,
                    "alpha_i_threshold": -100.0,
                    "generalized": 1,
                    "spatial_temporal": 1,
                    "energy_formulation": 1,
                },
                "options": {"geometry_switch": 1, "longitudinal_curvature": 1},
                "params": {
                    "ny": 150,
                    "yl_in": 0.0,
                    "tol_lst": 1e-05,
                    "max_iter": 15,
                    "x_s": 0.001,
                    "x_e": 0.75,
                    "i_step": 1,
                    "f_min": 10000,
                    "f_max": 500000,
                    "d_f": 10000,
                    "f_init": 100000,
                    "beta_s": 0,
                    "beta_e": 100,
                    "d_beta": 5,
                    "beta_init": 0.0,
                    "alpha_0": "(0,0)",
                },
                "io": {
                    "baseflow_input": "meanflow.bin",
                    "solution_output": "growth_rate.dat",
                },
            },
            "hpc": {"account": "", "nodes": "", "time": "", "partition": ""},
            "plotting": {
                "cmap": "viridis",
                "levels": 50,
                "save": "",
                "colorbar_label": "",
                "equal_aspect": True,
            },
        }

        with tempfile.NamedTemporaryFile(mode="wb", suffix=".toml", delete=False) as f:
            toml_w.dump(mock_toml, f)
            config_file = f.name

        try:
            config = read_config(path=config_file)
            # Verify key values survive the round-trip (type coercion may
            # normalise some values, e.g. alpha_0 → complex, "" → None)
            assert isinstance(config, Config)
            assert config.input_file == "mock.hdf5"
            assert config.flow_conditions.mach == 0.8
            assert config.geometry.type == 3
            assert config.lst.solver.is_simplified is True
            assert config.lst.params.ny == 150
            # alpha_0 is coerced from string "(0,0)" to complex
            assert config.lst.params.alpha_0 == complex(0, 0)
            # empty-string integers are coerced to None
            assert config.hpc.nodes is None
        finally:
            os.unlink(config_file)

    def test_tomllib_import(self):
        with mock.patch.dict(
            "sys.modules",
            {"tomllib": mock.MagicMock(__name__="tomllib"), "tomli": None},
        ):
            for mod in ("lst_tools.config.schema", "lst_tools.config.read_config"):
                if mod in sys.modules:
                    del sys.modules[mod]

            import lst_tools.config.schema as schema

            assert schema._toml.__name__ == "tomllib"

    def test_tomli_import(self):
        with mock.patch.dict(
            "sys.modules", {"tomllib": None, "tomli": mock.MagicMock(__name__="tomli")}
        ):
            for mod in ("lst_tools.config.schema", "lst_tools.config.read_config"):
                if mod in sys.modules:
                    del sys.modules[mod]

            import lst_tools.config.schema as schema

            assert schema._toml.__name__ == "tomli"

    def test_toml_missing_import(self):
        # Save the original modules
        original_tomllib = sys.modules.get("tomllib")
        original_tomli = sys.modules.get("tomli")

        # Remove the modules and the module that imports them
        for module in ["tomllib", "tomli", "lst_tools.config.schema", "lst_tools.config.read_config"]:
            if module in sys.modules:
                del sys.modules[module]

        # Mock both imports to raise ImportError
        with mock.patch.dict("sys.modules", {"tomllib": None, "tomli": None}):
            with pytest.raises(
                ImportError, match="lst_tools.config requires 'tomli' on Python <3.11"
            ):
                # This import will trigger the module-level import code
                import lst_tools.config.schema  # noqa

        # Cleanup: restore original modules if they existed
        if original_tomllib is not None:
            sys.modules["tomllib"] = original_tomllib
        if original_tomli is not None:
            sys.modules["tomli"] = original_tomli
