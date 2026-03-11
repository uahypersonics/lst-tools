"""Tests for Config.validate() constraint checking."""

from __future__ import annotations

import pytest

from lst_tools.config.schema import Config


class TestValidateConstraints:
    """Config.validate() enforces field constraints."""

    def test_default_config_is_valid(self):
        cfg = Config()
        assert cfg.validate() is cfg

    def test_valid_config_returns_self(self):
        cfg = Config.from_dict({
            "meanflow_conversion": {"i_s": 0, "d_i": 1},
            "geometry": {"r_nose": 0.5, "theta_deg": 7.0},
            "lst": {"solver": {"spatial_temporal": 1}},
        })
        assert cfg.validate() is cfg

    # -- meanflow_conversion -----------------------------------------------

    def test_i_s_negative(self):
        cfg = Config()
        cfg.meanflow_conversion.i_s = -1
        with pytest.raises(ValueError, match="i_s must be >= 0"):
            cfg.validate()

    def test_i_s_zero_is_valid(self):
        cfg = Config.from_dict({"meanflow_conversion": {"i_s": 0}})
        cfg.validate()

    def test_i_e_negative(self):
        cfg = Config()
        cfg.meanflow_conversion.i_e = -1
        with pytest.raises(ValueError, match="i_e must be >= 0"):
            cfg.validate()

    def test_i_e_none_is_valid(self):
        cfg = Config.from_dict({"meanflow_conversion": {"i_e": None}})
        cfg.validate()

    def test_d_i_zero(self):
        cfg = Config()
        cfg.meanflow_conversion.d_i = 0
        with pytest.raises(ValueError, match="d_i must be >= 1"):
            cfg.validate()

    # -- geometry ----------------------------------------------------------

    def test_r_nose_negative(self):
        cfg = Config()
        cfg.geometry.r_nose = -1.0
        with pytest.raises(ValueError, match="r_nose must be >= 0"):
            cfg.validate()

    def test_r_nose_zero_is_valid(self):
        cfg = Config()
        cfg.geometry.r_nose = 0.0
        cfg.validate()

    def test_r_nose_none_is_valid(self):
        cfg = Config.from_dict({"geometry": {"r_nose": None}})
        cfg.validate()

    def test_theta_deg_out_of_range(self):
        cfg = Config()
        cfg.geometry.theta_deg = 200.0
        with pytest.raises(ValueError, match="theta_deg must be in"):
            cfg.validate()

    def test_theta_deg_boundaries(self):
        Config.from_dict({"geometry": {"theta_deg": 0}}).validate()
        Config.from_dict({"geometry": {"theta_deg": 180}}).validate()

    # -- lst.solver --------------------------------------------------------

    def test_solver_type_negative(self):
        cfg = Config()
        cfg.lst.solver.type = -1
        with pytest.raises(ValueError, match="solver.type must be >= 0"):
            cfg.validate()

    def test_solver_type_none_is_valid(self):
        Config.from_dict({"lst": {"solver": {"type": None}}}).validate()

    def test_spatial_temporal_invalid(self):
        cfg = Config()
        cfg.lst.solver.spatial_temporal = 2
        with pytest.raises(ValueError, match="spatial_temporal must be 0 or 1"):
            cfg.validate()

    def test_spatial_temporal_valid_values(self):
        Config.from_dict({"lst": {"solver": {"spatial_temporal": 0}}}).validate()
        Config.from_dict({"lst": {"solver": {"spatial_temporal": 1}}}).validate()

    # -- hpc ---------------------------------------------------------------

    def test_hpc_nodes_zero(self):
        cfg = Config()
        cfg.hpc.nodes = 0
        with pytest.raises(ValueError, match="nodes must be > 0"):
            cfg.validate()

    def test_hpc_nodes_negative(self):
        cfg = Config()
        cfg.hpc.nodes = -1
        with pytest.raises(ValueError, match="nodes must be > 0"):
            cfg.validate()

    def test_hpc_nodes_none_is_valid(self):
        cfg = Config()
        cfg.hpc.nodes = None
        cfg.validate()

    # -- multiple errors ---------------------------------------------------

    def test_collects_multiple_errors(self):
        """All violations reported in a single ValueError."""
        cfg = Config()
        cfg.meanflow_conversion.i_s = -5
        cfg.meanflow_conversion.d_i = 0
        cfg.geometry.r_nose = -1.0
        cfg.geometry.theta_deg = 200.0

        with pytest.raises(ValueError) as exc_info:
            cfg.validate()

        msg = str(exc_info.value)
        assert "i_s must be >= 0" in msg
        assert "d_i must be >= 1" in msg
        assert "r_nose must be >= 0" in msg
        assert "theta_deg must be in [0, 180]" in msg


class TestFromDictValidation:
    """Config.from_dict() calls validate() automatically."""

    def test_from_dict_raises_on_invalid(self):
        with pytest.raises(ValueError, match="i_s must be >= 0"):
            Config.from_dict({"meanflow_conversion": {"i_s": -1}})

    def test_from_dict_valid(self):
        cfg = Config.from_dict({"meanflow_conversion": {"i_s": 5}})
        assert cfg.meanflow_conversion.i_s == 5

    def test_from_dict_type_coercion(self):
        cfg = Config.from_dict({
            "meanflow_conversion": {"i_s": "5", "d_i": "2"},
        })
        assert cfg.meanflow_conversion.i_s == 5
        assert cfg.meanflow_conversion.d_i == 2

    def test_from_dict_complex_field(self):
        cfg = Config.from_dict({"lst": {"params": {"alpha_0": "(1.5, 2.3)"}}})
        assert cfg.lst.params.alpha_0 == 1.5 + 2.3j

    def test_from_dict_minimal_fills_defaults(self):
        cfg = Config.from_dict({})
        assert cfg.input_file == "base_flow.hdf5"
        assert cfg.lst_exe == "lst.x"
        assert cfg.meanflow_conversion.i_s == 0
        assert cfg.meanflow_conversion.d_i == 1
        assert cfg.lst.io.baseflow_input == "meanflow.bin"

    def test_from_dict_nested_sections(self):
        cfg = Config.from_dict({
            "lst": {"solver": {"type": 1}, "params": {"ny": 100}},
        })
        assert cfg.lst.solver.type == 1
        assert cfg.lst.params.ny == 100
        assert cfg.lst.solver.is_simplified is False
        assert cfg.lst.params.tol_lst == 1e-5

    def test_from_dict_unknown_sections_dropped(self):
        cfg = Config.from_dict({"unknown_section": {"field": "value"}})
        assert not hasattr(cfg, "unknown_section")
