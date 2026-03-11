"""Tests for the typer-based lst-tools init subcommand."""

from __future__ import annotations

import re
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from lst_tools.config.schema import Config
from lst_tools.config.geometry import GeometryPreset, GEOMETRY_TEMPLATES
from lst_tools.config.merge import merge_dicts, merge_flow_defaults
from lst_tools.cli.main import cli

DEFAULTS = Config().to_dict()

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestMergeFlowDefaults:
    def test_merge_flow_defaults_no_flow_path(self):
        """Test that it initializes correctly WITHOUT a flow path"""
        result = merge_flow_defaults(DEFAULTS, None)
        assert result == DEFAULTS

    def test_merge_flow_defaults_with_valid_flow_path(self):
        """Test that it initializes correctly WITH a flow path"""
        from mocks import MOCK_FLOW_CONDITIONS_DAT

        result = merge_flow_defaults(DEFAULTS, MOCK_FLOW_CONDITIONS_DAT)

        assert "flow_conditions" in result
        assert result["flow_conditions"]["pres_0"] == 1362869.3601819816976786
        assert result["flow_conditions"]["temp_0"] == 450
        assert result["flow_conditions"]["mach"] == 5.2999999999999998
        assert "invalid_key" not in result["flow_conditions"]

    @patch("lst_tools.data_io.read_flow_conditions")
    def test_merge_flow_defaults_flow_read_exception(self, mock_read_flow, capsys):
        """Test that it handles exceptions when reading flow conditions file."""
        mock_read_flow.side_effect = OSError("File not found")

        mock_flow_path = MagicMock()
        mock_flow_path.exists.return_value = True

        result = merge_flow_defaults(DEFAULTS, mock_flow_path)

        # Should still return a valid dict even with exception
        assert result == DEFAULTS

    def test_merge_flow_defaults_flow_path_not_exists(self):
        """Ensure that if a flow path doesn't exist, initialize with default values anyways"""
        mock_flow_path = MagicMock()
        mock_flow_path.exists.return_value = False

        result = merge_flow_defaults(DEFAULTS, mock_flow_path)

        # Should return defaults unchanged
        assert result == DEFAULTS


class TestInitHelp:
    def test_init_help_shows_options(self):
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_RE.sub("", result.output)
        for opt in ("--out", "--force", "--flow", "--geometry"):
            assert opt in plain


class TestInitCommand:
    @patch("lst_tools.cli.cmd_init.write_config")
    def test_init_basic(self, mock_write_config, tmp_path):
        """Ensure that when `init` is invoked, it runs correctly"""
        out_file = tmp_path / "lst.cfg"
        mock_write_config.return_value = out_file

        result = runner.invoke(cli, ["init", "--out", str(out_file)])
        assert result.exit_code == 0
        mock_write_config.assert_called_once()

    @patch("lst_tools.cli.cmd_init.write_config")
    def test_init_file_exists_no_force(self, mock_write_config, tmp_path):
        """Ensure that when file exists without --force, a message is shown"""
        out_file = tmp_path / "lst.cfg"
        out_file.touch()
        mock_write_config.return_value = out_file

        result = runner.invoke(cli, ["init", "--out", str(out_file)])
        assert result.exit_code == 0
        assert "already exists" in result.output

    @patch("lst_tools.cli.cmd_init.write_config", side_effect=Exception("Permission denied"))
    def test_init_write_config_exception(self, mock_write_config, tmp_path):
        out_file = tmp_path / "lst.cfg"
        result = runner.invoke(cli, ["init", "--out", str(out_file)])
        assert result.exit_code == 1


class TestMergeDicts:
    def test_shallow_keys(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = merge_dicts(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_nested_merge(self):
        base = {"x": {"a": 1, "b": 2}, "y": 10}
        override = {"x": {"b": 99, "c": 3}}
        result = merge_dicts(base, override)
        assert result == {"x": {"a": 1, "b": 99, "c": 3}, "y": 10}

    def test_does_not_mutate_inputs(self):
        base = {"x": {"a": 1}}
        override = {"x": {"b": 2}}
        merge_dicts(base, override)
        assert base == {"x": {"a": 1}}
        assert override == {"x": {"b": 2}}


class TestGeometryPresets:
    """Verify each geometry preset produces the expected config values."""

    def test_cone_preset(self):
        seed = merge_dicts(DEFAULTS, GEOMETRY_TEMPLATES[GeometryPreset.cone])
        assert seed["geometry"]["type"] == 2
        assert seed["geometry"]["theta_deg"] == 7.0
        assert seed["geometry"]["r_nose"] == 5e-5
        assert seed["geometry"]["is_body_fitted"] is True
        assert seed["lst"]["solver"]["generalized"] == 0
        assert seed["lst"]["options"]["geometry_switch"] == 1
        assert seed["lst"]["options"]["longitudinal_curvature"] == 0

    def test_ogive_preset(self):
        seed = merge_dicts(DEFAULTS, GEOMETRY_TEMPLATES[GeometryPreset.ogive])
        assert seed["geometry"]["type"] == 3
        assert seed["geometry"]["is_body_fitted"] is False
        assert seed["lst"]["solver"]["generalized"] == 1
        assert seed["lst"]["options"]["geometry_switch"] == 1
        assert seed["lst"]["options"]["longitudinal_curvature"] == 1

    def test_flat_plate_preset(self):
        seed = merge_dicts(DEFAULTS, GEOMETRY_TEMPLATES[GeometryPreset.flat_plate])
        assert seed["geometry"]["type"] == 0
        assert seed["geometry"]["is_body_fitted"] is False
        assert seed["lst"]["solver"]["generalized"] == 0
        assert seed["lst"]["options"]["geometry_switch"] == 0
        assert seed["lst"]["options"]["longitudinal_curvature"] == 0

    def test_cylinder_preset(self):
        seed = merge_dicts(DEFAULTS, GEOMETRY_TEMPLATES[GeometryPreset.cylinder])
        assert seed["geometry"]["type"] == 1
        assert seed["geometry"]["is_body_fitted"] is False
        assert seed["lst"]["solver"]["generalized"] == 0
        assert seed["lst"]["options"]["geometry_switch"] == 0
        assert seed["lst"]["options"]["longitudinal_curvature"] == 0

    def test_all_presets_preserve_other_defaults(self):
        """Geometry presets should not clobber unrelated default fields."""
        for preset in GeometryPreset:
            seed = merge_dicts(DEFAULTS, GEOMETRY_TEMPLATES[preset])
            assert seed["input_file"] == DEFAULTS["input_file"]
            assert seed["lst"]["params"] == DEFAULTS["lst"]["params"]
            assert seed["lst"]["io"] == DEFAULTS["lst"]["io"]


class TestInitGeometryCLI:
    """Test the --geometry flag via the CLI runner."""

    @patch("lst_tools.cli.cmd_init.write_config")
    def test_init_geometry_cone(self, mock_write_config, tmp_path):
        out_file = tmp_path / "lst.cfg"
        mock_write_config.return_value = out_file

        result = runner.invoke(
            cli, ["init", "--out", str(out_file), "--geometry", "cone"]
        )
        assert result.exit_code == 0
        # write_config should receive cfg_data with cone presets applied
        call_kwargs = mock_write_config.call_args
        cfg_data = call_kwargs.kwargs.get("cfg_data") or call_kwargs[1].get("cfg_data")
        assert cfg_data is not None
        assert cfg_data["geometry"]["type"] == 2
        assert cfg_data["geometry"]["theta_deg"] == 7.0

    @patch("lst_tools.cli.cmd_init.write_config")
    def test_init_geometry_ogive(self, mock_write_config, tmp_path):
        out_file = tmp_path / "lst.cfg"
        mock_write_config.return_value = out_file

        result = runner.invoke(
            cli, ["init", "--out", str(out_file), "--geometry", "ogive"]
        )
        assert result.exit_code == 0
        call_kwargs = mock_write_config.call_args
        cfg_data = call_kwargs.kwargs.get("cfg_data") or call_kwargs[1].get("cfg_data")
        assert cfg_data is not None
        assert cfg_data["geometry"]["type"] == 3
        assert cfg_data["lst"]["solver"]["generalized"] == 1

    def test_init_geometry_help_shows_option(self):
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_RE.sub("", result.output)
        assert "--geometry" in plain

    def test_init_invalid_geometry(self):
        result = runner.invoke(cli, ["init", "--geometry", "wedge"])
        assert result.exit_code != 0
