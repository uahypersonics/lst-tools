"""Tests for the typer-based lst-tools init subcommand."""

from __future__ import annotations

import copy
import re
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from lst_tools.cli.app import cli
from lst_tools.cli.cmd_init import _inject_init_comments
from lst_tools.config.geometry import GEOMETRY_TEMPLATES, GeometryPreset
from lst_tools.config.merge import merge_dicts, merge_flow_defaults
from lst_tools.config.schema import Config

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
        # read
        try:
            from tests.mocks import MOCK_FLOW_CONDITIONS_DAT
        except ModuleNotFoundError:
            from mocks import MOCK_FLOW_CONDITIONS_DAT

        result = merge_flow_defaults(DEFAULTS, MOCK_FLOW_CONDITIONS_DAT)

        assert "flow_conditions" in result
        assert result["flow_conditions"]["pres_0"] == 1362869.3601819816976786
        assert result["flow_conditions"]["temp_0"] == 450
        assert result["flow_conditions"]["mach"] == 5.2999999999999998
        assert "invalid_key" not in result["flow_conditions"]

    def test_merge_flow_defaults_with_flow_state_json(self, tmp_path):
        """Flow-state JSON values should populate the lst.cfg schema."""
        flow_file = tmp_path / "flow_conditions.json"
        flow_file.write_text(
            """
            {
              "transport_model": {"type": "sutherland"},
              "pres": [654.8743896484375, "Pa"],
              "temp": [71.59325408935547, "K"],
              "dens": [0.031854961421076244, "kg/m^3"],
              "mach": [3.95, "-"],
              "uvel": [670.1184496810956, "m/s"],
              "re1": [4398841.682241216, "1/m"],
              "cp": [1005.0250000000001, "J/(kg*K)"],
              "cv": [717.8750000000001, "J/(kg*K)"],
              "gamma": [1.4, "-"],
              "r_gas": [287.15, "J/(kg*K)"],
              "pr": [0.71, "-"],
              "pres_stag": [92999.9513475091, "Pa"],
              "temp_stag": [295.0000034751892, "K"]
            }
            """,
            encoding="utf-8",
        )

        result = merge_flow_defaults(DEFAULTS, flow_file)
        flow_conditions = result["flow_conditions"]

        assert flow_conditions["mach"] == 3.95
        assert flow_conditions["re1"] == 4398841.682241216
        assert flow_conditions["pres_inf"] == 654.8743896484375
        assert flow_conditions["temp_inf"] == 71.59325408935547
        assert flow_conditions["dens_inf"] == 0.031854961421076244
        assert flow_conditions["uvel_inf"] == 670.1184496810956
        assert flow_conditions["pres_0"] == 92999.9513475091
        assert flow_conditions["temp_0"] == 295.0000034751892
        assert flow_conditions["rgas"] == 287.15
        assert flow_conditions["visc_law"] == 0

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
        for opt in ("--out", "--force", "--merge", "--flow", "--geometry"):
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

        call_kwargs = mock_write_config.call_args
        cfg_data = call_kwargs.kwargs.get("cfg_data") or call_kwargs[1].get("cfg_data")
        assert cfg_data is not None
        assert "tracking" in cfg_data["processing"]
        assert "spectra" in cfg_data["processing"]
        assert "parsing" not in cfg_data["processing"]
        assert cfg_data["processing"]["spectra"] == {
            "alpr_min": None,
            "alpr_max": None,
            "alpi_min": None,
            "alpi_max": None,
        }

    @patch("lst_tools.cli.cmd_init.write_config")
    def test_init_file_exists_no_force(self, mock_write_config, tmp_path):
        """Ensure that when file exists without --force, a message is shown"""
        out_file = tmp_path / "lst.cfg"
        out_file.touch()
        mock_write_config.return_value = out_file

        result = runner.invoke(cli, ["init", "--out", str(out_file)])
        assert result.exit_code == 0
        assert "already exists" in result.output

    def test_init_merge_rewrites_existing_config_with_new_defaults(self, tmp_path, monkeypatch):
        """Merge should preserve existing values while infusing new scaffold keys."""
        out_file = tmp_path / "lst.cfg"
        out_file.write_text(
            "input_file = \"legacy_baseflow.hdf5\"\n"
            "lst_exe = \"legacy_lst.x\"\n\n"
            "[flow_conditions]\n"
            "mach = 5.5\n"
            "pr = 0.88\n\n"
            "[extract]\n"
            "n_eta = 321\n"
            "eta_distribution = \"uniform\"\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(cli, ["init", "--out", str(out_file), "--merge"])

        assert result.exit_code == 0

        config_text = out_file.read_text(encoding="utf-8")
        assert 'input_file = "legacy_baseflow.hdf5"' in config_text
        assert 'lst_exe = "legacy_lst.x"' in config_text
        assert 'mach = 5.5' in config_text
        assert 'pr = 0.88' in config_text
        assert 'n_eta = 321' in config_text
        assert 'eta_distribution = "uniform"' in config_text
        assert 'eta_max = ""' in config_text
        assert 'eta_stretch = 2.0' in config_text
        assert 'eta_wall_spacing = ""' in config_text

    @patch("lst_tools.cli.cmd_init.write_config", side_effect=Exception("Permission denied"))
    def test_init_write_config_exception(self, mock_write_config, tmp_path):
        out_file = tmp_path / "lst.cfg"
        result = runner.invoke(cli, ["init", "--out", str(out_file)])
        assert result.exit_code == 1

    @patch("lst_tools.cli.cmd_init.merge_flow_defaults")
    @patch("lst_tools.cli.cmd_init.write_config")
    def test_init_flow_path_hdf5_autodetect_and_comment_injection(
        self,
        mock_write_config,
        mock_merge_flow_defaults,
        tmp_path,
        monkeypatch,
    ):
        """Explicit flow path, single HDF5 auto-detect, and comment injection all apply."""
        out_file = tmp_path / "lst.cfg"
        flow_file = tmp_path / "flow_conditions.dat"
        hdf5_file = tmp_path / "meanflow.hdf5"

        flow_file.write_text("mach = 5.0\n", encoding="utf-8")
        hdf5_file.write_text("not-really-hdf5", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        mock_merge_flow_defaults.return_value = copy.deepcopy(DEFAULTS)

        def _write_config_side_effect(out, *, overwrite, cfg_data):
            out.write_text(
                "[processing.spectra]\n"
                "alpr_min = \"\"\n"
                "alpr_max = \"\"\n"
                "alpi_min = \"\"\n"
                "alpi_max = \"\"\n",
                encoding="utf-8",
            )
            return out

        mock_write_config.side_effect = _write_config_side_effect

        result = runner.invoke(
            cli,
            ["init", "--out", str(out_file), "--flow", str(flow_file)],
        )

        assert result.exit_code == 0
        assert out_file.exists()

        merge_args = mock_merge_flow_defaults.call_args.args
        assert merge_args[1] == flow_file

        write_kwargs = mock_write_config.call_args.kwargs
        cfg_data = write_kwargs["cfg_data"]
        assert cfg_data["input_file"] == "meanflow.hdf5"

        config_text = out_file.read_text(encoding="utf-8")
        assert "# Optional alpha-space gates for spectra post-processing." in config_text
        assert "# Leave any bound empty to disable it." in config_text


class TestInitFormatting:
    def test_inject_init_comments_adds_spectra_guidance_once(self):
        config_text = (
            "[processing.tracking]\n"
            "interpolate = false\n\n"
            "[processing.spectra]\n"
            "alpr_min = \"\"\n"
            "alpr_max = \"\"\n"
            "alpi_min = \"\"\n"
            "alpi_max = \"\"\n"
        )

        updated_text = _inject_init_comments(config_text)
        updated_text_twice = _inject_init_comments(updated_text)

        assert "# Optional alpha-space gates for spectra post-processing." in updated_text
        assert "# Leave any bound empty to disable it." in updated_text
        assert updated_text.count("# Optional alpha-space gates for spectra post-processing.") == 1
        assert updated_text_twice == updated_text

    def test_inject_init_comments_annotates_populated_flow_values(self):
        config_text = (
            "[flow_conditions]\n"
            "mach = 3.95\n"
            "re1 = 4398841.682241216\n"
            "pres_inf = 654.8743896484375\n"
            "temp_inf = 71.59325408935547\n"
            "visc_law = 0\n"
        )

        updated_text = _inject_init_comments(config_text)
        updated_text_twice = _inject_init_comments(updated_text)

        assert "# freestream mach number (required)\nmach = 3.95" in updated_text
        assert "# unit reynolds number (1/m) (required)\nre1 = " in updated_text
        assert "# freestream pressure [Pa] (optional)\npres_inf = " in updated_text
        assert "# freestream temperature [K] (required)\ntemp_inf = " in updated_text
        assert "# viscosity law: 0 = Sutherland, 1 = power law\nvisc_law = 0" in updated_text
        assert updated_text_twice == updated_text

    def test_inject_init_comments_annotates_populated_geometry_values(self):
        config_text = (
            "[geometry]\n"
            "type = 2\n"
            "theta_deg = 7.0\n"
            "r_nose = 5e-05\n"
            "l_ref = 1.0\n"
            "is_body_fitted = true\n\n"
            "[lst.solver]\n"
            "type = 1\n"
        )

        updated_text = _inject_init_comments(config_text)
        updated_text_twice = _inject_init_comments(updated_text)

        assert "# geometry type (required): 0=flat-plate" in updated_text
        assert "# half-angle [deg] — cone" in updated_text
        assert "# nose radius [m] — cone" in updated_text
        assert "# reference length [m]\nl_ref = 1.0" in updated_text
        assert "# cone only: true if grid is body-fitted" in updated_text
        assert updated_text.count("# geometry type (required): 0=flat-plate") == 1
        assert "# solver type: 1=global parallel, 2=tracking, 3=3-D tracking" in updated_text
        assert updated_text_twice == updated_text


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
        assert seed["geometry"]["is_body_fitted"] is True
        assert seed["lst"]["solver"]["generalized"] == 1
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
