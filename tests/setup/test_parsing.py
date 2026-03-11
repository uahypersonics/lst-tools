"""Tests for the typer-based lst-tools parsing subcommand."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()


class TestParsingCLI:
    """Test suite for parsing CLI module."""

    def test_help_shows_options(self):
        result = runner.invoke(cli, ["setup", "parsing", "--help"])
        assert result.exit_code == 0
        for opt in ("--cfg", "--out", "--name", "--auto-fill"):
            assert opt in result.output

    @patch("lst_tools.cli.cmd_parsing.parsing_setup")
    @patch("lst_tools.cli.cmd_parsing.read_config")
    def test_run_with_defaults(self, mock_read_config, mock_parsing_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_parsing_setup.return_value = "written_file_path"

        result = runner.invoke(cli, ["setup", "parsing"])
        assert result.exit_code == 0
        mock_read_config.assert_called_once_with(path=None)
        mock_parsing_setup.assert_called_once_with(
            cfg=mock_cfg, out_dir=Path("."), out_name="lst_input.dat",
            auto_fill=False, force=False, cfg_path=None,
        )

    @patch("lst_tools.cli.cmd_parsing.parsing_setup")
    @patch("lst_tools.cli.cmd_parsing.read_config")
    def test_run_with_custom_values(self, mock_read_config, mock_parsing_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_parsing_setup.return_value = "written_file_path"

        result = runner.invoke(
            cli,
            [
                "setup", "parsing",
                "--cfg", "/path/to/config.cfg",
                "--out", "/output/dir",
                "--name", "custom_name.dat",
            ],
        )
        assert result.exit_code == 0
        mock_read_config.assert_called_once_with(path=Path("/path/to/config.cfg"))
        mock_parsing_setup.assert_called_once_with(
            cfg=mock_cfg,
            out_dir=Path("/output/dir"),
            out_name="custom_name.dat",
            auto_fill=False,
            force=False,
            cfg_path=Path("/path/to/config.cfg"),
        )

    @patch("lst_tools.cli.cmd_parsing.parsing_setup")
    @patch("lst_tools.cli.cmd_parsing.read_config")
    def test_run_with_no_cfg(self, mock_read_config, mock_parsing_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_parsing_setup.return_value = "written_file_path"

        result = runner.invoke(cli, ["setup", "parsing"])
        assert result.exit_code == 0
        mock_read_config.assert_called_once_with(path=None)
        mock_parsing_setup.assert_called_once_with(
            cfg=mock_cfg, out_dir=Path("."), out_name="lst_input.dat",
            auto_fill=False, force=False, cfg_path=None,
        )

    @patch("lst_tools.cli.cmd_parsing.parsing_setup")
    @patch("lst_tools.cli.cmd_parsing.read_config")
    def test_run_with_auto_fill(self, mock_read_config, mock_parsing_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_parsing_setup.return_value = "written_file_path"

        result = runner.invoke(cli, ["setup", "parsing", "--auto-fill"])
        assert result.exit_code == 0
        mock_parsing_setup.assert_called_once_with(
            cfg=mock_cfg, out_dir=Path("."), out_name="lst_input.dat",
            auto_fill=True, force=False, cfg_path=None,
        )

    @patch("lst_tools.cli.cmd_parsing.parsing_setup")
    @patch("lst_tools.cli.cmd_parsing.read_config")
    def test_run_prints_result(self, mock_read_config, mock_parsing_setup):
        mock_read_config.return_value = Config()
        mock_parsing_setup.return_value = "/path/to/written/file.dat"

        result = runner.invoke(cli, ["setup", "parsing"])
        assert result.exit_code == 0
        assert "/path/to/written/file.dat" in result.output

    @patch("lst_tools.cli.cmd_parsing.parsing_setup")
    @patch("lst_tools.cli.cmd_parsing.read_config")
    def test_run_debug_output(self, mock_read_config, mock_parsing_setup, caplog):
        mock_read_config.return_value = Config()
        mock_parsing_setup.return_value = "written_file_path"

        with caplog.at_level(logging.DEBUG, logger="lst_tools"):
            result = runner.invoke(cli, ["--debug", "setup", "parsing"])
        assert result.exit_code == 0
        assert "setting up input deck for parsing step" in caplog.text

    @patch("lst_tools.cli.cmd_parsing.read_config", side_effect=Exception("Test error"))
    def test_run_handles_exceptions(self, mock_read_config):
        result = runner.invoke(cli, ["setup", "parsing"])
        assert result.exit_code != 0
