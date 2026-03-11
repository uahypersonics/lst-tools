"""Tests for the typer-based lst-tools hpc subcommand."""

from __future__ import annotations

import re
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestHpcHelp:
    def test_help_shows_options(self):
        result = runner.invoke(cli, ["hpc", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_RE.sub("", result.output)
        assert "--cfg" in plain


class TestHpcCommand:
    @patch("lst_tools.cli.cmd_hpc.read_config")
    @patch("lst_tools.cli.cmd_hpc.hpc_configure")
    @patch("lst_tools.cli.cmd_hpc.script_build")
    def test_run_basic_debug(
        self,
        mock_script_build,
        mock_hpc_configure,
        mock_read_config,
    ):
        mock_read_config.return_value = Config()
        mock_hpc_cfg = MagicMock()
        mock_hpc_configure.return_value = mock_hpc_cfg
        mock_script_build.return_value = "script.sh"

        result = runner.invoke(cli, ["--debug", "hpc"])
        assert result.exit_code == 0
        assert "run script written to" in result.output

    @patch("lst_tools.cli.cmd_hpc.read_config", side_effect=Exception("Config error"))
    def test_run_config_exception_propagates(
        self,
        mock_read_config,
    ):
        result = runner.invoke(cli, ["hpc"])
        assert result.exit_code != 0

    @patch("lst_tools.cli.cmd_hpc.read_config", return_value=Config())
    @patch("lst_tools.cli.cmd_hpc.hpc_configure")
    @patch("lst_tools.cli.cmd_hpc.script_build", return_value="my_script.sh")
    def test_run_exec_none(
        self,
        mock_script_build,
        mock_hpc_configure,
        mock_read_config,
    ):
        mock_hpc_cfg = MagicMock()
        mock_hpc_configure.return_value = mock_hpc_cfg

        result = runner.invoke(cli, ["hpc"])
        assert result.exit_code == 0
        assert "run script written to my_script.sh" in result.output
