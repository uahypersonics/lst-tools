"""Tests for the typer-based lst-tools tracking subcommand."""

from __future__ import annotations

import re
from unittest.mock import patch

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestTrackingCLI:
    """Test suite for tracking CLI module."""

    def test_help_shows_options(self):
        result = runner.invoke(cli, ["setup", "tracking", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_RE.sub("", result.output)
        assert "--cfg" in plain

    @patch("lst_tools.cli.cmd_tracking.tracking_setup")
    @patch("lst_tools.cli.cmd_tracking.read_config")
    def test_run_without_debug(self, mock_read_config, mock_tracking_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        result = runner.invoke(cli, ["setup", "tracking"])
        assert result.exit_code == 0
        mock_read_config.assert_called_once_with(path=None)
        mock_tracking_setup.assert_called_once_with(cfg=mock_cfg, debug_path=None, auto_fill=False, force=False, cfg_path=None)

    @patch("lst_tools.cli.cmd_tracking.tracking_setup")
    @patch("lst_tools.cli.cmd_tracking.read_config")
    def test_run_with_debug(self, mock_read_config, mock_tracking_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        result = runner.invoke(cli, ["--debug", "setup", "tracking"])
        assert result.exit_code == 0
        mock_tracking_setup.assert_called_once()
        call_kwargs = mock_tracking_setup.call_args[1]
        assert call_kwargs["cfg"] is mock_cfg
        assert call_kwargs["debug_path"] is not None
        assert "tracking setup complete" in result.output

    @patch("lst_tools.cli.cmd_tracking.read_config", side_effect=Exception("Test error"))
    def test_run_handles_exceptions(self, mock_read_config):
        result = runner.invoke(cli, ["setup", "tracking"])
        assert result.exit_code != 0
