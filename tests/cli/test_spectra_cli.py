"""Tests for the typer-based lst-tools spectra setup subcommand."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestSpectraCLI:
    """Test suite for spectra setup CLI module."""

    def test_help_shows_options(self):
        result = runner.invoke(cli, ["setup", "spectra", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_RE.sub("", result.output)
        assert "--cfg" in plain

    @patch("lst_tools.cli.cmd_spectra.spectra_setup")
    @patch("lst_tools.cli.cmd_spectra.read_config")
    def test_run_with_defaults(self, mock_read_config, mock_spectra_setup):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_spectra_setup.return_value = [Path("x_000/lst_input.dat")]

        result = runner.invoke(cli, ["setup", "spectra"])

        assert result.exit_code == 0
        assert "spectra setup complete" in result.output
        mock_read_config.assert_called_once_with(path=None)
        mock_spectra_setup.assert_called_once_with(cfg=mock_cfg)

    @patch("lst_tools.cli.cmd_spectra.spectra_setup")
    @patch("lst_tools.cli.cmd_spectra.read_config")
    def test_run_with_cfg_path(self, mock_read_config, mock_spectra_setup, tmp_path: Path):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_spectra_setup.return_value = []

        cfg_path = tmp_path / "custom.cfg"
        cfg_path.write_text("", encoding="utf-8")

        result = runner.invoke(cli, ["setup", "spectra", "--cfg", str(cfg_path)])

        assert result.exit_code == 0
        mock_read_config.assert_called_once_with(path=cfg_path)

    @patch("lst_tools.cli.cmd_spectra.read_config", side_effect=Exception("boom"))
    def test_run_handles_exceptions(self, mock_read_config):
        result = runner.invoke(cli, ["setup", "spectra"])
        assert result.exit_code != 0
        assert "error: boom" in result.output
