"""Tests for the typer-based lst-tools spectra process subcommand."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import ANY, patch

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class TestSpectraProcessCLI:
    """Test suite for spectra process CLI module."""

    def test_help_shows_options(self):
        result = runner.invoke(cli, ["process", "spectra", "--help"])
        assert result.exit_code == 0
        plain = _ANSI_RE.sub("", result.output)
        assert "--cfg" in plain

    @patch("lst_tools.cli.cmd_spectra_process.spectra_process")
    @patch("lst_tools.cli.cmd_spectra_process.read_config")
    def test_run_success(self, mock_read_config, mock_spectra_process):
        mock_cfg = Config()
        mock_read_config.return_value = mock_cfg
        mock_spectra_process.return_value = Path("spectra_results.dat")

        result = runner.invoke(cli, ["process", "spectra"])

        assert result.exit_code == 0
        assert "spectra post-processing complete: spectra_results.dat" in result.output
        mock_read_config.assert_called_once_with(path=None)
        mock_spectra_process.assert_called_once_with(
            cfg=mock_cfg,
            reporter=ANY,
            do_animate=True,
            do_branches=True,
            do_classify=False,
        )

    @patch("lst_tools.cli.cmd_spectra_process.spectra_process", side_effect=NotImplementedError("todo"))
    @patch("lst_tools.cli.cmd_spectra_process.read_config")
    def test_not_implemented_error(self, mock_read_config, mock_spectra_process):
        mock_read_config.return_value = Config()

        result = runner.invoke(cli, ["process", "spectra"])

        assert result.exit_code != 0
        assert "error: todo" in result.output

    @patch("lst_tools.cli.cmd_spectra_process.spectra_process", side_effect=RuntimeError("bad"))
    @patch("lst_tools.cli.cmd_spectra_process.read_config")
    def test_generic_error_path(self, mock_read_config, mock_spectra_process):
        mock_read_config.return_value = Config()

        result = runner.invoke(cli, ["process", "spectra"])

        assert result.exit_code != 0
        assert "error: bad" in result.output
