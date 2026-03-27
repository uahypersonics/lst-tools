"""Tests for the typer-based tracking process subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()


class TestTrackingProcessCLI:
    """Test suite for process tracking CLI module."""

    @patch("lst_tools.cli.cmd_tracking_process.tracking_process")
    @patch("lst_tools.cli.cmd_tracking_process.read_config")
    def test_default_runs_both_steps(self, mock_read_config, mock_tracking_process):
        mock_read_config.return_value = Config()

        result = runner.invoke(cli, ["process", "tracking"])

        assert result.exit_code == 0
        mock_read_config.assert_called_once_with(path=None)
        mock_tracking_process.assert_called_once_with(
            cfg=mock_read_config.return_value,
            do_maxima=True,
            do_volume=True,
            kc_dirs=None,
            interpolate=None,
        )

    @patch("lst_tools.cli.cmd_tracking_process.tracking_process")
    @patch("lst_tools.cli.cmd_tracking_process.read_config")
    def test_maxima_only_flag(self, mock_read_config, mock_tracking_process):
        mock_read_config.return_value = Config()

        result = runner.invoke(cli, ["process", "tracking", "--maxima"])

        assert result.exit_code == 0
        kwargs = mock_tracking_process.call_args.kwargs
        assert kwargs["do_maxima"] is True
        assert kwargs["do_volume"] is False

    @patch("lst_tools.cli.cmd_tracking_process.tracking_process")
    @patch("lst_tools.cli.cmd_tracking_process.read_config")
    def test_volume_only_flag(self, mock_read_config, mock_tracking_process):
        mock_read_config.return_value = Config()

        result = runner.invoke(cli, ["process", "tracking", "--volume"])

        assert result.exit_code == 0
        kwargs = mock_tracking_process.call_args.kwargs
        assert kwargs["do_maxima"] is False
        assert kwargs["do_volume"] is True

    @patch("lst_tools.cli.cmd_tracking_process.tracking_process")
    @patch("lst_tools.cli.cmd_tracking_process.read_config")
    def test_dir_disables_volume(self, mock_read_config, mock_tracking_process, tmp_path: Path):
        mock_read_config.return_value = Config()
        d1 = tmp_path / "kc_0000pt00"
        d1.mkdir()

        result = runner.invoke(cli, ["process", "tracking", "--dir", str(d1)])

        assert result.exit_code == 0
        kwargs = mock_tracking_process.call_args.kwargs
        assert kwargs["kc_dirs"] == [d1]
        assert kwargs["do_volume"] is False
        assert kwargs["do_maxima"] is True

    @patch("lst_tools.cli.cmd_tracking_process.tracking_process")
    @patch("lst_tools.cli.cmd_tracking_process.read_config")
    def test_interpolate_switches(self, mock_read_config, mock_tracking_process):
        mock_read_config.return_value = Config()

        result_yes = runner.invoke(cli, ["process", "tracking", "--interpolate"])
        assert result_yes.exit_code == 0
        assert mock_tracking_process.call_args.kwargs["interpolate"] is True

        mock_tracking_process.reset_mock()

        result_no = runner.invoke(cli, ["process", "tracking", "--no-interpolate"])
        assert result_no.exit_code == 0
        assert mock_tracking_process.call_args.kwargs["interpolate"] is False

    @patch("lst_tools.cli.cmd_tracking_process.read_config", side_effect=Exception("boom"))
    def test_error_path_returns_nonzero(self, mock_read_config):
        result = runner.invoke(cli, ["process", "tracking"])
        assert result.exit_code != 0
        assert "error: boom" in result.output
