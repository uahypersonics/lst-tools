"""Tests for the typer-based lst-tools lastrac subcommand."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()


class TestLastracHelp:
    def test_help_shows_options(self):
        result = runner.invoke(cli, ["lastrac", "--help"])
        assert result.exit_code == 0
        assert "--cfg" in result.output


class TestLastracCommand:
    @patch("lst_tools.cli.cmd_lastrac.read_config")
    @patch("lst_tools.cli.cmd_lastrac._load_with_cfd_io")
    @patch("lst_tools.cli.cmd_lastrac.convert_meanflow")
    def test_run_with_defaults(
        self,
        mock_convert,
        mock_load,
        mock_read_config,
        tmp_path,
    ):
        """Test lastrac with default values from config."""
        sample_config = Config.from_dict({
            "input_file": "test_input.hdf5",
            "lst": {"io": {"baseflow_input": "test_meanflow.bin"}},
        })
        mock_read_config.return_value = sample_config
        mock_load.return_value = (MagicMock(), MagicMock(), {})

        # Create the expected input file so Path.exists() succeeds
        hdf5_file = tmp_path / "test_input.hdf5"
        hdf5_file.touch()

        import os
        prev_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner.invoke(cli, ["lastrac"])
        finally:
            os.chdir(prev_dir)

        mock_load.assert_called_once()
        mock_convert.assert_called_once()

    @patch("lst_tools.cli.cmd_lastrac.read_config")
    @patch("lst_tools.cli.cmd_lastrac._load_with_cfd_io")
    @patch("lst_tools.cli.cmd_lastrac.convert_meanflow")
    def test_run_with_cli_overrides(
        self,
        mock_convert,
        mock_load,
        mock_read_config,
        tmp_path,
    ):
        """Test lastrac with CLI overrides."""
        mock_read_config.return_value = Config()
        mock_load.return_value = (MagicMock(), MagicMock(), {})

        inp_file = tmp_path / "base_flow.hdf5"
        inp_file.touch()
        cfg_file = tmp_path / "custom_config.toml"
        cfg_file.touch()

        import os
        prev_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner.invoke(
                cli,
                ["lastrac", "--cfg", str(cfg_file)],
            )
        finally:
            os.chdir(prev_dir)

        mock_convert.assert_called_once()

    def test_run_missing_input_file(self, tmp_path):
        """Test lastrac with missing input file."""
        import os
        prev_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(cli, ["lastrac"])
        finally:
            os.chdir(prev_dir)
        assert result.exit_code != 0

    @patch("lst_tools.cli.cmd_lastrac.read_config")
    @patch("lst_tools.cli.cmd_lastrac._load_with_cfd_io")
    @patch("lst_tools.cli.cmd_lastrac.convert_meanflow")
    def test_run_with_fallbacks(
        self,
        mock_convert,
        mock_load,
        mock_read_config,
        tmp_path,
    ):
        """Test lastrac with fallback values when config is empty."""
        mock_read_config.return_value = Config()
        mock_load.return_value = (MagicMock(), MagicMock(), {})

        # Create the default fallback file
        fallback = tmp_path / "base_flow.hdf5"
        fallback.touch()

        import os
        prev_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            runner.invoke(cli, ["lastrac"])
        finally:
            os.chdir(prev_dir)

        mock_convert.assert_called_once()

    @patch("lst_tools.cli.cmd_lastrac.read_config", side_effect=Exception("Config error"))
    def test_run_config_exception_propagates(self, mock_read_config, tmp_path):
        """Test lastrac propagates config loading exceptions."""
        import os
        prev_dir = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(cli, ["lastrac"])
        finally:
            os.chdir(prev_dir)
        assert result.exit_code != 0
