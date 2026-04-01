"""Tests for the typer-based lst-tools CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from lst_tools.cli.main import cli

runner = CliRunner()


class TestCLIApp:
    """Test the top-level typer app."""

    def test_help(self):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "lst-tools" in result.output

    def test_version(self):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "lst-tools" in result.output

    def test_all_subcommands_registered(self):
        """All expected top-level commands appear in --help."""
        result = runner.invoke(cli, ["--help"])
        for cmd in [
            "hpc",
            "init",
            "lastrac",
            "setup",
            "process",
            "visualize",
        ]:
            assert cmd in result.output, f"missing subcommand: {cmd}"

    def test_setup_subcommands_registered(self):
        """All expected setup subcommands appear in setup --help."""
        result = runner.invoke(cli, ["setup", "--help"])
        for cmd in ["parsing", "tracking", "spectra"]:
            assert cmd in result.output, f"missing setup subcommand: {cmd}"

    def test_process_subcommands_registered(self):
        """All expected process subcommands appear in process --help."""
        result = runner.invoke(cli, ["process", "--help"])
        for cmd in ["tracking", "spectra"]:
            assert cmd in result.output, f"missing process subcommand: {cmd}"

    def test_subcommand_help(self):
        """Each top-level and nested subcommand responds to --help."""
        # top-level commands
        for cmd in ["hpc", "init", "lastrac", "visualize"]:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"{cmd} --help failed"

        # setup subcommands
        for cmd in ["parsing", "tracking", "spectra"]:
            result = runner.invoke(cli, ["setup", cmd, "--help"])
            assert result.exit_code == 0, f"setup {cmd} --help failed"

        # process subcommands
        for cmd in ["tracking", "spectra"]:
            result = runner.invoke(cli, ["process", cmd, "--help"])
            assert result.exit_code == 0, f"process {cmd} --help failed"

    def test_convert_removed(self):
        """Convert subcommand is no longer registered."""
        result = runner.invoke(cli, ["convert", "--help"])
        assert result.exit_code != 0
