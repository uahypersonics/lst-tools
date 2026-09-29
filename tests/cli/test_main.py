"""Tests for the typer-based lst-tools CLI entry point."""

from __future__ import annotations

from typer.testing import CliRunner

from lst_tools.cli.app import cli
from lst_tools.status import read_run_log

runner = CliRunner()


def test_read_convergence_issues_reports_x_from_matching_station(tmp_path):
    log_path = tmp_path / "run.log"
    log_path.write_text(
        " Initialization Stage, Station LST#:   267/  273 (x =         0.0084568011 )\n"
        " 2 Method did not converge for proc. 0 , freq. 20000.000000000000 , x-station 267\n"
        " 2 Method did not converge for proc. 0 , freq. 25000.000000000000 , x-station 268\n"
        " Total Time 0.27430E+04sec\n",
        encoding="utf-8",
    )

    run_status = read_run_log(log_path)

    assert run_status.finished
    assert [
        (issue.station, issue.frequency, issue.x) for issue in run_status.issues
    ] == [
        (267, 20000.0, 0.0084568011),
        (268, 25000.0, None),
    ]


def test_status_tracking_reports_finished_and_incomplete_cases(tmp_path):
    finished_case = tmp_path / "kc_0006pt00"
    finished_case.mkdir()
    (finished_case / "run.log").write_text(
        " Initialization Stage, Station LST#:   267/  273 (x =         0.0084568011 )\n"
        " 2 Method did not converge for proc. 0 , freq. 20000.0 , x-station 267\n"
        " Total Time 0.27430E+04sec\n",
        encoding="utf-8",
    )
    incomplete_case = tmp_path / "kc_0010pt00"
    incomplete_case.mkdir()
    (incomplete_case / "run.log").write_text(
        " Initialization Stage\n", encoding="utf-8"
    )
    (tmp_path / "kc_0020pt00").mkdir()

    result = runner.invoke(cli, ["status", "tracking", str(tmp_path)])
    full_result = runner.invoke(
        cli, ["status", "tracking", str(tmp_path), "--all-issues"]
    )

    assert result.exit_code == 0
    assert "[finished] kc_0006pt00\n" in result.output
    assert "station 267:" not in result.output
    assert "station 267:" not in full_result.output
    assert "[incomplete/unknown] kc_0010pt00" in result.output
    assert "[no log] kc_0020pt00" in result.output
    assert "3 cases, 1 finished, 1 incomplete/unknown, 1 without log" in result.output
    assert "0 unfinished-case convergence issue(s)" in result.output


def test_status_tracking_limits_issue_details_until_requested(tmp_path):
    case_dir = tmp_path / "kc_0006pt00"
    case_dir.mkdir()
    lines = []
    for station in range(1, 13):
        lines.append(
            f" Initialization Stage, Station LST#: {station:5d}/   12 (x = {station / 100:.10f} )\n"
        )
        lines.append(
            f" 2 Method did not converge for proc. 0 , freq. 20000.0 , x-station {station}\n"
        )
    (case_dir / "run.log").write_text("".join(lines), encoding="utf-8")

    result = runner.invoke(cli, ["status", "tracking", str(tmp_path)])
    full_result = runner.invoke(
        cli, ["status", "tracking", str(tmp_path), "--all-issues"]
    )

    assert "station 10: x=0.1, f=20000 Hz" in result.output
    assert "station 11:" not in result.output
    assert "--all-issues" in result.output
    assert "station 12: x=0.12, f=20000 Hz" in full_result.output


def test_status_tracking_flags_nonconvergence_without_finish(tmp_path):
    case_dir = tmp_path / "kc_0006pt00"
    case_dir.mkdir()
    (case_dir / "run.log").write_text(
        " Initialization Stage, Station LST#:   267/  273 (x =         0.0084568011 )\n"
        " 2 Method did not converge for proc. 0 , freq. 20000.0 , x-station 267\n",
        encoding="utf-8",
    )

    result = runner.invoke(cli, ["status", "tracking", str(tmp_path)])

    assert result.exit_code == 0
    assert "[stopped after nonconvergence] kc_0006pt00" in result.output
    assert "station 267: x=0.0084568011, f=20000 Hz" in result.output


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
            "status",
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
