"""Tests for the info CLI handler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from lst_tools.cli.cmd_info import cmd_info


def test_cmd_info_missing_file_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit with code 1 when target file does not exist."""
    # build
    missing = tmp_path / "missing.bin"

    # execute / validate
    with pytest.raises(typer.Exit) as exc:
        cmd_info(missing)

    assert exc.value.exit_code == 1
    captured = capsys.readouterr()
    assert "not found" in captured.err


def test_cmd_info_success_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Read station headers and print summary output."""
    # build
    fpath = tmp_path / "meanflow.bin"
    fpath.write_bytes(b"dummy")

    header = {
        "title": "test-case",
        "n_station": 2,
        "igas": 1,
        "iunit": 1,
        "Pr": 0.72,
        "stat_pres": 101325.0,
        "nsp": 1,
    }
    sh0 = {
        "s": 0.1,
        "n_eta": 5,
        "re1": 1.0e6,
        "lref": 0.5,
        "stat_temp": 250.0,
        "stat_uvel": 1200.0,
        "stat_dens": 0.2,
        "kappa": 0.01,
    }
    sh1 = {
        "s": 0.4,
        "n_eta": 5,
        "re1": 1.0e6,
        "lref": 0.5,
        "stat_temp": 250.0,
        "stat_uvel": 1200.0,
        "stat_dens": 0.2,
        "kappa": 0.03,
    }

    reader = MagicMock()
    reader.read_header.return_value = header
    reader.read_station_header.side_effect = [sh0, sh1]

    # execute
    with patch("lst_tools.cli.cmd_info.LastracReader", return_value=reader):
        cmd_info(fpath)

    # validate
    captured = capsys.readouterr()
    assert "station summary" in captured.out
    assert "reference quantities" in captured.out
    assert "n_station:  2" in captured.out
    reader.skip_records.assert_called_with(6)
    assert reader.skip_records.call_count == 2
    reader.close.assert_called_once()


def test_cmd_info_reader_error_exits(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Exit with code 1 when reading the file raises an exception."""
    # build
    fpath = tmp_path / "meanflow.bin"
    fpath.write_bytes(b"dummy")

    # execute / validate
    with patch("lst_tools.cli.cmd_info.LastracReader", side_effect=RuntimeError("boom")):
        with pytest.raises(typer.Exit) as exc:
            cmd_info(fpath)

    assert exc.value.exit_code == 1
    captured = capsys.readouterr()
    assert "error: boom" in captured.err
