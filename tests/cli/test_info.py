"""Tests for the info CLI handler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
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
        "i_loc": 1,
        "s": 0.1,
        "n_eta": 3,
        "re1": 1.0e6,
        "lref": 0.5,
        "stat_temp": 250.0,
        "stat_uvel": 1200.0,
        "stat_dens": 0.2,
        "kappa": 0.01,
        "rloc": 0.02,
        "drdx": 0.03,
    }
    sh1 = {
        "i_loc": 2,
        "s": 0.4,
        "n_eta": 4,
        "re1": 2.0e6,
        "lref": 0.75,
        "stat_temp": 275.0,
        "stat_uvel": 1300.0,
        "stat_dens": 0.3,
        "kappa": 0.03,
        "rloc": 0.04,
        "drdx": 0.05,
    }

    reader = MagicMock()
    reader.read_header.return_value = header
    reader.read_station_header.side_effect = [sh0, sh1]
    reader.read_station_vector.side_effect = [
        np.array([0.0, 0.5, 1.0]),
        np.array([0.0, 0.4, 0.8, 1.2]),
    ]

    # execute
    with patch("lst_tools.cli.cmd_info.LastracReader", return_value=reader):
        cmd_info(fpath)

    # validate
    captured = capsys.readouterr()
    assert "station 1" in captured.out
    assert "station 2" in captured.out
    assert "reference quantities" not in captured.out
    assert "n_station:  2" in captured.out
    assert "n_eta:      3" in captured.out
    assert "n_eta:      4" in captured.out
    assert "eta_max:    1.000000e+00" in captured.out
    assert "eta_max:    1.200000e+00" in captured.out
    assert "re1:        1.000000e+06" in captured.out
    assert "re1:        2.000000e+06" in captured.out
    reader.skip_records.assert_called_with(5)
    assert reader.skip_records.call_count == 2
    assert reader.read_station_vector.call_count == 2
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


def test_cmd_info_writes_profiles_tecplot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Export station vectors to Tecplot ASCII when profiles_out is requested."""
    # build
    fpath = tmp_path / "meanflow.bin"
    fpath.write_bytes(b"dummy")
    out_path = tmp_path / "profiles.dat"

    header = {
        "title": "test-case",
        "n_station": 1,
        "igas": 1,
        "iunit": 1,
        "Pr": 0.72,
        "stat_pres": 101325.0,
        "nsp": 1,
    }
    sh0 = {
        "i_loc": 1,
        "s": 0.1,
        "n_eta": 2,
        "re1": 1.0e6,
        "lref": 0.5,
        "stat_temp": 250.0,
        "stat_uvel": 1200.0,
        "stat_dens": 0.2,
        "kappa": 0.01,
        "rloc": 0.02,
        "drdx": 0.03,
    }

    reader = MagicMock()
    reader.read_header.return_value = header
    reader.read_station_header.side_effect = [sh0]
    reader.read_station_vector.side_effect = [
        np.array([0.0, 1.0]),
        np.array([0.1, 0.2]),
        np.array([0.01, 0.02]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.1]),
        np.array([2.0, 2.1]),
    ]

    # execute
    with patch("lst_tools.cli.cmd_info.LastracReader", return_value=reader):
        cmd_info(fpath, profiles_out=out_path)

    # validate terminal output
    captured = capsys.readouterr()
    assert "profiles_out:" in captured.out

    # validate file output
    text = out_path.read_text(encoding="utf-8")
    assert 'TITLE = "meanflow_profiles"' in text
    assert 'VARIABLES = "s" "eta" "u" "v" "w" "T" "p"' in text
    assert 'ZONE T="station_0000_iloc_0001_s_1.000000e-01", I=2, DATAPACKING=POINT' in text
    assert "1.00000000e-01 0.00000000e+00 1.00000000e-01" in text

    # validate read path behavior
    reader.skip_records.assert_not_called()
    assert reader.read_station_vector.call_count == 6
