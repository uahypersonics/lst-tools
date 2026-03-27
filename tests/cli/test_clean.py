"""Tests for clean command helpers and subcommands."""

from __future__ import annotations

from pathlib import Path

import pytest
import typer

from lst_tools.cli.cmd_clean import (
    _collect_targets,
    _confirm_and_remove,
    _remove,
    cmd_clean_parsing,
    cmd_clean_spectra,
    cmd_clean_tracking,
)


def test_collect_targets_deduplicates_and_sorts(tmp_path: Path) -> None:
    """Collect targets from multiple patterns without duplicates."""
    # build
    a = tmp_path / "run.slurm.host"
    b = tmp_path / "Frequency01"
    a.write_text("a", encoding="utf-8")
    b.write_text("b", encoding="utf-8")

    # execute
    targets = _collect_targets(tmp_path, ["run.*.*", "Frequency*", "run.*.*"])

    # validate
    assert targets == [a, b]


def test_remove_handles_files_and_dirs(tmp_path: Path) -> None:
    """Remove a mix of files and directories."""
    # build
    fpath = tmp_path / "file.dat"
    dpath = tmp_path / "x_0001"
    fpath.write_text("x", encoding="utf-8")
    dpath.mkdir()
    (dpath / "nested.txt").write_text("y", encoding="utf-8")

    # execute
    removed = _remove([fpath, dpath])

    # validate
    assert removed == 2
    assert not fpath.exists()
    assert not dpath.exists()


def test_confirm_and_remove_no_targets_exits(capsys: pytest.CaptureFixture[str]) -> None:
    """Print no-op message and exit when there is nothing to remove."""
    # execute / validate
    with pytest.raises(typer.Exit):
        _confirm_and_remove([], force=True)

    captured = capsys.readouterr()
    assert "nothing to clean" in captured.out


def test_cmd_clean_parsing_force_removes_targets(tmp_path: Path) -> None:
    """Remove parsing artifacts without prompting when force=True."""
    # build
    (tmp_path / "lst_input.dat").write_text("deck", encoding="utf-8")
    (tmp_path / "run.slurm.host").write_text("run", encoding="utf-8")
    (tmp_path / "Frequency01").write_text("freq", encoding="utf-8")

    # execute
    cmd_clean_parsing(directory=tmp_path, force=True)

    # validate
    assert not (tmp_path / "lst_input.dat").exists()
    assert not (tmp_path / "run.slurm.host").exists()
    assert not (tmp_path / "Frequency01").exists()


def test_cmd_clean_tracking_default_directory_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Default tracking clean scans kc_* directories in cwd."""
    # build
    kc_dir = tmp_path / "kc_0000pt00"
    kc_dir.mkdir()
    (kc_dir / "fort.10").write_text("solver", encoding="utf-8")

    # execute
    monkeypatch.chdir(tmp_path)
    cmd_clean_tracking(directory=None, force=True)

    # validate
    assert not (kc_dir / "fort.10").exists()


def test_cmd_clean_spectra_removes_case_dirs_and_launcher(tmp_path: Path) -> None:
    """Remove x_* spectra case directories and run_cases launcher."""
    # build
    x_dir = tmp_path / "x_000000"
    x_dir.mkdir()
    (x_dir / "input.dat").write_text("input", encoding="utf-8")
    (tmp_path / "run_cases.sh").write_text("#!/bin/sh", encoding="utf-8")

    # execute
    cmd_clean_spectra(directory=tmp_path, force=True)

    # validate
    assert not x_dir.exists()
    assert not (tmp_path / "run_cases.sh").exists()
