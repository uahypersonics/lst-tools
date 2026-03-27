"""Tests for lst_tools.hpc.scripts script_build wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from lst_tools.hpc._resolve import ResolvedJob
from lst_tools.hpc.scripts import script_build


def _job(**kwargs) -> ResolvedJob:
    """Build a minimal resolved job object for script tests."""
    defaults = dict(
        nodes=1,
        ntasks_per_node=4,
        time=1.0,
        partition="standard",
        account="acct",
        scheduler="slurm",
        launcher="mpirun",
        hostname="generic",
        fname_run_script="run.slurm.generic",
    )
    defaults.update(kwargs)
    return ResolvedJob(**defaults)


def test_script_build_writes_script_file_and_sets_permissions(tmp_path: Path) -> None:
    """Render script text, write it to expected file, and chmod executable."""
    cfg = _job(fname_run_script="run.slurm.test")

    with patch("lst_tools.hpc.scripts.render", return_value="# test script\n") as mock_render:
        out = script_build(
            cfg,
            tmp_path,
            args=["lst_input.dat", ">run.log"],
            lst_exe="lst.x",
            extra_env={"OMP_NUM_THREADS": "1"},
        )

    assert out == tmp_path / "run.slurm.test"
    assert out.read_text(encoding="utf-8") == "# test script\n"
    assert out.stat().st_mode & 0o111

    mock_render.assert_called_once_with(
        cfg,
        lst_exe="lst.x",
        args=["lst_input.dat", ">run.log"],
        extra_env={"OMP_NUM_THREADS": "1"},
    )


def test_script_build_uses_default_executable_when_none(tmp_path: Path) -> None:
    """Fallback to lst.x when lst_exe is not provided."""
    cfg = _job(fname_run_script="run.slurm.default")

    with patch("lst_tools.hpc.scripts.render", return_value="echo ok\n") as mock_render:
        out = script_build(cfg, tmp_path, args=["lst_input.dat"], lst_exe=None)

    assert out == tmp_path / "run.slurm.default"
    mock_render.assert_called_once_with(
        cfg,
        lst_exe="lst.x",
        args=["lst_input.dat"],
        extra_env=None,
    )


def test_script_build_raises_for_unknown_scheduler_config(tmp_path: Path) -> None:
    """Reject unresolved scheduler configs with missing run script filename."""
    cfg = _job(scheduler="unknown", fname_run_script=None)

    with pytest.raises(ValueError, match="scheduler is unknown"):
        script_build(cfg, tmp_path)
