"""Additional orchestration tests for tracking setup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lst_tools.config.schema import Config
from lst_tools.setup import tracking as tracking_mod


class _FakeTecplotData:
    """Minimal Tecplot-like object exposing the field API."""

    def __init__(self) -> None:
        self._beta = np.array([[[10.0]], [[20.0]]])
        self._s = np.array([[[0.1, 0.2, 0.3]]])

    def field(self, name: str) -> np.ndarray:
        if name == "beta":
            return self._beta
        if name == "s":
            return self._s
        raise KeyError(name)


def test_tracking_setup_runs_orchestration_and_writes_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run tracking setup orchestration with patched internals."""
    monkeypatch.chdir(tmp_path)

    cfg = Config()
    cfg.lst.params.tracking_dir = 0
    cfg.lst.params.x_s = 0.15
    cfg.lst.params.x_e = 0.25
    cfg.lst.io.baseflow_input = "meanflow.bin"

    (tmp_path / "meanflow.bin").write_bytes(b"dummy")

    monkeypatch.setattr(tracking_mod, "resolve_config", lambda _: cfg)
    monkeypatch.setattr(tracking_mod, "_read_parsing_solution", lambda _: _FakeTecplotData())
    monkeypatch.setattr(tracking_mod, "read_baseflow_stations", lambda _: np.array([0.1, 0.2, 0.3]))
    monkeypatch.setattr(tracking_mod, "_resolve_beta_values", lambda *_: np.array([10.0, 20.0]))

    created_dirs: list[str] = []

    def _fake_setup_case_directory(betr_loc: float, cfg_obj):
        name = f"kc_{betr_loc:07.2f}".replace(".", "pt")
        created_dirs.append(name)
        return name, "lst.x"

    monkeypatch.setattr(tracking_mod, "_setup_case_directory", _fake_setup_case_directory)
    monkeypatch.setattr(
        tracking_mod,
        "_find_initial_guess",
        lambda *args, **kwargs: {"idx_x": 1, "idx_f": 0, "alpi": 1.0, "alpr": 2.0, "freq": 1000.0},
    )

    fake_hpc = SimpleNamespace(scheduler="slurm", fname_run_script="run.slurm")
    monkeypatch.setattr(tracking_mod, "_build_and_write_case", lambda *args, **kwargs: fake_hpc)

    launcher_calls: list[tuple[list[str], str | None]] = []

    def _fake_launcher(dirs, *, script_name, submit_cmd, fname_run_script):
        launcher_calls.append((dirs, submit_cmd))
        return Path(script_name)

    monkeypatch.setattr(tracking_mod, "write_launcher_script", _fake_launcher)

    out = tracking_mod.tracking_setup(cfg=cfg)

    assert out == Path("run_jobs.sh")
    assert created_dirs == ["kc_0010pt00", "kc_0020pt00"]
    assert launcher_calls == [(["kc_0010pt00", "kc_0020pt00"], "sbatch")]


def test_tracking_setup_raises_when_no_betas(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise RuntimeError when no beta values remain after filtering."""
    cfg = Config()
    monkeypatch.setattr(tracking_mod, "resolve_config", lambda _: cfg)
    monkeypatch.setattr(tracking_mod, "_read_parsing_solution", lambda _: _FakeTecplotData())
    monkeypatch.setattr(tracking_mod, "_resolve_beta_values", lambda *_: np.array([]))

    with pytest.raises(RuntimeError):
        tracking_mod.tracking_setup(cfg=cfg)


def test_read_parsing_solution_missing_default_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise FileNotFoundError when default parsing file does not exist."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        tracking_mod._read_parsing_solution(None)
