"""Unit tests for spectra processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from lst_tools.process.spectra import spectra_process


def test_spectra_process_no_case_dirs_returns_current_path(tmp_path: Path, monkeypatch) -> None:
    """Return current path marker when no matching case directories exist."""
    monkeypatch.chdir(tmp_path)

    out = spectra_process(cfg=None)

    assert out == Path(".")


def test_spectra_process_case_dirs_raise_not_implemented(tmp_path: Path, monkeypatch) -> None:
    """Raise NotImplementedError once case scanning and loading steps complete."""
    monkeypatch.chdir(tmp_path)

    case_dir = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir.mkdir()
    eig_file = case_dir / "Eigenvalues_case.dat"
    eig_file.write_text("dummy", encoding="utf-8")

    monkeypatch.setattr(
        "lst_tools.process.spectra.read_tecplot_ascii",
        lambda _: {"mock": "tecplot"},
    )

    with pytest.raises(NotImplementedError):
        spectra_process(cfg=None)


def test_spectra_process_handles_missing_eigenvalue_files(tmp_path: Path, monkeypatch) -> None:
    """Continue past missing spectra files and still reach not-implemented branch."""
    monkeypatch.chdir(tmp_path)

    # matching directory but no Eigenvalues_* file
    (tmp_path / "x_00pt20_m_f_0008pt00_khz_beta_neg0100pt00").mkdir()

    with pytest.raises(NotImplementedError):
        spectra_process(cfg=None)
