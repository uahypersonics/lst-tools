"""Unit tests for maxima extraction and ridge tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lst_tools.process import maxima


class _FakeTecplotData:
    """Minimal Tecplot-like object for maxima tests."""

    def __init__(self, data: np.ndarray, variables: list[str], var_index: dict[str, int]) -> None:
        self.data = data
        self.variables = variables
        self.var_index = var_index

    def field(self, name: str) -> np.ndarray:
        idx = self.var_index[name]
        return self.data[:, :, :, idx]


def _build_tracking_tp() -> _FakeTecplotData:
    """Build synthetic tracking data with freq/alpi/nfac fields."""
    nf = 5
    nx = 3
    nvar = 4
    data = np.zeros((1, nf, nx, nvar), dtype=float)

    freq = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])

    alpi_cols = [
        np.array([0.0, 2.0, 0.0, 2.0, 0.0]),
        np.array([0.0, 2.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 2.0, 0.0, 0.0]),
    ]

    nfac_cols = [
        np.array([0.0, 1.5, 0.0, 1.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.8, 0.0]),
        np.array([0.0, 0.7, 0.0, 0.6, 0.0]),
    ]

    for i in range(nx):
        data[0, :, i, 0] = freq
        data[0, :, i, 1] = alpi_cols[i]
        data[0, :, i, 2] = nfac_cols[i]
        data[0, :, i, 3] = 0.1 * (i + 1)

    variables = ["freq", "alpi", "nfac", "s"]
    var_index = {"freq": 0, "alpi": 1, "nfac": 2, "s": 3}
    return _FakeTecplotData(data, variables, var_index)


def test_find_peaks_filters_non_positive() -> None:
    """Keep only local maxima with positive values."""
    values = np.array([-1.0, 0.0, 2.0, 0.0, 1.0, 0.0])
    peaks = maxima._find_peaks(values)
    assert np.array_equal(peaks, np.array([2, 4]))


def test_find_peaks_parabolic_interpolation_returns_fractional_index() -> None:
    """Refine integer peak location with parabolic fit."""
    values = np.array([0.0, 1.0, 2.0, 1.5, 0.0])
    int_peaks, frac_peaks = maxima._find_peaks_parabolic_interpolation(values)

    assert np.array_equal(int_peaks, np.array([2]))
    assert 2.0 < frac_peaks[0] < 2.5


def test_track_ridges_gate_rejects_large_frequency_jump() -> None:
    """Create a new ridge when assignment exceeds gate tolerance."""
    freq_2d = np.tile(np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])[:, None], (1, 3))

    target_2d = np.column_stack(
        [
            np.array([0.0, 2.0, 0.0, 2.0, 0.0]),
            np.array([0.0, 2.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 2.0, 0.0, 0.0]),
        ]
    )

    ridges = maxima._track_ridges(target_2d, freq_2d, gate_tol=0.1, interpolate=False)

    assert len(ridges) == 3
    assert any(len(r.indices) == 2 for r in ridges)
    assert any(len(r.indices) == 1 for r in ridges)


def test_track_ridges_with_interpolation_stores_float_indices() -> None:
    """Store sub-grid peak positions when interpolation is enabled."""
    freq_2d = np.tile(np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])[:, None], (1, 1))
    target_2d = np.array([[0.0], [1.0], [2.0], [1.5], [0.0]])

    ridges = maxima._track_ridges(target_2d, freq_2d, interpolate=True)

    assert len(ridges) == 1
    assert isinstance(ridges[0].indices[0][1], float)


def test_write_ridge_files_interpolates_fractional_rows(tmp_path: Path, monkeypatch) -> None:
    """Interpolate ridge row data for fractional frequency indices."""
    data_2d = np.array(
        [
            [[10.0, 100.0], [11.0, 101.0]],
            [[20.0, 200.0], [21.0, 201.0]],
            [[30.0, 300.0], [31.0, 301.0]],
        ]
    )
    ridges = [maxima.Ridge(indices=[(0, 0.5), (1, 2)])]

    captured = {}

    def _fake_write(path: Path, data: dict[str, np.ndarray], title: str, zone: str) -> None:
        captured["path"] = path
        captured["data"] = data
        captured["title"] = title
        captured["zone"] = zone

    monkeypatch.setattr(maxima, "write_tecplot_ascii", _fake_write)

    out = maxima._write_ridge_files(
        ridges=ridges,
        data_2d=data_2d,
        prefix="alpi_max_mode",
        variables=["a", "b"],
        dir_name=tmp_path,
        min_valid=1,
    )

    assert len(out) == 1
    assert out[0] == tmp_path / "alpi_max_mode_001.dat"
    assert np.allclose(captured["data"]["a"], np.array([15.0, 31.0]))
    assert np.allclose(captured["data"]["b"], np.array([150.0, 301.0]))


def test_extract_maxima_missing_solution_returns_empty(tmp_path: Path) -> None:
    """Return empty list when solution file does not exist."""
    out = maxima.extract_maxima(tmp_path)
    assert out == []


def test_extract_maxima_runs_two_ridge_passes(tmp_path: Path, monkeypatch) -> None:
    """Run both alpi and nfac ridge extraction passes and return outputs."""
    sol = tmp_path / "growth_rate_with_nfact_amps.dat"
    sol.write_text("dummy", encoding="utf-8")

    tp = _build_tracking_tp()
    monkeypatch.setattr(maxima, "read_tecplot_ascii", lambda _: tp)

    ridge_calls = {"n": 0}

    def _fake_track(*args, **kwargs):
        ridge_calls["n"] += 1
        return [maxima.Ridge(indices=[(0, 1), (1, 1), (2, 2)])]

    monkeypatch.setattr(maxima, "_track_ridges", _fake_track)

    write_calls = []

    def _fake_write_files(ridges, data_2d, prefix, variables, dir_name, min_valid):
        write_calls.append(prefix)
        return [dir_name / f"{prefix}_001.dat"]

    monkeypatch.setattr(maxima, "_write_ridge_files", _fake_write_files)

    out = maxima.extract_maxima(tmp_path, interpolate=True, min_valid=1)

    assert ridge_calls["n"] == 2
    assert write_calls == ["alpi_max_mode", "nfac_max_mode"]
    assert len(out) == 2
