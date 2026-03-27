"""Unit tests for volume assembly processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lst_tools.process import volume


class _FakeTecplotData:
    """Minimal Tecplot-like object used by assemble_volume tests."""

    def __init__(
        self,
        data: np.ndarray,
        variables: list[str],
        var_index: dict[str, int],
    ) -> None:
        self.data = data
        self.variables = variables
        self.var_index = var_index

    def field(self, name: str) -> np.ndarray:
        """Return 3-D field array (K, J, I) for a variable."""
        idx = self.var_index[name]
        return self.data[:, :, :, idx]


def _build_slice_data(x_vals: np.ndarray, f_vals: np.ndarray) -> _FakeTecplotData:
    """Build a synthetic 2-D slice with variables s, freq., beta, and alpi."""
    n_freq = len(f_vals)
    n_x = len(x_vals)
    n_var = 4

    # build (K=1, J=n_freq, I=n_x, n_var) array
    data = np.zeros((1, n_freq, n_x, n_var), dtype=float)

    for j, freq in enumerate(f_vals):
        for i, x_val in enumerate(x_vals):
            data[0, j, i, 0] = x_val
            data[0, j, i, 1] = freq
            data[0, j, i, 2] = 0.0
            data[0, j, i, 3] = 0.1 * x_val + 0.001 * freq

    variables = ["s", "freq.", "beta", "alpi"]
    var_index = {"s": 0, "freq": 1, "freq.": 1, "beta": 2, "alpi": 3}

    return _FakeTecplotData(data=data, variables=variables, var_index=var_index)


def test_parse_kc_value() -> None:
    """Parse kc directory name into float value."""
    assert volume._parse_kc_value("kc_0045pt25") == 45.25


def test_parse_kc_value_non_kc_prefix() -> None:
    """Parse case value from directory names with non-kc prefixes."""
    assert volume._parse_kc_value("slice_0045pt25") == 45.25


def test_assemble_volume_no_kc_dirs_returns_none(tmp_path: Path) -> None:
    """Return None when no kc_* directories exist."""
    out = volume.assemble_volume(tmp_path)
    assert out is None


def test_assemble_volume_no_completed_slices_returns_none(tmp_path: Path) -> None:
    """Return None when kc_* directories have no solution files."""
    (tmp_path / "kc_0001pt00").mkdir()
    (tmp_path / "kc_0002pt00").mkdir()

    out = volume.assemble_volume(tmp_path)
    assert out is None


def test_assemble_volume_custom_dir_pattern(tmp_path: Path, monkeypatch) -> None:
    """Allow custom directory discovery pattern via dir_pattern."""
    case1 = tmp_path / "slice_0001pt00"
    case2 = tmp_path / "slice_0002pt00"
    case1.mkdir()
    case2.mkdir()

    (case1 / "growth_rate_with_nfact_amps.dat").write_text("dummy", encoding="utf-8")
    (case2 / "growth_rate_with_nfact_amps.dat").write_text("dummy", encoding="utf-8")

    tp = _build_slice_data(
        x_vals=np.array([0.100, 0.200, 0.300]),
        f_vals=np.array([1000.0, 2000.0]),
    )

    monkeypatch.setattr(volume, "_NX_COMMON", 5)
    monkeypatch.setattr(volume, "_FREQ_SPACING", 1000.0)
    monkeypatch.setattr(volume, "read_tecplot_ascii", lambda _path: tp)

    captured: dict[str, object] = {}

    def _fake_write(
        path: Path,
        data: dict[str, np.ndarray],
        title: str,
        zone: str,
        progress_callback=None,
    ) -> None:
        captured["path"] = path
        captured["data"] = data
        captured["title"] = title
        captured["zone"] = zone

    monkeypatch.setattr(volume, "write_tecplot_ascii", _fake_write)

    out_path = volume.assemble_volume(tmp_path, dir_pattern="slice_*")

    assert out_path == tmp_path / "lst_vol.dat"
    assert captured["path"] == out_path


def test_assemble_volume_writes_3d_file_and_keeps_frequency_valid(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Assemble slices and verify freq field is populated on the output grid."""
    # build directories and solution markers
    kc1 = tmp_path / "kc_0001pt00"
    kc2 = tmp_path / "kc_0002pt00"
    kc1.mkdir()
    kc2.mkdir()

    (kc1 / "growth_rate_with_nfact_amps.dat").write_text("dummy", encoding="utf-8")
    (kc2 / "growth_rate_with_nfact_amps.dat").write_text("dummy", encoding="utf-8")

    # first slice has broader x-range; second starts later to force x out-of-bounds
    tp1 = _build_slice_data(
        x_vals=np.array([0.100, 0.200, 0.300]),
        f_vals=np.array([1000.0, 2000.0]),
    )
    tp2 = _build_slice_data(
        x_vals=np.array([0.200, 0.300, 0.400]),
        f_vals=np.array([1000.0, 2000.0]),
    )

    # reduce interpolation grid size for a fast test
    monkeypatch.setattr(volume, "_NX_COMMON", 5)
    monkeypatch.setattr(volume, "_FREQ_SPACING", 1000.0)

    # read function should return the synthetic slice by path
    def _fake_read(path: Path):
        if "kc_0001pt00" in str(path):
            return tp1
        return tp2

    monkeypatch.setattr(volume, "read_tecplot_ascii", _fake_read)

    captured: dict[str, object] = {}

    # capture writer payload
    def _fake_write(
        path: Path,
        data: dict[str, np.ndarray],
        title: str,
        zone: str,
        progress_callback=None,
    ) -> None:
        captured["path"] = path
        captured["data"] = data
        captured["title"] = title
        captured["zone"] = zone

    monkeypatch.setattr(volume, "write_tecplot_ascii", _fake_write)

    # execute
    out_path = volume.assemble_volume(tmp_path)

    # validate path and writer metadata
    assert out_path == tmp_path / "lst_vol.dat"
    assert captured["path"] == out_path
    assert captured["title"] == "lst_vol"
    assert captured["zone"] == "lst_vol"

    # validate shape convention: (K=n_kc, J=nf, I=nx)
    var_dict = captured["data"]
    freq_grid = var_dict["freq."]
    beta_grid = var_dict["beta"]
    assert freq_grid.shape == (2, 2, 5)

    # freq values should be valid and never use fill value
    assert not np.any(freq_grid == volume._FILL_VALUE)

    # beta should be filled with parsed kc values for each K slice
    assert np.allclose(beta_grid[0, :, :], 1.0)
    assert np.allclose(beta_grid[1, :, :], 2.0)
