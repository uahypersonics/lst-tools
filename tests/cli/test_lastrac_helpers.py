"""Unit tests for lastrac helper functions and debug path."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from lst_tools.cli.cmd_lastrac import _load_with_cfd_io, _to_2d
from lst_tools.cli.main import cli
from lst_tools.config.schema import Config

runner = CliRunner()


class _FakeStructuredGrid:
    """Minimal structured grid stand-in for cfd-io objects."""

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_to_2d_accepts_2d_array() -> None:
    """Return 2-D arrays unchanged."""
    arr = np.arange(6.0).reshape(2, 3)
    out = _to_2d(arr, "grid/x")
    assert np.array_equal(out, arr)


def test_to_2d_squeezes_singleton_3d() -> None:
    """Squeeze 3-D arrays when one axis is singleton."""
    arr = np.arange(6.0).reshape(2, 3, 1)
    out = _to_2d(arr, "grid/x")
    assert out.shape == (2, 3)


def test_to_2d_rejects_non_singleton_3d() -> None:
    """Reject full 3-D volumes for 2-D LST conversion."""
    arr = np.zeros((2, 3, 4))
    try:
        _to_2d(arr, "grid/x")
    except ValueError as exc:
        assert "single-plane" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-singleton 3-D array")


def test_load_with_cfd_io_transposes_when_needed(monkeypatch) -> None:
    """Transpose arrays when streamwise direction is not along axis 1."""
    x = np.array([[3.0, 2.0], [3.0, 2.0]])
    y = np.array([[0.0, 0.0], [1.0, 1.0]])
    z = np.ones_like(x)

    grid = _FakeStructuredGrid(x=x, y=y, z=z)
    flow = {"u": np.array([[10.0, 20.0], [30.0, 40.0]])}
    ds = SimpleNamespace(grid=grid, flow=flow, attrs={"case": "demo"})

    monkeypatch.setattr("lst_tools.cli.cmd_lastrac.StructuredGrid", _FakeStructuredGrid)
    monkeypatch.setattr("lst_tools.cli.cmd_lastrac.cfd_read_file", lambda _: ds)

    grid_out, flow_out, attrs_out = _load_with_cfd_io(Path("dummy.h5"))

    assert grid_out.shape == (2, 2)
    assert np.array_equal(grid_out.x, x.T)
    assert np.array_equal(grid_out.y, y.T)
    assert np.array_equal(grid_out.z, z.T)
    assert np.array_equal(flow_out.fields["u"], flow["u"].T)
    assert attrs_out == {"case": "demo"}


def test_load_with_cfd_io_rejects_non_structured_grid(monkeypatch) -> None:
    """Raise TypeError when cfd-io returns a non-structured grid."""
    ds = SimpleNamespace(grid=object(), flow={}, attrs={})
    monkeypatch.setattr("lst_tools.cli.cmd_lastrac.StructuredGrid", _FakeStructuredGrid)
    monkeypatch.setattr("lst_tools.cli.cmd_lastrac.cfd_read_file", lambda _: ds)

    try:
        _load_with_cfd_io(Path("dummy.h5"))
    except TypeError as exc:
        assert "structured grid" in str(exc)
    else:
        raise AssertionError("expected TypeError for non-structured grid")


@patch("lst_tools.cli.cmd_lastrac.convert_meanflow")
@patch("lst_tools.cli.cmd_lastrac._load_with_cfd_io")
@patch("lst_tools.cli.cmd_lastrac.read_config")
def test_lastrac_debug_mode_writes_debug_file(
    mock_read_config,
    mock_load_with_cfd_io,
    mock_convert,
    tmp_path: Path,
) -> None:
    """Write debug tecplot file when global --debug is enabled."""
    cfg = Config()
    cfg.input_file = "base_flow.hdf5"
    mock_read_config.return_value = cfg

    h5_path = tmp_path / "base_flow.hdf5"
    h5_path.touch()

    fake_grid = SimpleNamespace(
        x=np.array([[1.0, 2.0], [1.0, 2.0]]),
        y=np.array([[0.0, 0.0], [1.0, 1.0]]),
        shape=(2, 2),
    )
    fake_flow = SimpleNamespace(fields={"u": np.ones((2, 2))})
    mock_load_with_cfd_io.return_value = (fake_grid, fake_flow, {})

    with patch("lst_tools.cli.cmd_lastrac.write_tecplot_ascii") as mock_write:
        prev = Path.cwd()
        try:
            import os
            os.chdir(tmp_path)
            result = runner.invoke(cli, ["--debug", "lastrac"])
        finally:
            os.chdir(prev)

    assert result.exit_code == 0
    assert mock_write.call_count == 1
    kwargs = mock_convert.call_args.kwargs
    assert kwargs["debug_path"] == Path("./debug")
