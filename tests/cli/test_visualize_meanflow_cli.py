"""Tests for the meanflow visualization CLI and plotting helper."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
from typer.testing import CliRunner

from lst_tools.cli.cmd_visualize_meanflow import _visualize_meanflow
from lst_tools.cli.app import cli


runner = CliRunner()


class _FakeAxes:
    """Lightweight stand-in for matplotlib axes."""

    def __init__(self) -> None:
        self.labels: dict[str, object] = {}
        self.plots: list[tuple[np.ndarray, np.ndarray]] = []
        self.vlines: list[float] = []

    def plot(self, x_values: np.ndarray, y_values: np.ndarray, *_args, **_kwargs) -> None:
        self.plots.append((x_values, y_values))

    def axvline(self, x_value: float, **_kwargs) -> None:
        self.vlines.append(x_value)

    def set_xlabel(self, value: str) -> None:
        self.labels["xlabel"] = value

    def set_ylabel(self, value: str) -> None:
        self.labels["ylabel"] = value

    def set_ylim(self, lower: float, upper: float) -> None:
        self.labels["ylim"] = (lower, upper)

    def set_title(self, value: str) -> None:
        self.labels["title"] = value


class _FakeFigure:
    """Lightweight stand-in for matplotlib figures."""

    def __init__(self, saved_files: list[tuple[Path, int]]) -> None:
        self.saved_files = saved_files
        self.tight_layout_called = False

    def tight_layout(self) -> None:
        self.tight_layout_called = True

    def savefig(self, out_file: Path, dpi: int) -> None:
        self.saved_files.append((Path(out_file), dpi))


class _FakePyplot:
    """Minimal pyplot replacement used by tests."""

    def __init__(self) -> None:
        self.axes: list[_FakeAxes] = []
        self.closed_figures: list[_FakeFigure] = []
        self.saved_files: list[tuple[Path, int]] = []

    def subplots(self, **_kwargs) -> tuple[_FakeFigure, _FakeAxes]:
        figure = _FakeFigure(self.saved_files)
        axes = _FakeAxes()
        self.axes.append(axes)
        return figure, axes

    def close(self, figure: _FakeFigure) -> None:
        self.closed_figures.append(figure)


def _install_fake_matplotlib(monkeypatch, fake_pyplot: _FakePyplot) -> None:
    """Register a fake matplotlib.pyplot module for the helper import."""

    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot_module = types.ModuleType("matplotlib.pyplot")
    fake_pyplot_module.subplots = fake_pyplot.subplots
    fake_pyplot_module.close = fake_pyplot.close
    fake_matplotlib.pyplot = fake_pyplot_module

    monkeypatch.setitem(sys.modules, "matplotlib", fake_matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_pyplot_module)


def _build_profiles(*, include_crossflow: bool) -> dict[str, object]:
    """Build synthetic profile data for plotting tests."""

    x_values = np.array([0.10, 0.30, 0.60], dtype=float)
    eta_values = np.array([0.00, 0.10, 0.20, 0.35, 0.60, 1.00], dtype=float)

    eta_list = [eta_values.copy() for _ in x_values]
    uvel_list = [
        np.array([0.00, 0.25, 0.55, 0.82, 0.95, 1.00], dtype=float) + 0.02 * index
        for index, _ in enumerate(x_values)
    ]
    temp_list = [
        np.array([1.00, 0.98, 0.90, 0.74, 0.58, 0.52], dtype=float) - 0.01 * index
        for index, _ in enumerate(x_values)
    ]

    if include_crossflow:
        wvel_list = [
            np.array([0.00, 0.02, 0.05, 0.03, 0.01, 0.00], dtype=float) + 0.005 * index
            for index, _ in enumerate(x_values)
        ]
    else:
        wvel_list = [np.zeros_like(eta_values) for _ in x_values]

    return {
        "x": x_values,
        "eta": eta_list,
        "uvel": uvel_list,
        "wvel": wvel_list,
        "temp": temp_list,
    }


class TestVisualizeMeanflowCLI:
    """Exercise the meanflow visualization command paths."""

    def test_visualize_group_help_lists_meanflow(self):
        result = runner.invoke(cli, ["visualize", "--help"])

        assert result.exit_code == 0
        assert "meanflow" in result.output

    def test_visualize_meanflow_missing_input(self):
        result = runner.invoke(cli, ["visualize", "meanflow", "missing_meanflow.bin"])

        assert result.exit_code == 1
        assert "error: missing_meanflow.bin not found" in result.output

    @patch("lst_tools.cli.cmd_visualize_meanflow._visualize_meanflow")
    def test_visualize_meanflow_dispatches_to_helper(self, mock_visualize_meanflow, tmp_path: Path):
        meanflow_path = tmp_path / "meanflow.bin"
        meanflow_path.write_bytes(b"synthetic")
        out_dir = tmp_path / "plots"

        result = runner.invoke(
            cli,
            [
                "visualize",
                "meanflow",
                str(meanflow_path),
                "--out",
                str(out_dir),
                "--dpi",
                "144",
            ],
        )

        assert result.exit_code == 0
        mock_visualize_meanflow.assert_called_once_with(meanflow_path, out_dir, 144)

    @patch("lst_tools.cli.cmd_visualize_meanflow._visualize_meanflow", side_effect=RuntimeError("boom"))
    def test_visualize_meanflow_reports_helper_errors(self, _mock_visualize_meanflow, tmp_path: Path):
        meanflow_path = tmp_path / "meanflow.bin"
        meanflow_path.write_bytes(b"synthetic")

        result = runner.invoke(cli, ["visualize", "meanflow", str(meanflow_path)])

        assert result.exit_code == 1
        assert "error: boom" in result.output


class TestVisualizeMeanflowHelper:
    """Exercise the plotting helper without importing real matplotlib."""

    def test_visualize_meanflow_without_matplotlib_raises(self, tmp_path: Path):
        meanflow_path = tmp_path / "meanflow.bin"
        meanflow_path.write_bytes(b"synthetic")

        # read
        original_import = builtins.__import__

        # build selective import failure for matplotlib
        def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "matplotlib.pyplot":
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=_raising_import):
            try:
                _visualize_meanflow(meanflow_path, tmp_path / "plots", 100)
            except ModuleNotFoundError as exc:
                assert "matplotlib is required" in str(exc)
            else:
                raise AssertionError("expected ModuleNotFoundError")

    @patch("lst_tools.cli.cmd_visualize_meanflow.read_baseflow_profiles")
    def test_visualize_meanflow_writes_u_and_temp_plots(
        self,
        mock_read_baseflow_profiles,
        tmp_path: Path,
        monkeypatch,
        capsys,
    ):
        meanflow_path = tmp_path / "meanflow.bin"
        meanflow_path.write_bytes(b"synthetic")
        out_dir = tmp_path / "meanflow_profiles"
        fake_pyplot = _FakePyplot()

        mock_read_baseflow_profiles.return_value = _build_profiles(include_crossflow=False)
        _install_fake_matplotlib(monkeypatch, fake_pyplot)

        _visualize_meanflow(meanflow_path, out_dir, 111)

        saved_names = [path.name for path, _dpi in fake_pyplot.saved_files]
        saved_dpis = [dpi for _path, dpi in fake_pyplot.saved_files]

        assert saved_names == ["meanflow_uvel.png", "meanflow_temp.png"]
        assert saved_dpis == [111, 111]
        assert out_dir.is_dir()
        assert len(fake_pyplot.axes) == 2
        assert len(fake_pyplot.closed_figures) == 2

        output = capsys.readouterr().out
        assert "gradient-based deltas:" in output
        assert "wrote" in output

    @patch("lst_tools.cli.cmd_visualize_meanflow.read_baseflow_profiles")
    def test_visualize_meanflow_writes_crossflow_plot(
        self,
        mock_read_baseflow_profiles,
        tmp_path: Path,
        monkeypatch,
    ):
        meanflow_path = tmp_path / "meanflow.bin"
        meanflow_path.write_bytes(b"synthetic")
        out_dir = tmp_path / "meanflow_profiles"
        fake_pyplot = _FakePyplot()

        mock_read_baseflow_profiles.return_value = _build_profiles(include_crossflow=True)
        _install_fake_matplotlib(monkeypatch, fake_pyplot)

        _visualize_meanflow(meanflow_path, out_dir, 200)

        saved_names = [path.name for path, _dpi in fake_pyplot.saved_files]

        assert saved_names == [
            "meanflow_uvel.png",
            "meanflow_wvel.png",
            "meanflow_temp.png",
        ]
        assert len(fake_pyplot.axes) == 3