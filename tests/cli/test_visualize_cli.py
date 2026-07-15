"""Tests for the typer-based lst-tools visualize wrappers."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from typer.testing import CliRunner

from lst_tools.cli.app import cli
from lst_tools.cli.cmd_visualize import (
    _compute_shared_bounds,
    _discover_tracking_files,
    _resolve_field_name,
    _split_candidates,
    _visualize_data,
)

runner = CliRunner()


class TestVisualizeCLI:
    """Test suite for visualize parsing/tracking wrappers."""

    def test_visualize_group_help(self):
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "parsing" in result.output
        assert "tracking" in result.output

    def test_visualize_group_no_args_shows_help(self):
        result = runner.invoke(cli, ["visualize"])
        assert result.exit_code == 0
        assert "Visualize LST results." in result.output
        assert "parsing" in result.output
        assert "tracking" in result.output

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_parsing_dispatch(self, mock_import_module, tmp_path: Path):
        input_file = tmp_path / "growth_rate_with_nfact_amps.dat"
        input_file.write_text("dummy", encoding="utf-8")

        out_dir = tmp_path / "viz_parsing"
        mock_render = mock_import_module.return_value.render_lst_contours
        mock_render.return_value = [
            out_dir / "alpi_kc_0000.png",
            out_dir / "alpi_kc_0005.png",
        ]

        result = runner.invoke(
            cli,
            [
                "visualize",
                "parsing",
                "--input",
                str(input_file),
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert "visualization complete (parsing)" in result.output
        mock_render.assert_called_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["path"] == input_file
        assert kwargs["out_dir"] == out_dir
        assert kwargs["field"] == "-im(alpha)"
        assert kwargs["all_k"] is True
        assert kwargs["show"] is False

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_tracking_dispatch(self, mock_import_module, tmp_path: Path):
        input_file = tmp_path / "lst_vol.dat"
        input_file.write_text("dummy", encoding="utf-8")

        out_dir = tmp_path / "viz_tracking"
        mock_render = mock_import_module.return_value.render_lst_contours
        mock_render.return_value = [out_dir / "alpi_kc_0100.png"]

        result = runner.invoke(
            cli,
            [
                "visualize",
                "tracking",
                "--input",
                str(input_file),
                "--out",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert "visualization complete (tracking)" in result.output
        mock_render.assert_called_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["field"] == "-im(alpha)"
        assert kwargs["all_k"] is True
        assert kwargs["k_index"] == 1

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module", side_effect=ImportError("visualization backend missing"))
    def test_visualize_missing_dependency(self, _mock_import_module, tmp_path: Path):
        input_file = tmp_path / "growth_rate_with_nfact_amps.dat"
        input_file.write_text("dummy", encoding="utf-8")

        result = runner.invoke(
            cli,
            [
                "visualize",
                "parsing",
                "--input",
                str(input_file),
            ],
        )

        assert result.exit_code != 0
        assert "visualization support is required for visualize commands" in result.output

    def test_visualize_input_missing(self):
        result = runner.invoke(
            cli,
            [
                "visualize",
                "parsing",
                "--input",
                "missing_input.dat",
            ],
        )

        assert result.exit_code != 0
        assert "input file not found" in result.output

    @patch("lst_tools.cli.cmd_visualize._visualize_data")
    @patch("lst_tools.cli.cmd_visualize._compute_shared_bounds", return_value=(0.0, 50.0))
    def test_visualize_tracking_fallback_kc_dirs(
        self,
        _mock_bounds,
        mock_visualize_data,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        kc0 = tmp_path / "kc_0000"
        kc5 = tmp_path / "kc_0005"
        kc0.mkdir()
        kc5.mkdir()
        (kc0 / "growth_rate_with_nfact_amps.dat").write_text("dummy", encoding="utf-8")
        (kc5 / "growth_rate_with_nfact_amps.dat").write_text("dummy", encoding="utf-8")

        mock_visualize_data.side_effect = [
            [Path("alpi_contours_tracking/alpi_kc_kc_0000_0000.png")],
            [Path("alpi_contours_tracking/alpi_kc_kc_0005_0005.png")],
        ]

        result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code == 0
        assert "tracking fallback: kc_* slices" in result.output
        assert mock_visualize_data.call_count == 2

        first_call = mock_visualize_data.call_args_list[0].kwargs
        second_call = mock_visualize_data.call_args_list[1].kwargs

        assert first_call["all_k"] is False
        assert first_call["k_index"] == 1
        assert first_call["level_min_override"] == 0.0
        assert first_call["level_max_override"] == 50.0
        assert second_call["all_k"] is False
        assert second_call["k_index"] == 1
        assert second_call["level_min_override"] == 0.0
        assert second_call["level_max_override"] == 50.0

    def test_visualize_tracking_fallback_missing_all_inputs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code != 0
        assert "lst_vol.dat not found and no kc_* tracking slices discovered" in result.output


class _FakeFlowField:
    """Small stand-in for cfd_io field objects."""

    def __init__(self, values):
        self.data = values


class _FakeDataset:
    """Small stand-in for cfd_io datasets."""

    def __init__(self, flow):
        self.flow = flow


class TestVisualizeHelpers:
    """Direct tests for helper logic in cmd_visualize."""

    def test_split_candidates_strips_and_filters_empty(self):
        result = _split_candidates(" alpha, beta , , gamma ,, ")

        assert result == ["alpha", "beta", "gamma"]

    def test_resolve_field_name_returns_first_match(self):
        flow = {"beta": object(), "-im(alpha)": object()}

        result = _resolve_field_name(flow, "missing, -im(alpha), beta")

        assert result == "-im(alpha)"

    def test_resolve_field_name_raises_with_available_fields(self):
        flow = {"alpha": object(), "beta": object()}

        try:
            _resolve_field_name(flow, "missing, other")
        except KeyError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected KeyError")

        assert "none of the requested fields were found" in message
        assert "alpha" in message
        assert "beta" in message

    def test_discover_tracking_files_ignores_non_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "kc_0000").mkdir()
        (tmp_path / "kc_0000" / "growth_rate_with_nfact_amps.dat").write_text(
            "dummy", encoding="utf-8"
        )
        (tmp_path / "kc_0005").mkdir()
        (tmp_path / "kc_0010").write_text("not a directory", encoding="utf-8")

        result = _discover_tracking_files(Path("."))

        assert result == [Path("kc_0000/growth_rate_with_nfact_amps.dat")]

    def test_compute_shared_bounds_positive_rounded(self, monkeypatch):
        fake_module = types.ModuleType("cfd_io")

        data_map = {
            "first.dat": _FakeDataset(
                {"-im(alpha)": _FakeFlowField(values=__import__("numpy").array([1.2, 9.8]))}
            ),
            "second.dat": _FakeDataset(
                {"-im(alpha)": _FakeFlowField(values=__import__("numpy").array([2.0, 12.1]))}
            ),
        }

        def _fake_read_file(path: str):
            return data_map[path]

        fake_module.read_file = _fake_read_file
        monkeypatch.setitem(sys.modules, "cfd_io", fake_module)

        level_min, level_max = _compute_shared_bounds(
            input_files=[Path("first.dat"), Path("second.dat")],
            field="-im(alpha)",
            levels_policy="positive-rounded",
        )

        assert level_min == 0.0
        assert level_max == 20.0

    def test_compute_shared_bounds_global_auto_and_degenerate_max(self, monkeypatch):
        import numpy as np

        fake_module = types.ModuleType("cfd_io")

        data_map = {
            "same.dat": _FakeDataset(
                {"beta": _FakeFlowField(values=np.array([3.0, 3.0]))}
            )
        }

        def _fake_read_file(path: str):
            return data_map[path]

        fake_module.read_file = _fake_read_file
        monkeypatch.setitem(sys.modules, "cfd_io", fake_module)

        level_min, level_max = _compute_shared_bounds(
            input_files=[Path("same.dat")],
            field="beta",
            levels_policy="global-auto",
        )

        assert level_min == 3.0
        assert level_max == 4.0

    def test_compute_shared_bounds_unknown_policy_raises(self, monkeypatch):
        import numpy as np

        fake_module = types.ModuleType("cfd_io")
        fake_module.read_file = lambda _path: _FakeDataset(
            {"beta": _FakeFlowField(values=np.array([1.0, 2.0]))}
        )
        monkeypatch.setitem(sys.modules, "cfd_io", fake_module)

        try:
            _compute_shared_bounds(
                input_files=[Path("sample.dat")],
                field="beta",
                levels_policy="bad-policy",
            )
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected ValueError")

        assert "unknown levels policy" in message

    def test_visualize_data_missing_input_raises(self, tmp_path: Path):
        missing_input = tmp_path / "missing.dat"

        try:
            _visualize_data(
                stage="parsing",
                input_path=missing_input,
                out_dir=tmp_path / "out",
                prefix="alpi",
                field="-im(alpha)",
                xvar="s",
                yvar="freq",
                kvar="beta",
                all_k=True,
                k_index=1,
                levels_policy="positive-rounded",
                levels_count=60,
                clip_below=True,
                dpi=300,
            )
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")

        assert "input file not found" in message

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module", side_effect=ImportError("missing backend"))
    def test_visualize_data_missing_backend_raises_runtime_error(self, _mock_import_module, tmp_path: Path):
        input_file = tmp_path / "input.dat"
        input_file.write_text("dummy", encoding="utf-8")

        try:
            _visualize_data(
                stage="parsing",
                input_path=input_file,
                out_dir=tmp_path / "out",
                prefix="alpi",
                field="-im(alpha)",
                xvar="s",
                yvar="freq",
                kvar="beta",
                all_k=True,
                k_index=1,
                levels_policy="positive-rounded",
                levels_count=60,
                clip_below=True,
                dpi=300,
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected RuntimeError")

        assert "visualization support is required" in message

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_data_dispatch_and_summary(self, mock_import_module, tmp_path: Path, capsys):
        input_file = tmp_path / "input.dat"
        input_file.write_text("dummy", encoding="utf-8")
        out_dir = tmp_path / "out"
        mock_render = mock_import_module.return_value.render_lst_contours
        mock_render.return_value = [out_dir / "first.png", out_dir / "last.png"]

        result = _visualize_data(
            stage="tracking",
            input_path=input_file,
            out_dir=out_dir,
            prefix="alpi",
            field="-im(alpha)",
            xvar="s",
            yvar="freq",
            kvar="beta",
            all_k=False,
            k_index=2,
            levels_policy="positive-rounded",
            levels_count=25,
            level_min_override=0.0,
            level_max_override=10.0,
            clip_below=True,
            dpi=150,
        )

        assert result == [out_dir / "first.png", out_dir / "last.png"]
        mock_render.assert_called_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["path"] == input_file
        assert kwargs["out_dir"] == out_dir
        assert kwargs["prefix"] == "alpi"
        assert kwargs["all_k"] is False
        assert kwargs["k_index"] == 2
        assert kwargs["level_min_override"] == 0.0
        assert kwargs["level_max_override"] == 10.0
        assert kwargs["levels_count"] == 25
        assert kwargs["dpi"] == 150

        output = capsys.readouterr().out
        assert "visualization complete (tracking)" in output
        assert "wrote 2 plot(s)" in output
        assert "first:" in output
        assert "last:" in output

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_data_emit_summary_false_suppresses_output(self, mock_import_module, tmp_path: Path, capsys):
        input_file = tmp_path / "input.dat"
        input_file.write_text("dummy", encoding="utf-8")
        mock_import_module.return_value.render_lst_contours.return_value = []

        _visualize_data(
            stage="tracking",
            input_path=input_file,
            out_dir=tmp_path / "out",
            prefix="alpi",
            field="-im(alpha)",
            xvar="s",
            yvar="freq",
            kvar="beta",
            all_k=False,
            k_index=1,
            levels_policy="positive-rounded",
            levels_count=60,
            clip_below=True,
            dpi=300,
            emit_summary=False,
        )

        output = capsys.readouterr().out
        assert output == ""
