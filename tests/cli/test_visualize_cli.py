"""Tests for the typer-based lst-tools visualize wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from lst_tools.cli.app import cli
from lst_tools.cli.cmd_visualize import (
    _discover_tracking_files,
)

runner = CliRunner()


@dataclass(frozen=True)
class _FakePlotConfig:
    """Minimal plotting configuration used by wrapper tests."""

    input_path: Path | None
    output_dir: Path


class TestVisualizeCLI:
    """Test suite for visualize parsing/tracking wrappers."""

    def test_visualize_group_help(self):
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "parsing" in result.output
        assert "tracking" in result.output

    def test_visualize_group_no_args_shows_help(self):
        result = runner.invoke(cli, ["visualize"])
        # Click/Typer versions differ here: some return 0 with no_args_is_help,
        # others return 2 while still printing the subgroup help text.
        assert result.exit_code in (0, 2)
        assert "Visualize LST results." in result.output
        assert "parsing" in result.output
        assert "tracking" in result.output

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_init_dispatch(self, mock_import_module, tmp_path: Path):
        config_path = tmp_path / "cfd-viz-lst.toml"
        mock_writer = mock_import_module.return_value.write_default_lst_config
        mock_writer.return_value = config_path

        result = runner.invoke(
            cli,
            ["visualize", "init", str(config_path)],
        )

        assert result.exit_code == 0
        assert f"wrote {config_path}" in result.output
        mock_writer.assert_called_once_with(config_path, force=False)

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_parsing_dispatch(self, mock_import_module, tmp_path: Path):
        input_file = tmp_path / "growth_rate_with_nfact_amps.dat"
        input_file.write_text("dummy", encoding="utf-8")

        out_dir = tmp_path / "viz_parsing"
        mock_module = mock_import_module.return_value
        plot_config = _FakePlotConfig(
            input_path=Path("unused.dat"), output_dir=Path("unused")
        )
        mock_module.default_lst_config.return_value = plot_config
        mock_render = mock_module.render_configured_lst_collection
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
        mock_render.assert_called_once_with(
            plot_config,
            [input_file],
            output_dir=out_dir,
            prefix_suffixes=None,
            single_plane=False,
            show=False,
        )

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_tracking_dispatch(self, mock_import_module, tmp_path: Path):
        input_file = tmp_path / "lst_vol.dat"
        input_file.write_text("dummy", encoding="utf-8")

        out_dir = tmp_path / "viz_tracking"
        mock_module = mock_import_module.return_value
        plot_config = _FakePlotConfig(
            input_path=Path("unused.dat"), output_dir=Path("unused")
        )
        mock_module.default_lst_config.return_value = plot_config
        mock_render = mock_module.render_configured_lst_collection
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
        mock_render.assert_called_once_with(
            plot_config,
            [input_file],
            output_dir=out_dir,
            prefix_suffixes=None,
            single_plane=False,
            show=False,
        )

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_tracking_discovers_and_uses_config(
        self,
        mock_import_module,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "cfd-viz-lst.toml"
        config_path.write_text("config", encoding="utf-8")
        input_file = tmp_path / "configured_tracking.dat"
        input_file.write_text("dummy", encoding="utf-8")
        out_dir = tmp_path / "configured_plots"
        plot_config = _FakePlotConfig(input_path=input_file, output_dir=out_dir)

        mock_module = mock_import_module.return_value
        mock_module.load_lst_config.return_value = plot_config
        mock_module.render_configured_lst_collection.return_value = [
            out_dir / "alpi_kc_0000.png"
        ]

        result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code == 0
        mock_module.load_lst_config.assert_called_once_with(Path("cfd-viz-lst.toml"))
        mock_module.render_configured_lst_collection.assert_called_once_with(
            plot_config,
            [input_file],
            output_dir=out_dir,
            prefix_suffixes=None,
            single_plane=False,
            show=False,
        )

    @patch(
        "lst_tools.cli.cmd_visualize.importlib.import_module",
        side_effect=ImportError("visualization backend missing"),
    )
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
        assert (
            "visualization support is required for visualize commands" in result.output
        )

    @patch(
        "lst_tools.cli.cmd_visualize.importlib.import_module",
        side_effect=ImportError("visualization backend missing"),
    )
    def test_visualize_input_missing(self, mock_import_module):
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
        mock_import_module.assert_not_called()

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_tracking_fallback_kc_dirs(
        self,
        mock_import_module,
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

        mock_module = mock_import_module.return_value
        plot_config = _FakePlotConfig(
            input_path=Path("lst_vol.dat"), output_dir=Path("alpi_contours_tracking")
        )
        mock_module.default_lst_config.return_value = plot_config
        mock_module.render_configured_lst_collection.return_value = [
            Path("alpi_contours_tracking/alpi_kc_kc_0000_0000.png"),
        ]

        result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code == 0
        assert "tracking fallback: kc_* slices" in result.output
        mock_module.render_configured_lst_collection.assert_called_once_with(
            plot_config,
            [
                kc0 / "growth_rate_with_nfact_amps.dat",
                kc5 / "growth_rate_with_nfact_amps.dat",
            ],
            output_dir=Path("alpi_contours_tracking"),
            prefix_suffixes=["0000", "0005"],
            single_plane=True,
            show=False,
        )

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module")
    def test_visualize_tracking_empty_config_input_discovers_kc_dirs(
        self,
        mock_import_module,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "cfd-viz-lst.toml"
        config_path.write_text('[input]\npath = ""\n', encoding="utf-8")
        slice_dir = tmp_path / "kc_0000"
        slice_dir.mkdir()
        slice_file = slice_dir / "growth_rate_with_nfact_amps.dat"
        slice_file.write_text("dummy", encoding="utf-8")

        mock_module = mock_import_module.return_value
        plot_config = _FakePlotConfig(input_path=None, output_dir=tmp_path / "plots")
        mock_module.load_lst_config.return_value = plot_config
        mock_module.render_configured_lst_collection.return_value = [
            tmp_path / "plots" / "alpi_kc_kc_0000_0000.png"
        ]

        result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code == 0
        mock_module.render_configured_lst_collection.assert_called_once_with(
            plot_config,
            [slice_file],
            output_dir=tmp_path / "plots",
            prefix_suffixes=["0000"],
            single_plane=True,
            show=False,
        )

    @patch(
        "lst_tools.cli.cmd_visualize.importlib.import_module",
        side_effect=ImportError("visualization backend missing"),
    )
    def test_visualize_tracking_fallback_missing_all_inputs(
        self,
        mock_import_module,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code != 0
        assert (
            "lst_vol.dat not found and no kc_* tracking slices discovered"
            in result.output
        )
        mock_import_module.assert_not_called()


class TestVisualizeHelpers:
    """Direct tests for helper logic in cmd_visualize."""

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
