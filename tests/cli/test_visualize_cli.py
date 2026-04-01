"""Tests for the typer-based lst-tools visualize wrappers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from lst_tools.cli.main import cli

runner = CliRunner()


class TestVisualizeCLI:
    """Test suite for visualize parsing/tracking wrappers."""

    def test_visualize_group_help(self):
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
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
                "--single-k",
                "--k-index",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert "visualization complete (tracking)" in result.output
        mock_render.assert_called_once()
        kwargs = mock_render.call_args.kwargs
        assert kwargs["all_k"] is False
        assert kwargs["k_index"] == 1

    @patch("lst_tools.cli.cmd_visualize.importlib.import_module", side_effect=ImportError("cfd-viz missing"))
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
        assert "cfd-viz is required for visualization wrappers" in result.output

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
    ):
        with runner.isolated_filesystem():
            kc0 = Path("kc_0000")
            kc5 = Path("kc_0005")
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

    def test_visualize_tracking_fallback_missing_all_inputs(self):
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["visualize", "tracking"])

        assert result.exit_code != 0
        assert "lst_vol.dat not found and no kc_* tracking slices discovered" in result.output
