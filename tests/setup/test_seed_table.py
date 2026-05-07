"""Tests for seed-table generation helpers and per-case writer."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from lst_tools.process.maxima import Ridge
from lst_tools.setup import _seed_table as seed_table_mod


class _FakeTecplotData:
    """Minimal Tecplot-like object exposing field arrays used by seed tables."""

    def __init__(self, fields: dict[str, np.ndarray]) -> None:
        self._fields = fields

    def field(self, name: str) -> np.ndarray:
        return self._fields[name]


def _build_cfg(**overrides) -> SimpleNamespace:
    """Build a minimal config namespace for seed-table tests."""

    defaults = {
        "enabled": True,
        "n_seeds": 4,
        "min_growth": 0.0,
        "x_range": [],
        "f_range": [],
        "gate_tol": 0.05,
        "min_valid": 2,
        "smooth_passes": 0,
        "gate_by_keep_mask": False,
        "threshold": 0.15,
        "output_file": "seed_alpha.dat",
    }
    defaults.update(overrides)
    seed_table = SimpleNamespace(**defaults)
    return SimpleNamespace(seed_table=seed_table)


def _build_tp() -> _FakeTecplotData:
    """Build a consistent fake parsing solution."""

    fields = {
        "s": np.array([[[0.0, 1.0, 2.0, 3.0]]], dtype=float),
        "freq": np.array([[[100.0], [200.0], [300.0]]], dtype=float),
        "alpr": np.array(
            [
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [20.0, 21.0, 22.0, 23.0],
                    [30.0, 31.0, 32.0, 33.0],
                ]
            ],
            dtype=float,
        ),
        "alpi": np.array(
            [
                [
                    [0.10, 0.20, 0.30, 0.40],
                    [0.50, 0.60, 0.70, 0.80],
                    [0.15, 0.25, 0.35, 0.45],
                ]
            ],
            dtype=float,
        ),
    }
    return _FakeTecplotData(fields)


class TestRidgeToSeeds:
    """Exercise ridge filtering and downsampling behavior."""

    def test_ridge_to_seeds_filters_and_converts_sign(self):
        ridge = Ridge(indices=[(0, 0), (1, 1.2), (2, 2), (10, 1)])
        x_arr = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
        freq_arr = np.array([100.0, 200.0, 300.0], dtype=float)
        alpr_2d = np.array(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
                [30.0, 31.0, 32.0, 33.0],
            ],
            dtype=float,
        )
        alpi_2d = np.array(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.50, 0.60, 0.70, 0.80],
                [0.15, 0.25, 0.35, 0.45],
            ],
            dtype=float,
        )
        keep_mask = np.ones((3, 4), dtype=bool)
        keep_mask[2, 2] = False

        seeds = seed_table_mod._ridge_to_seeds(
            ridge,
            x_arr=x_arr,
            freq_arr=freq_arr,
            alpr_2d=alpr_2d,
            alpi_2d=alpi_2d,
            n_seeds=5,
            min_growth=0.55,
            x_range=[0.5, 2.5],
            f_range=[150.0, 300.0],
            keep_mask=keep_mask,
        )

        assert seeds == [(1.0, 200.0, 21.0, -0.60)]

    def test_ridge_to_seeds_downsamples_evenly(self):
        x_arr = np.arange(10, dtype=float)
        freq_arr = np.array([100.0, 200.0, 300.0], dtype=float)
        alpr_2d = np.tile(np.arange(10, dtype=float), (3, 1))
        alpi_2d = np.full((3, 10), 0.5, dtype=float)
        ridge = Ridge(indices=[(index, 1) for index in range(10)])

        seeds = seed_table_mod._ridge_to_seeds(
            ridge,
            x_arr=x_arr,
            freq_arr=freq_arr,
            alpr_2d=alpr_2d,
            alpi_2d=alpi_2d,
            n_seeds=3,
            min_growth=0.0,
            x_range=[],
            f_range=[],
            keep_mask=None,
        )

        seed_x = [row[0] for row in seeds]
        assert seed_x == [0.0, 4.0, 9.0]


class TestWriteSeedFile:
    """Exercise ASCII seed file writing."""

    def test_write_seed_file_writes_expected_content(self, tmp_path: Path):
        out_path = tmp_path / "seed_alpha.dat"
        seeds = [(1.0, 200.0, 21.0, -0.6), (2.0, 300.0, 32.0, -0.35)]

        seed_table_mod._write_seed_file(
            out_path,
            seeds,
            threshold=0.15,
            source_label="parsing.dat",
            beta_label="10",
        )

        content = out_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        data_lines = [line for line in lines if line and not line.startswith("#")]

        assert "# seed_alpha.dat -- tracking solver initial-guess seeds" in content
        assert "# source        : parsing.dat" in content
        assert "# beta          : 10" in content
        assert "# threshold     : 0.15" in content
        assert data_lines[0] == "2"
        assert len(data_lines) == 3


class TestWriteSeedTableForCase:
    """Exercise the top-level per-case seed generation flow."""

    def test_returns_none_when_disabled(self, tmp_path: Path):
        cfg = _build_cfg(enabled=False)
        out_path, seeds = seed_table_mod.write_seed_table_for_case(
            case_dir=tmp_path,
            cfg=cfg,
            tp=_build_tp(),
            idx_betr=0,
            betr_loc=10.0,
            source_label="parsing.dat",
        )

        assert out_path is None
        assert seeds == []

    @patch("lst_tools.setup._seed_table._track_ridges")
    def test_generates_and_sorts_seed_file(self, mock_track_ridges, tmp_path: Path):
        cfg = _build_cfg(min_valid=2, n_seeds=5, gate_by_keep_mask=False)
        tp = _build_tp()
        mock_track_ridges.return_value = [
            Ridge(indices=[(2, 1), (0, 1)]),
            Ridge(indices=[(3, 2)]),
        ]

        out_path, seeds = seed_table_mod.write_seed_table_for_case(
            case_dir=tmp_path,
            cfg=cfg,
            tp=tp,
            idx_betr=0,
            betr_loc=10.0,
            source_label="parsing.dat",
        )

        assert out_path == tmp_path / "seed_alpha.dat"
        assert out_path.is_file()
        assert [row[0] for row in seeds] == [0.0, 2.0]
        assert seeds[0] == (0.0, 200.0, 20.0, -0.5)
        assert seeds[1] == (2.0, 200.0, 22.0, -0.7)

    @patch("lst_tools.setup.tracking.smooth_contour_field")
    @patch("lst_tools.setup._seed_table._track_ridges")
    def test_uses_smoothing_and_keep_mask_gate(
        self,
        mock_track_ridges,
        mock_smooth_contour_field,
        tmp_path: Path,
    ):
        # smooth_passes=2 means smooth_contour_field is called ONCE for detection.
        # The keep_mask for gating is now built from the union of detected ridge
        # bands (ridge-union approach), not from a second smooth_contour_field call.
        cfg = _build_cfg(
            min_valid=1,
            smooth_passes=2,
            gate_by_keep_mask=True,
            n_seeds=5,
        )
        tp = _build_tp()
        detection_mask = np.ones((3, 4), dtype=bool)
        mock_smooth_contour_field.return_value = (
            tp.field("alpi")[0, :, :],
            detection_mask,
        )
        # Ridge has two points: (i_x=0, j_f=1) and (i_x=2, j_f=1).
        # The ridge-union keep_mask covers ±3 freq-bins around j=1 at those x
        # columns (all 3 freq rows), so both candidates pass gating.
        mock_track_ridges.return_value = [Ridge(indices=[(0, 1), (2, 1)])]

        out_path, seeds = seed_table_mod.write_seed_table_for_case(
            case_dir=tmp_path,
            cfg=cfg,
            tp=tp,
            idx_betr=0,
            betr_loc=10.0,
            source_label="parsing.dat",
        )

        assert out_path == tmp_path / "seed_alpha.dat"
        # both ridge points survive the ridge-union gate
        assert seeds == [(0.0, 200.0, 20.0, -0.5), (2.0, 200.0, 22.0, -0.7)]
        # smooth_contour_field called once (detection only; mask from ridges)
        assert mock_smooth_contour_field.call_count == 1

    @patch("lst_tools.setup._seed_table._track_ridges")
    def test_writes_empty_file_when_no_seeds_survive(self, mock_track_ridges, tmp_path: Path):
        cfg = _build_cfg(min_valid=1, min_growth=10.0)
        tp = _build_tp()
        mock_track_ridges.return_value = [Ridge(indices=[(0, 1), (2, 1)])]

        out_path, seeds = seed_table_mod.write_seed_table_for_case(
            case_dir=tmp_path,
            cfg=cfg,
            tp=tp,
            idx_betr=0,
            betr_loc=10.0,
            source_label="parsing.dat",
        )

        content_lines = out_path.read_text(encoding="utf-8").splitlines()
        data_lines = [line for line in content_lines if line and not line.startswith("#")]
        assert seeds == []
        assert data_lines[0] == "0"