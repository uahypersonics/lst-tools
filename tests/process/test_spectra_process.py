"""Unit tests for spectra processing."""

from __future__ import annotations

from pathlib import Path

import pytest

from lst_tools.config import Config
from lst_tools.process.spectra import spectra_process
from lst_tools.cli.cmd_spectra_process import cmd_spectra_process


def test_spectra_process_no_case_dirs_returns_current_path(tmp_path: Path, monkeypatch) -> None:
    """Return current path marker when no matching case directories exist."""
    monkeypatch.chdir(tmp_path)

    out = spectra_process(cfg=None)

    assert out == Path(".")


def test_spectra_process_writes_animation_file_for_loaded_cases(tmp_path: Path, monkeypatch) -> None:
    """Write one Tecplot animation file for a loaded (freq, beta) group."""
    monkeypatch.chdir(tmp_path)
    status_messages: list[str] = []

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    eig_file_a = case_dir_a / "Eigenvalues_case.dat"
    eig_file_a.write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    eig_file_b = case_dir_b / "Eigenvalues_case.dat"
    eig_file_b.write_text("1.1 2.1\n3.1 4.1\n", encoding="utf-8")

    out = spectra_process(cfg=None, reporter=status_messages.append)

    assert out == Path("spectra_animation")
    assert any("found 2 spectra case directories" in message for message in status_messages)
    assert "parameter space:" in status_messages
    assert any("- 2 x-station(s) [0.10000 ... 0.20000]" in message for message in status_messages)
    assert any("- 1 frequency value(s) [6000 ... 6000] Hz" in message for message in status_messages)
    assert any("- 1 beta value(s) [500.00 ... 500.00]" in message for message in status_messages)
    assert any("isolation score: 3-neighbor distance in normalized alpha-space" in message for message in status_messages)
    assert any("detachment score: nearest-neighbor gap normalized by nearby branch spacing" in message for message in status_messages)
    assert any("writing group: f = 6.000 kHz, beta = 500.00, x-stations = 2" in message for message in status_messages)
    assert any("seed selection kept" in message for message in status_messages)
    assert any("tracked 2 forward branch(es): f = 6.000 kHz, beta = 500.00" in message for message in status_messages)
    assert any("tracked 2 backward branch(es): f = 6.000 kHz, beta = 500.00" in message for message in status_messages)
    assert any("top detached branch candidates:" == message for message in status_messages)
    assert any("wrote 1 Tecplot animation file(s)" in message for message in status_messages)
    assert any("wrote 2 tracked-branch file(s)" in message for message in status_messages)
    assert any("wrote 2 tracked-branch contour file(s)" in message for message in status_messages)
    assert any("wrote 2 tracked-branch summary file(s)" in message for message in status_messages)

    out_file = tmp_path / out / "spectra_f_0006pt00_khz_beta_pos0500pt00.dat"
    assert out_file.exists()

    branch_file = tmp_path / "spectra_branches" / "spectra_branches_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    backward_branch_file = tmp_path / "spectra_branches" / "spectra_branches_backward_f_0006pt00_khz_beta_pos0500pt00.dat"
    assert branch_file.exists()
    assert backward_branch_file.exists()

    branch_contour_file = tmp_path / "spectra_branches" / "spectra_branch_contours_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    backward_branch_contour_file = tmp_path / "spectra_branches" / "spectra_branch_contours_backward_f_0006pt00_khz_beta_pos0500pt00.dat"
    assert branch_contour_file.exists()
    assert backward_branch_contour_file.exists()

    branch_summary_file = tmp_path / "spectra_branches" / "spectra_branch_summary_forward_f_0006pt00_khz_beta_pos0500pt00.csv"
    backward_branch_summary_file = tmp_path / "spectra_branches" / "spectra_branch_summary_backward_f_0006pt00_khz_beta_pos0500pt00.csv"
    assert branch_summary_file.exists()
    assert backward_branch_summary_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert 'TITLE = "spectra_animation"' in content
    assert 'ZONE T="x=1.000000e-01 f=6.000kHz beta=5.000000e+02"' in content
    assert 'ZONE T="x=2.000000e-01 f=6.000kHz beta=5.000000e+02"' in content
    assert "1.00000000e+00 2.00000000e+00 1.00000000e-01" in content
    assert "1.10000000e+00 2.10000000e+00 2.00000000e-01" in content

    branch_content = branch_file.read_text(encoding="utf-8")
    assert 'TITLE = "spectra_branches"' in branch_content
    assert 'VARIABLES = "x" "alpha_r" "alpha_i" "isolation_score" "gap_score" "branch_score" "freq" "beta" "branch_id"' in branch_content
    assert 'ZONE T="branch_001", I=2, DATAPACKING=POINT' in branch_content
    assert "1.00000000e-01 1.00000000e+00 2.00000000e+00" in branch_content
    assert "2.00000000e-01 1.10000000e+00 2.10000000e+00" in branch_content

    branch_contour_content = branch_contour_file.read_text(encoding="utf-8")
    assert 'TITLE = "spectra_branch_contours_forward"' in branch_contour_content
    assert 'VARIABLES = "x" "branch_number" "alpha_r" "alpha_i" "isolation_score" "gap_score" "branch_score" "blank" "freq" "beta"' in branch_contour_content
    assert 'ZONE T="tracked_branches_forward", I=2, J=2, DATAPACKING=POINT' in branch_contour_content
    assert "1.00000000e-01 1.00000000e+00 1.00000000e+00 2.00000000e+00" in branch_contour_content
    assert "2.00000000e-01 2.00000000e+00 3.10000000e+00 4.10000000e+00" in branch_contour_content
    assert "nan" not in branch_contour_content.lower()
    assert "-9.99000000e+02" not in branch_contour_content
    assert " 0.00000000e+00 6.00000000e+03 5.00000000e+02" in branch_contour_content

    branch_summary_content = branch_summary_file.read_text(encoding="utf-8")
    assert branch_summary_content.startswith("branch_number,score,n_points")
    assert "gap_median,gap_max,detached_fraction,longest_detached_run,birth_x,absorption_x" in branch_summary_content
    assert ",2," in branch_summary_content


def test_spectra_process_branch_contours_use_fill_value_and_blank_mask_for_missing_points(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Use -999 fill values and blank=1 when a branch is absent at a station."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text("1.1 2.1\n3.1 4.1\n", encoding="utf-8")

    case_dir_c = tmp_path / "x_00pt30_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_c.mkdir()
    (case_dir_c / "Eigenvalues_case.dat").write_text("1.2 2.2\n", encoding="utf-8")

    cfg = Config.from_dict(
        {
            "processing": {
                "spectra": {
                    "branch_min_points": 1,
                },
            }
        }
    )

    spectra_process(cfg=cfg, do_animate=False, do_branches=True, do_classify=False)

    branch_contour_file = tmp_path / "spectra_branches" / "spectra_branch_contours_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    branch_contour_content = branch_contour_file.read_text(encoding="utf-8")

    assert 'ZONE T="tracked_branches_forward", I=3, J=2, DATAPACKING=POINT' in branch_contour_content
    assert "3.00000000e-01 2.00000000e+00 -9.99000000e+02 -9.99000000e+02 -9.99000000e+02" in branch_contour_content
    assert "-9.99000000e+02 0.00000000e+00 1.00000000e+00 6.00000000e+03 5.00000000e+02" in branch_contour_content
    assert "1.00000000e+00 6.00000000e+03 5.00000000e+02" in branch_contour_content


def test_spectra_process_seed_filter_blocks_unmatched_nonseed_births(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Allow new branch births only from accepted seed points."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text(
        "0.0 0.0\n0.1 0.0\n0.2 0.0\n1.0 1.0\n",
        encoding="utf-8",
    )

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text(
        "0.0 0.0\n0.1 0.0\n0.2 0.0\n1.1 1.1\n5.0 5.0\n",
        encoding="utf-8",
    )

    spectra_process(cfg=None, do_animate=False, do_branches=True, do_classify=False)

    branch_file = tmp_path / "spectra_branches" / "spectra_branches_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    branch_content = branch_file.read_text(encoding="utf-8")

    zone_count = branch_content.count('ZONE T="branch_')
    assert zone_count == 3
    assert 'ZONE T="branch_004"' not in branch_content
    assert "5.00000000e+00 5.00000000e+00" not in branch_content


def test_spectra_process_detached_points_can_continue_across_relaxed_gate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Allow detached branch points to continue across a slightly looser match gate."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text(
        "0.0 0.0\n0.1 0.0\n0.2 0.0\n0.3 -20.0\n",
        encoding="utf-8",
    )

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text(
        "0.0 0.0\n0.1 0.0\n0.2 0.0\n0.36 -28.0\n",
        encoding="utf-8",
    )

    spectra_process(cfg=None, do_animate=False, do_branches=True, do_classify=False)

    branch_file = tmp_path / "spectra_branches" / "spectra_branches_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    branch_content = branch_file.read_text(encoding="utf-8")

    assert 'ZONE T="branch_001", I=2, DATAPACKING=POINT' in branch_content or 'ZONE T="branch_002", I=2, DATAPACKING=POINT' in branch_content
    assert "1.00000000e-01 3.00000000e-01 -2.00000000e+01" in branch_content
    assert "2.00000000e-01 3.60000000e-01 -2.80000000e+01" in branch_content


def test_spectra_process_applies_optional_alpha_gating_to_branch_tracking(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Apply optional alpha gating before tracking spectra branches across x."""
    monkeypatch.chdir(tmp_path)
    status_messages: list[str] = []

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    eig_file_a = case_dir_a / "Eigenvalues_case.dat"
    eig_file_a.write_text("10.0 1.0\n100.0 50.0\n", encoding="utf-8")

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    eig_file_b = case_dir_b / "Eigenvalues_case.dat"
    eig_file_b.write_text("11.0 1.2\n101.0 49.0\n", encoding="utf-8")

    cfg = Config.from_dict(
        {
            "processing": {
                "spectra": {
                    "alpr_min": 0.0,
                    "alpr_max": 20.0,
                    "alpi_min": 0.0,
                    "alpi_max": 10.0,
                    "branch_min_points": 2,
                },
            }
        }
    )

    out = spectra_process(cfg=cfg, reporter=status_messages.append)

    assert out == Path("spectra_animation")
    assert "alpha-space gating:" in status_messages
    assert any("alpha gating kept 2/4 eigenvalues" in message for message in status_messages)
    assert any("isolation score: 3-neighbor distance in normalized alpha-space" in message for message in status_messages)
    assert any("detachment score: nearest-neighbor gap normalized by nearby branch spacing" in message for message in status_messages)
    assert any("tracked 1 forward branch(es): f = 6.000 kHz, beta = 500.00" in message for message in status_messages)
    assert any("tracked 1 backward branch(es): f = 6.000 kHz, beta = 500.00" in message for message in status_messages)

    branch_file = tmp_path / "spectra_branches" / "spectra_branches_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    assert branch_file.exists()

    branch_content = branch_file.read_text(encoding="utf-8")
    assert 'ZONE T="branch_001", I=2, DATAPACKING=POINT' in branch_content
    assert "1.00000000e-01 1.00000000e+01 1.00000000e+00" in branch_content
    assert "2.00000000e-01 1.10000000e+01 1.20000000e+00" in branch_content


def test_spectra_process_writes_meaningful_isolation_scores_along_branches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Write larger isolation scores for branches that are separated from dense clusters."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.05 0.02\n0.10 0.10\n0.14 0.12\n10.00 10.00\n",
        encoding="utf-8",
    )

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text(
        "0.00 0.03\n0.05 0.05\n0.10 0.15\n0.14 0.17\n10.10 10.10\n",
        encoding="utf-8",
    )

    spectra_process(cfg=None)

    branch_file = tmp_path / "spectra_branches" / "spectra_branches_forward_f_0006pt00_khz_beta_pos0500pt00.dat"
    assert branch_file.exists()

    zones: dict[str, list[list[float]]] = {}
    current_zone = ""
    for line in branch_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("ZONE T="):
            current_zone = line.split('"')[1]
            zones[current_zone] = []
        elif line and not line.startswith("TITLE") and not line.startswith("VARIABLES"):
            zones[current_zone].append([float(value) for value in line.split()])

    zone_scores = {
        zone_name: max(row[3] for row in zone_rows)
        for zone_name, zone_rows in zones.items()
        if zone_rows
    }
    isolated_zone_name = max(zone_scores, key=zone_scores.get)
    clustered_zone_name = min(zone_scores, key=zone_scores.get)

    isolated_scores = [row[3] for row in zones[isolated_zone_name]]
    clustered_scores = [row[3] for row in zones[clustered_zone_name]]

    assert min(isolated_scores) > max(clustered_scores)


def test_spectra_process_handles_missing_eigenvalue_files(tmp_path: Path, monkeypatch) -> None:
    """Return current path marker when matching directories have no spectra files."""
    monkeypatch.chdir(tmp_path)

    # matching directory but no Eigenvalues_* file
    (tmp_path / "x_00pt20_m_f_0008pt00_khz_beta_neg0100pt00").mkdir()

    out = spectra_process(cfg=None)

    assert out == Path(".")


def test_spectra_process_can_run_only_branch_output(tmp_path: Path, monkeypatch) -> None:
    """Allow branch tracking to run without writing the raw animation output."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text("1.0 2.0\n3.0 4.0\n", encoding="utf-8")

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text("1.1 2.1\n3.1 4.1\n", encoding="utf-8")

    out = spectra_process(cfg=None, do_animate=False, do_branches=True, do_classify=False)

    assert out == Path("spectra_branches")
    assert not (tmp_path / "spectra_animation").exists()
    assert (tmp_path / "spectra_branches").exists()


def test_spectra_process_can_classify_isolated_branches(tmp_path: Path, monkeypatch) -> None:
    """Write only classified branches when the isolation criterion is configured."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.05 0.02\n0.10 0.10\n0.14 0.12\n10.00 10.00\n",
        encoding="utf-8",
    )

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text(
        "0.00 0.03\n0.05 0.05\n0.10 0.15\n0.14 0.17\n10.10 10.10\n",
        encoding="utf-8",
    )

    cfg = Config.from_dict(
        {
            "processing": {
                "spectra": {
                    "isolation_threshold": 1.0,
                    "classify_min_points": 2,
                },
            }
        }
    )

    out = spectra_process(
        cfg=cfg,
        do_animate=False,
        do_branches=False,
        do_classify=True,
    )

    assert out == Path("spectra_classified")
    classified_file = tmp_path / "spectra_classified" / "spectra_classified_f_0006pt00_khz_beta_pos0500pt00.dat"
    classified_contour_file = tmp_path / "spectra_classified" / "spectra_classified_contours_f_0006pt00_khz_beta_pos0500pt00.dat"
    classified_summary_file = tmp_path / "spectra_classified" / "spectra_classified_summary_f_0006pt00_khz_beta_pos0500pt00.csv"
    assert classified_file.exists()
    assert classified_contour_file.exists()
    assert classified_summary_file.exists()
    classified_content = classified_file.read_text(encoding="utf-8")
    classified_contour_content = classified_contour_file.read_text(encoding="utf-8")
    assert 'ZONE T="branch_001", I=2, DATAPACKING=POINT' in classified_content
    assert 'TITLE = "spectra_classified_contours"' in classified_contour_content
    assert 'ZONE T="classified_branches", I=2, J=1, DATAPACKING=POINT' in classified_contour_content
    assert "nan" not in classified_contour_content.lower()


def test_spectra_process_summary_reports_detachment_birth_and_absorption(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Write detachment history markers when a tracked branch peels off and returns."""
    monkeypatch.chdir(tmp_path)

    case_dir_a = tmp_path / "x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_a.mkdir()
    (case_dir_a / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.12 0.04\n",
        encoding="utf-8",
    )

    case_dir_b = tmp_path / "x_00pt20_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_b.mkdir()
    (case_dir_b / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.25 0.08\n",
        encoding="utf-8",
    )

    case_dir_c = tmp_path / "x_00pt30_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_c.mkdir()
    (case_dir_c / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.60 0.30\n",
        encoding="utf-8",
    )

    case_dir_d = tmp_path / "x_00pt40_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_d.mkdir()
    (case_dir_d / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.85 0.42\n",
        encoding="utf-8",
    )

    case_dir_e = tmp_path / "x_00pt50_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_e.mkdir()
    (case_dir_e / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n1.10 0.55\n",
        encoding="utf-8",
    )

    case_dir_f = tmp_path / "x_00pt60_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_f.mkdir()
    (case_dir_f / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.28 0.04\n",
        encoding="utf-8",
    )

    case_dir_g = tmp_path / "x_00pt70_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_g.mkdir()
    (case_dir_g / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.24 0.02\n",
        encoding="utf-8",
    )

    case_dir_h = tmp_path / "x_00pt80_m_f_0006pt00_khz_beta_pos0500pt00"
    case_dir_h.mkdir()
    (case_dir_h / "Eigenvalues_case.dat").write_text(
        "0.00 0.00\n0.10 0.00\n0.20 0.00\n0.21 0.01\n",
        encoding="utf-8",
    )

    cfg = Config.from_dict(
        {
            "processing": {
                "spectra": {
                    "branch_gate": 1.2,
                    "branch_min_points": 1,
                },
            }
        }
    )

    spectra_process(cfg=cfg, do_animate=False, do_branches=True, do_classify=False)

    branch_summary_file = tmp_path / "spectra_branches" / "spectra_branch_summary_forward_f_0006pt00_khz_beta_pos0500pt00.csv"
    summary_lines = branch_summary_file.read_text(encoding="utf-8").splitlines()

    assert len(summary_lines) >= 2
    header = summary_lines[0].split(",")
    first_row = summary_lines[1].split(",")
    row_map = dict(zip(header, first_row, strict=False))

    assert float(row_map["longest_detached_run"]) >= 3.0
    assert row_map["birth_x"] == "3.00000000e-01"
    assert row_map["absorption_x"] == "6.00000000e-01"


def test_spectra_process_classify_requires_threshold(tmp_path: Path, monkeypatch) -> None:
    """Require an explicit isolation threshold before classification can run."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="spectra classification requires"):
        spectra_process(cfg=None, do_animate=False, do_branches=False, do_classify=True)
