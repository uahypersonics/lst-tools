"""Unit tests for tracking post-processing orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from lst_tools.cli.app import cli
from lst_tools.config.schema import Config
from lst_tools.process.modes import (
    ModeCurve,
    load_mode_curves,
    match_mode_curves,
    process_modes,
)
from lst_tools.process.tracking import tracking_process


def test_process_modes_cli_reports_missing_ridges(tmp_path: Path) -> None:
    """An empty work directory must not create an output directory."""
    result = CliRunner().invoke(cli, ["process", "modes", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "no nfac_max_mode_* or alpi_max_mode_* curves" in result.output
    assert not (tmp_path / "mode_surfaces").exists()


def test_mode_matching_follows_frequency_through_local_label_swap() -> None:
    """Different folder numbers can still represent the same physical ridge."""
    stations = np.linspace(0.05, 0.15, 20)
    curves_by_beta = {
        14.0: [
            ModeCurve(
                Path("mode_001/kc_0014pt00.dat"),
                14.0,
                stations,
                np.full(20, 70000.0),
                stations,
            ),
            ModeCurve(
                Path("mode_002/kc_0014pt00.dat"),
                14.0,
                stations,
                np.full(20, 2000.0),
                stations,
            ),
        ],
        15.0: [
            ModeCurve(
                Path("mode_001/kc_0015pt00.dat"),
                15.0,
                stations,
                np.full(20, 2000.0),
                stations,
            ),
            ModeCurve(
                Path("mode_002/kc_0015pt00.dat"),
                15.0,
                stations,
                np.full(20, 70000.0),
                stations,
            ),
        ],
    }

    assignments = match_mode_curves(curves_by_beta)

    assert [
        (assignment.curve.path.parent.name, assignment.mode)
        for assignment in assignments
    ] == [
        ("mode_001", 1),
        ("mode_002", 2),
        ("mode_001", 2),
        ("mode_002", 1),
    ]


def test_mode_matching_leaves_ambiguous_frequencies_unassigned() -> None:
    """Two near-identical candidate branches must not be stitched by folder order."""
    stations = np.linspace(0.05, 0.15, 20)
    curves = {
        5.0: [
            ModeCurve(
                Path(f"first_{index}.dat"),
                5.0,
                stations,
                np.full(20, frequency),
                stations,
            )
            for index, frequency in enumerate((20000.0, 20200.0))
        ],
        6.0: [
            ModeCurve(Path("next.dat"), 6.0, stations, np.full(20, 20100.0), stations)
        ],
    }

    assignments = match_mode_curves(curves)

    assert assignments[-1].mode == 3
    assert assignments[-1].match_cost is None


def test_load_mode_curves_reads_existing_tecplot_ridge_format(tmp_path: Path) -> None:
    """Ridge ingestion uses the existing Tecplot parser and canonical field aliases."""
    from lst_tools.data_io import write_tecplot_ascii

    mode_dir = tmp_path / "nfac_max_mode_001"
    mode_dir.mkdir()
    write_tecplot_ascii(
        mode_dir / "kc_0005pt00.dat",
        {
            "s": np.linspace(0.1, 0.2, 6),
            "freq.": np.full(6, 20000.0),
            "Beta": np.full(6, 5.0),
            "re(alpha)": np.linspace(100.0, 110.0, 6),
            "Nfac": np.linspace(0.1, 1.0, 6),
            "Nfac3": np.linspace(0.2, 2.0, 6),
        },
    )

    curves = load_mode_curves(tmp_path, "nfac")

    assert list(curves) == [5.0]
    np.testing.assert_allclose(curves[5.0][0].value, np.linspace(0.2, 2.0, 6))
    np.testing.assert_allclose(
        curves[5.0][0].fields["re(alpha)"], np.linspace(100.0, 110.0, 6)
    )


def test_process_modes_writes_matched_surface_with_masked_zero_padding(
    tmp_path: Path,
) -> None:
    """Local label changes do not alter IDs; unsupported x is zero with valid=0."""
    from lst_tools.data_io import read_tecplot_ascii, write_tecplot_ascii

    for local_mode, beta, frequency, s_values in (
        (1, 5.0, 70000.0, np.linspace(0.1, 0.2, 12)),
        (2, 5.0, 2000.0, np.linspace(0.12, 0.19, 12)),
        (1, 6.0, 2000.0, np.linspace(0.13, 0.2, 10)),
        (2, 6.0, 70000.0, np.linspace(0.13, 0.2, 10)),
    ):
        mode_dir = tmp_path / f"nfac_max_mode_{local_mode:03d}"
        mode_dir.mkdir(exist_ok=True)
        if frequency == 70000.0 and beta == 5.0:
            nfac3 = 3.0 - 10.0 * s_values
            nfac = np.full(s_values.size, 100.0)
        else:
            nfac3 = 10.0 * s_values
            nfac = np.ones(s_values.size)
        alpha_real = np.full(s_values.size, 100.0 + beta)
        amplitude = np.full(s_values.size, beta / 10.0)
        write_tecplot_ascii(
            mode_dir / f"kc_{beta:07.2f}".replace(".", "pt").__add__(".dat"),
            {
                "s": s_values,
                "freq.": np.full(s_values.size, frequency),
                "Beta": np.full(s_values.size, beta),
                "re(alpha)": alpha_real,
                "Amp": amplitude,
                "Nfac": nfac,
                "Nfac3": nfac3,
            },
        )

    output_dir = tmp_path / "mode_surfaces"
    written = process_modes(tmp_path, output_dir)
    surface = read_tecplot_ascii(output_dir / "nfac_max_mode_001.dat")
    envelope = read_tecplot_ascii(output_dir / "nfac_envelope_mode_001.dat")
    unsupported_envelope = read_tecplot_ascii(output_dir / "nfac_envelope_mode_002.dat")
    report = (output_dir / "mode_assignments.csv").read_text(encoding="utf-8")

    assert len(written) == 4
    assert "nfac_max_mode_002/kc_0006pt00.dat" in report
    assert surface.zone.I == 500
    np.testing.assert_allclose(surface.field("s")[0, 0, :], np.linspace(0.1, 0.2, 500))
    assert surface.field("nfac")[0, 1, 0] == 0.0
    assert surface.field("freq")[0, 1, 0] == 0.0
    assert surface.field("valid")[0, 1, 0] == 0.0
    assert surface.field("nfac3")[0, 1, -1] > 0
    assert surface.field("alpr")[0, 1, -1] == 106.0
    assert surface.field("ampl")[0, 1, -1] == 0.6
    assert surface.field("valid")[0, 1, -1] == 1.0
    assert envelope.field("beta")[0, 0, 0] == 5.0
    assert envelope.field("alpr")[0, 0, 0] == 105.0
    assert envelope.field("ampl")[0, 0, 0] == 0.5
    assert envelope.field("beta")[0, 0, -1] == 6.0
    assert envelope.field("freq")[0, 0, -1] == 70000.0
    assert envelope.field("nfac")[0, 0, -1] == 1.0
    assert envelope.field("nfac3")[0, 0, -1] == 2.0
    assert envelope.field("alpr")[0, 0, -1] == 106.0
    assert envelope.field("ampl")[0, 0, -1] == 0.6
    assert envelope.field("valid")[0, 0, -1] == 1.0
    assert unsupported_envelope.field("nfac")[0, 0, 0] == 0.0
    assert unsupported_envelope.field("beta")[0, 0, 0] == 0.0
    assert unsupported_envelope.field("valid")[0, 0, 0] == 0.0

    runner = CliRunner()
    cli_args = ["process", "modes", "--dir", str(tmp_path), "--out", "cli_surfaces"]
    cli_result = runner.invoke(cli, cli_args)
    repeated_result = runner.invoke(cli, cli_args)

    assert cli_result.exit_code == 0
    assert "wrote 2 maximum surface(s) and 2 envelope(s)" in cli_result.output
    assert (tmp_path / "cli_surfaces" / "mode_assignments.csv").is_file()
    assert repeated_result.exit_code == 1
    assert "use --force to replace it" in repeated_result.output

    stale_path = tmp_path / "cli_surfaces" / "stale.dat"
    stale_path.write_text("stale", encoding="utf-8")
    forced_result = runner.invoke(cli, [*cli_args, "--force"])

    assert forced_result.exit_code == 0
    assert not stale_path.exists()
    assert (tmp_path / "cli_surfaces" / "nfac_max_mode_001.dat").is_file()

    coarse_args = [
        "process",
        "modes",
        "--dir",
        str(tmp_path),
        "--out",
        "coarse_surfaces",
        "--nx",
        "25",
    ]
    coarse_result = runner.invoke(cli, coarse_args)
    assert coarse_result.exit_code == 0
    coarse_surface = read_tecplot_ascii(
        tmp_path / "coarse_surfaces" / "nfac_max_mode_001.dat"
    )
    assert coarse_surface.zone.I == 25


def test_tracking_process_no_kc_dirs_returns_workdir(tmp_path: Path) -> None:
    """Return work_dir unchanged when no kc_* directories are present."""
    out = tracking_process(work_dir=tmp_path)
    assert out == tmp_path


def test_tracking_process_runs_maxima_and_volume_with_config_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run both steps and pass config-derived maxima settings."""
    # build
    kc1 = tmp_path / "kc_0000pt00"
    kc2 = tmp_path / "kc_0010pt00"
    kc1.mkdir()
    kc2.mkdir()

    cfg = Config()
    cfg.processing.tracking.interpolate = True
    cfg.processing.tracking.gate_tol = 0.25
    cfg.processing.tracking.min_valid = 12

    maxima_calls: list[tuple[Path, bool, float, int, int, Path | None]] = []

    def _fake_extract_maxima(
        kc_dir: Path,
        *,
        interpolate: bool,
        gate_tol: float,
        min_valid: int,
        peak_order: int = 1,
        mode_root_dir: Path | None = None,
    ):
        maxima_calls.append(
            (kc_dir, interpolate, gate_tol, min_valid, peak_order, mode_root_dir)
        )
        return [kc_dir / "alpi_max_mode_001.dat"]

    monkeypatch.setattr(
        "lst_tools.process.tracking.extract_maxima", _fake_extract_maxima
    )
    monkeypatch.setattr(
        "lst_tools.process.tracking.assemble_volume",
        lambda work_dir, plain_output=False: work_dir / "lst_vol.dat",
    )

    # execute
    out = tracking_process(cfg=cfg, work_dir=tmp_path)

    # validate
    assert out == tmp_path
    assert len(maxima_calls) == 2
    assert maxima_calls[0][1:] == (True, 0.25, 12, 1, tmp_path)
    assert maxima_calls[1][1:] == (True, 0.25, 12, 1, tmp_path)


def test_tracking_process_interpolate_override_and_volume_no_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Allow CLI interpolate override and handle empty volume output."""
    # build
    kc = tmp_path / "kc_0000pt00"
    kc.mkdir()

    cfg = Config()
    cfg.processing.tracking.interpolate = True
    cfg.processing.tracking.gate_tol = 0.11
    cfg.processing.tracking.min_valid = 7

    captured: dict[str, object] = {}

    def _fake_extract_maxima(
        kc_dir: Path,
        *,
        interpolate: bool,
        gate_tol: float,
        min_valid: int,
        peak_order: int = 1,
        mode_root_dir: Path | None = None,
    ):
        captured["interpolate"] = interpolate
        captured["gate_tol"] = gate_tol
        captured["min_valid"] = min_valid
        captured["peak_order"] = peak_order
        captured["mode_root_dir"] = mode_root_dir
        return []

    monkeypatch.setattr(
        "lst_tools.process.tracking.extract_maxima", _fake_extract_maxima
    )
    monkeypatch.setattr(
        "lst_tools.process.tracking.assemble_volume",
        lambda _, plain_output=False: None,
    )

    # execute
    tracking_process(
        cfg=cfg,
        work_dir=tmp_path,
        interpolate=False,
        do_maxima=True,
        do_volume=True,
    )

    # validate
    assert captured == {
        "interpolate": False,
        "gate_tol": 0.11,
        "min_valid": 7,
        "peak_order": 1,
        "mode_root_dir": tmp_path,
    }


def test_tracking_process_volume_only_mode(tmp_path: Path, monkeypatch) -> None:
    """Skip maxima when do_maxima is False."""
    kc = tmp_path / "kc_0000pt00"
    kc.mkdir()

    called = {"maxima": 0, "volume": 0}

    def _fake_extract(*args, **kwargs):
        called["maxima"] += 1
        return []

    def _fake_volume(*args, **kwargs):
        called["volume"] += 1
        return tmp_path / "lst_vol.dat"

    monkeypatch.setattr("lst_tools.process.tracking.extract_maxima", _fake_extract)
    monkeypatch.setattr("lst_tools.process.tracking.assemble_volume", _fake_volume)

    tracking_process(work_dir=tmp_path, do_maxima=False, do_volume=True)

    assert called["maxima"] == 0
    assert called["volume"] == 1


def test_tracking_process_plain_output_forwarded(tmp_path: Path, monkeypatch) -> None:
    """Forward plain_output flag to volume assembly."""
    kc = tmp_path / "kc_0000pt00"
    kc.mkdir()

    captured: dict[str, object] = {}

    def _fake_volume(work_dir: Path, plain_output: bool = False):
        captured["work_dir"] = work_dir
        captured["plain_output"] = plain_output
        return tmp_path / "lst_vol.dat"

    monkeypatch.setattr(
        "lst_tools.process.tracking.extract_maxima", lambda *args, **kwargs: []
    )
    monkeypatch.setattr("lst_tools.process.tracking.assemble_volume", _fake_volume)

    tracking_process(
        work_dir=tmp_path, do_maxima=False, do_volume=True, plain_output=True
    )

    assert captured == {
        "work_dir": tmp_path,
        "plain_output": True,
    }


def test_tracking_process_maxima_uses_progress_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Use progress context for maxima when plain_output is False."""
    kc1 = tmp_path / "kc_0000pt00"
    kc2 = tmp_path / "kc_0010pt00"
    kc1.mkdir()
    kc2.mkdir()

    captured: dict[str, object] = {
        "total": None,
        "desc": None,
        "persist": None,
        "advance_calls": 0,
    }

    class _FakeProgressCtx:
        def __enter__(self):
            def _advance(n: int = 1):
                captured["advance_calls"] += int(n)

            return _advance

        def __exit__(self, exc_type, exc, tb):
            return None

    def _fake_progress(
        *, total: int, desc=None, description=None, persist: bool = True, **kwargs
    ):
        captured["total"] = total
        captured["desc"] = desc if desc is not None else description
        captured["persist"] = persist
        return _FakeProgressCtx()

    monkeypatch.setattr("lst_tools.process.tracking.progress", _fake_progress)
    monkeypatch.setattr(
        "lst_tools.process.tracking.extract_maxima", lambda *args, **kwargs: []
    )

    tracking_process(
        work_dir=tmp_path, do_maxima=True, do_volume=False, plain_output=False
    )

    assert captured["total"] == 2
    assert captured["desc"] == "process.tracking.maxima"
    assert captured["persist"] is True
    assert captured["advance_calls"] == 2


def test_tracking_process_maxima_plain_output_prints_dirs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Print per-directory lines for maxima when plain_output is True."""
    kc1 = tmp_path / "kc_0000pt00"
    kc2 = tmp_path / "kc_0010pt00"
    kc1.mkdir()
    kc2.mkdir()

    monkeypatch.setattr(
        "lst_tools.process.tracking.extract_maxima", lambda *args, **kwargs: []
    )

    tracking_process(
        work_dir=tmp_path, do_maxima=True, do_volume=False, plain_output=True
    )

    out = capsys.readouterr().out
    assert "- kc_0000pt00" in out
    assert "- kc_0010pt00" in out
