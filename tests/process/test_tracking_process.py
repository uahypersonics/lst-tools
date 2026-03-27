"""Unit tests for tracking post-processing orchestration."""

from __future__ import annotations

from pathlib import Path

from lst_tools.config.schema import Config
from lst_tools.process.tracking import tracking_process


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
    cfg.processing.interpolate = True
    cfg.processing.gate_tol = 0.25
    cfg.processing.min_valid = 12

    maxima_calls: list[tuple[Path, bool, float, int]] = []

    def _fake_extract_maxima(kc_dir: Path, *, interpolate: bool, gate_tol: float, min_valid: int):
        maxima_calls.append((kc_dir, interpolate, gate_tol, min_valid))
        return [kc_dir / "alpi_max_mode_001.dat"]

    monkeypatch.setattr("lst_tools.process.tracking.extract_maxima", _fake_extract_maxima)
    monkeypatch.setattr(
        "lst_tools.process.tracking.assemble_volume",
        lambda work_dir: work_dir / "lst_vol.dat",
    )

    # execute
    out = tracking_process(cfg=cfg, work_dir=tmp_path)

    # validate
    assert out == tmp_path
    assert len(maxima_calls) == 2
    assert maxima_calls[0][1:] == (True, 0.25, 12)
    assert maxima_calls[1][1:] == (True, 0.25, 12)


def test_tracking_process_interpolate_override_and_volume_no_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Allow CLI interpolate override and handle empty volume output."""
    # build
    kc = tmp_path / "kc_0000pt00"
    kc.mkdir()

    cfg = Config()
    cfg.processing.interpolate = True
    cfg.processing.gate_tol = 0.11
    cfg.processing.min_valid = 7

    captured: dict[str, object] = {}

    def _fake_extract_maxima(kc_dir: Path, *, interpolate: bool, gate_tol: float, min_valid: int):
        captured["interpolate"] = interpolate
        captured["gate_tol"] = gate_tol
        captured["min_valid"] = min_valid
        return []

    monkeypatch.setattr("lst_tools.process.tracking.extract_maxima", _fake_extract_maxima)
    monkeypatch.setattr("lst_tools.process.tracking.assemble_volume", lambda _: None)

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
