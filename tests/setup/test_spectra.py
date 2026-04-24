"""Unit tests for spectra setup workflow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lst_tools.config.schema import Config
from lst_tools.setup import spectra as spectra_setup_mod


def test_build_case_name_formats_sign_and_units() -> None:
    """Format spectra case names with expected sign and unit encoding."""
    pos_name = spectra_setup_mod._build_case_name(0.1, 6000.0, 500.0)
    neg_name = spectra_setup_mod._build_case_name(0.1, 6000.0, -500.0)

    assert pos_name == "x_00pt10000_m_f_0006pt00_khz_beta_pos0500pt00"
    assert neg_name == "x_00pt10000_m_f_0006pt00_khz_beta_neg0500pt00"


def test_resolve_frequencies_builds_array() -> None:
    """Build frequency array from f_min/f_max/d_f."""
    cfg = Config()
    cfg.lst.params.f_min = 1000.0
    cfg.lst.params.f_max = 3000.0
    cfg.lst.params.d_f = 1000.0

    out = spectra_setup_mod._resolve_frequencies(cfg)

    assert np.allclose(out, np.array([1000.0, 2000.0, 3000.0]))


def test_resolve_frequencies_missing_values_raises() -> None:
    """Reject incomplete frequency range configuration."""
    cfg = Config()
    cfg.lst.params.f_min = 1000.0
    cfg.lst.params.f_max = None
    cfg.lst.params.d_f = 1000.0

    with pytest.raises(ValueError):
        spectra_setup_mod._resolve_frequencies(cfg)


def test_resolve_wavenumbers_builds_array() -> None:
    """Build wavenumber array from beta_s/beta_e/d_beta."""
    cfg = Config()
    cfg.lst.params.beta_s = -100.0
    cfg.lst.params.beta_e = 100.0
    cfg.lst.params.d_beta = 100.0

    out = spectra_setup_mod._resolve_wavenumbers(cfg)

    assert np.allclose(out, np.array([-100.0, 0.0, 100.0]))


def test_resolve_x_locations_uses_full_range_when_bounds_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use full baseflow range when x_s/x_e are not set."""
    baseflow_path = tmp_path / "meanflow.bin"
    baseflow_path.write_bytes(b"dummy")

    cfg = Config()
    cfg.lst.io.baseflow_input = str(baseflow_path)
    cfg.lst.params.x_s = None
    cfg.lst.params.x_e = None
    cfg.lst.params.i_step = 2

    monkeypatch.setattr(
        spectra_setup_mod,
        "read_baseflow_stations",
        lambda _: np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    )

    out = spectra_setup_mod._resolve_x_locations(cfg)

    assert np.allclose(out, np.array([0.1, 0.3, 0.5]))


def test_resolve_x_locations_missing_baseflow_file_raises(tmp_path: Path) -> None:
    """Reject spectra setup when baseflow input file is absent."""
    cfg = Config()
    cfg.lst.io.baseflow_input = str(tmp_path / "missing.bin")

    with pytest.raises(FileNotFoundError):
        spectra_setup_mod._resolve_x_locations(cfg)


def test_spectra_setup_generates_cases_and_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate all combinations and write launcher script."""
    monkeypatch.chdir(tmp_path)

    cfg = Config()
    cfg.lst.io.baseflow_input = "meanflow.bin"

    monkeypatch.setattr(spectra_setup_mod, "resolve_config", lambda _: cfg)
    monkeypatch.setattr(spectra_setup_mod, "_resolve_x_locations", lambda _: np.array([0.1, 0.2]))
    monkeypatch.setattr(spectra_setup_mod, "_resolve_frequencies", lambda _: np.array([1000.0]))
    monkeypatch.setattr(spectra_setup_mod, "_resolve_wavenumbers", lambda _: np.array([-50.0, 50.0]))

    written_inputs: list[Path] = []

    def _fake_setup_single_case(cfg_obj, cfg_spec, case_dir, input_path, baseflow_input):
        written_inputs.append(input_path)
        return input_path

    monkeypatch.setattr(spectra_setup_mod, "_setup_single_case", _fake_setup_single_case)

    env = SimpleNamespace(scheduler=SimpleNamespace(value="slurm"))
    monkeypatch.setattr(spectra_setup_mod, "detect", lambda: env)

    launcher_calls: list[tuple[list[str], str | None]] = []

    def _fake_write_launcher(case_dirs, script_name, submit_cmd):
        launcher_calls.append((case_dirs, submit_cmd))
        return Path(script_name)

    monkeypatch.setattr(spectra_setup_mod, "write_launcher_script", _fake_write_launcher)

    out = spectra_setup_mod.spectra_setup(cfg=cfg)

    assert len(out) == 4
    assert len(written_inputs) == 4
    assert len(launcher_calls) == 1
    assert launcher_calls[0][1] == "sbatch"


def test_spectra_setup_continues_when_case_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Continue setup when one case fails and report only successful outputs."""
    monkeypatch.chdir(tmp_path)

    cfg = Config()
    cfg.lst.io.baseflow_input = "meanflow.bin"

    monkeypatch.setattr(spectra_setup_mod, "resolve_config", lambda _: cfg)
    monkeypatch.setattr(spectra_setup_mod, "_resolve_x_locations", lambda _: np.array([0.1]))
    monkeypatch.setattr(spectra_setup_mod, "_resolve_frequencies", lambda _: np.array([1000.0, 2000.0]))
    monkeypatch.setattr(spectra_setup_mod, "_resolve_wavenumbers", lambda _: np.array([10.0]))

    counter = {"n": 0}

    def _fake_setup_single_case(cfg_obj, cfg_spec, case_dir, input_path, baseflow_input):
        counter["n"] += 1
        if counter["n"] == 1:
            raise OSError("boom")
        return input_path

    monkeypatch.setattr(spectra_setup_mod, "_setup_single_case", _fake_setup_single_case)

    env = SimpleNamespace(scheduler=SimpleNamespace(value="unknown"))
    monkeypatch.setattr(spectra_setup_mod, "detect", lambda: env)
    monkeypatch.setattr(
        spectra_setup_mod,
        "write_launcher_script",
        lambda *args, **kwargs: Path("run_cases.sh"),
    )

    out = spectra_setup_mod.spectra_setup(cfg=cfg)

    assert len(out) == 1
