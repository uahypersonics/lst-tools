"""Deep branch tests for tracking setup internals."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from lst_tools.config.schema import Config
from lst_tools.setup import tracking as tracking_mod


class _FakeTecplotData:
    """Minimal Tecplot-like object exposing field arrays used by tracking setup."""

    def __init__(
        self,
        s: np.ndarray,
        freq: np.ndarray,
        beta: np.ndarray,
        alpi: np.ndarray,
        alpr: np.ndarray,
    ) -> None:
        self._fields = {
            "s": s,
            "freq": freq,
            "beta": beta,
            "alpi": alpi,
            "alpr": alpr,
        }

    def field(self, name: str) -> np.ndarray:
        return self._fields[name]


def _build_tp(alpi_vals: np.ndarray) -> _FakeTecplotData:
    """Build a consistent fake Tecplot container for initial-guess tests."""
    # s: (K=1, J=1, I=3)
    s = np.array([[[0.0, 0.5, 1.0]]], dtype=float)
    # freq: (K=1, J=3, I=1)
    freq = np.array([[[100.0], [200.0], [300.0]]], dtype=float)
    # beta: (K=1, J=1, I=1)
    beta = np.array([[[25.0]]], dtype=float)
    # alpi/alpr: (Kbeta=1, Jfreq=3, Ix=3)
    alpi = alpi_vals[None, :, :].astype(float)
    alpr = np.full_like(alpi, 2.0)
    return _FakeTecplotData(s=s, freq=freq, beta=beta, alpi=alpi, alpr=alpr)


def test_find_initial_guess_with_finit_and_debug_output(tmp_path: Path) -> None:
    """Use finit branch, clamp x_ini, and write debug Tecplot file."""
    cfg = Config()
    cfg.flow_conditions.uvel_inf = 1000.0
    cfg.lst.params.tracking_dir = 1

    alpi_vals = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
        ]
    )
    tp = _build_tp(alpi_vals)

    with patch("lst_tools.data_io.tecplot_ascii.write_tecplot_ascii") as mock_write:
        guess = tracking_mod._find_initial_guess(
            tp=tp,
            x_ini=9.0,
            idx_betr=0,
            cfg=cfg,
            debug_path=tmp_path / "debug",
            finit=210.0,
        )

    assert guess["idx_x"] == 2
    assert guess["idx_f"] == 1
    assert guess["freq"] == 200.0
    assert mock_write.call_count == 1


def test_find_initial_guess_fallback_when_no_positive_growth() -> None:
    """Fall back to initial station when no positive alpi is found."""
    cfg = Config()
    cfg.flow_conditions.uvel_inf = 1000.0
    cfg.lst.params.tracking_dir = 1
    cfg.lst.params.f_min = 100.0
    cfg.lst.params.f_max = 300.0

    # all non-positive values force fallback path
    alpi_vals = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-2.0, -2.0, -2.0],
            [-3.0, -3.0, -3.0],
        ]
    )
    tp = _build_tp(alpi_vals)

    guess = tracking_mod._find_initial_guess(
        tp=tp,
        x_ini=0.5,
        idx_betr=0,
        cfg=cfg,
        debug_path=None,
        finit=None,
    )

    assert guess["idx_x"] == 1
    assert guess["idx_f"] == 0
    assert guess["freq"] == 100.0


def test_build_and_write_case_downstream_uses_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Downstream case uses computed node/time defaults when user did not set them."""
    cfg = Config()
    cfg.lst.params.tracking_dir = 0
    cfg.lst.params.x_s = None
    cfg.lst.params.x_e = None
    cfg.lst.params.i_step = None
    cfg.hpc.nodes = None
    cfg.hpc.time = None

    tp = _build_tp(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    x_baseflow = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    initial_guess = {"idx_x": 1, "idx_f": 1, "alpi": 1.0, "alpr": 2.0, "freq": 200.0}

    monkeypatch.setattr(tracking_mod, "detect", lambda: SimpleNamespace(cpus_per_node=4))

    captured = {}

    def _fake_hpc_configure(cfg_obj, *, set_defaults, nodes_override, time_override):
        captured["nodes_override"] = nodes_override
        captured["time_override"] = time_override
        return SimpleNamespace(scheduler="slurm", hostname="puma", fname_run_script="run.slurm.puma")

    monkeypatch.setattr(tracking_mod, "hpc_configure", _fake_hpc_configure)
    monkeypatch.setattr(tracking_mod, "script_build", lambda *args, **kwargs: None)

    generated = {}

    def _fake_generate_lst_input_deck(*, out_path, cfg):
        generated["out_path"] = out_path
        generated["cfg"] = cfg

    monkeypatch.setattr(tracking_mod, "generate_lst_input_deck", _fake_generate_lst_input_deck)

    hpc_cfg = tracking_mod._build_and_write_case(
        dir_name="kc_0010pt00",
        cfg=cfg,
        initial_guess=initial_guess,
        betr_loc=10.0,
        tp=tp,
        x_baseflow=x_baseflow,
        lst_exe="lst.x",
    )

    assert hpc_cfg.scheduler == "slurm"
    assert captured["nodes_override"] == 1
    assert captured["time_override"] == 0.5
    assert generated["out_path"] == Path("kc_0010pt00") / "lst_input.dat"
    # downstream tracking sets x_s to initial-guess location
    assert generated["cfg"].lst.params.x_s == 0.5


def test_build_and_write_case_raises_when_station_count_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raise ValueError when i_step is too large and no stations would be tracked."""
    cfg = Config()
    cfg.lst.params.tracking_dir = 1
    cfg.lst.params.x_s = 0.0
    cfg.lst.params.x_e = 1.0
    cfg.lst.params.i_step = 999

    tp = _build_tp(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    x_baseflow = np.array([0.0, 0.5, 1.0])
    initial_guess = {"idx_x": 1, "idx_f": 1, "alpi": 1.0, "alpr": 2.0, "freq": 200.0}

    with pytest.raises(ValueError, match="computed 0 tracking stations"):
        tracking_mod._build_and_write_case(
            dir_name="kc_0010pt00",
            cfg=cfg,
            initial_guess=initial_guess,
            betr_loc=10.0,
            tp=tp,
            x_baseflow=x_baseflow,
            lst_exe="lst.x",
        )


def test_auto_fill_tracking_updates_and_persists(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fill missing tracking params and persist config when path is provided."""
    cfg = Config()
    cfg.lst.params.beta_s = None
    cfg.lst.params.beta_e = None
    cfg.lst.params.d_beta = None
    cfg.lst.params.i_step = None

    captured = {}

    def _fake_write_config(*, path, overwrite, cfg_data):
        captured["path"] = path
        captured["overwrite"] = overwrite
        captured["cfg_data"] = cfg_data

    monkeypatch.setattr(tracking_mod, "write_config", _fake_write_config)

    out = tracking_mod.auto_fill_tracking(cfg, cfg_path=tmp_path / "lst.cfg")

    assert out is True
    assert cfg.lst.params.beta_s == 0.0
    assert cfg.lst.params.beta_e == 100.0
    assert cfg.lst.params.d_beta == 10.0
    assert cfg.lst.params.i_step == 1
    assert captured["overwrite"] is True


def test_auto_fill_tracking_no_changes_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do nothing when params are already set and force is False."""
    cfg = Config()
    cfg.lst.params.beta_s = 1.0
    cfg.lst.params.beta_e = 2.0
    cfg.lst.params.d_beta = 0.5
    cfg.lst.params.i_step = 3

    monkeypatch.setattr(tracking_mod, "write_config", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not write")))

    out = tracking_mod.auto_fill_tracking(cfg, force=False, cfg_path=None)
    assert out is False


def test_smooth_contour_field_zero_passes_returns_identity() -> None:
    """Return original field and all-True mask when npasses=0."""
    field = np.array([[1.0, 2.0], [3.0, 4.0]])
    smoothed, mask = tracking_mod.smooth_contour_field(field, npasses=0)

    assert np.array_equal(smoothed, field)
    assert np.array_equal(mask, np.ones(field.shape, dtype=bool))


def test_read_parsing_solution_uses_default_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use default parsing filename when fname is None and file exists."""
    fpath = tmp_path / "growth_rate_with_nfact_amps.dat"
    fpath.write_text("dummy", encoding="utf-8")

    captured = {}

    def _fake_read(path):
        captured["path"] = path
        return "tp"

    monkeypatch.setattr(tracking_mod, "read_tecplot_ascii", _fake_read)
    monkeypatch.chdir(tmp_path)

    out = tracking_mod._read_parsing_solution(None)

    assert out == "tp"
    assert captured["path"] == "growth_rate_with_nfact_amps.dat"


def test_setup_case_directory_formats_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Format kc directory name and call scaffold helper."""
    cfg = Config()
    cfg.lst_exe = "solver.x"

    captured = {}

    def _fake_scaffold(case_dir, meanflow_src, lst_exe):
        captured["case_dir"] = case_dir
        captured["meanflow_src"] = meanflow_src
        captured["lst_exe"] = lst_exe

    monkeypatch.setattr(tracking_mod, "scaffold_case_dir", _fake_scaffold)

    dir_name, exe = tracking_mod._setup_case_directory(12.5, cfg)

    assert dir_name == "kc_0012pt50"
    assert exe == "solver.x"
    assert captured["case_dir"] == "kc_0012pt50"


def test_find_initial_guess_downstream_positive_branch() -> None:
    """Take downstream search direction and exit loop on first positive alpi."""
    cfg = Config()
    cfg.flow_conditions.uvel_inf = 1000.0
    cfg.lst.params.tracking_dir = 0
    cfg.lst.params.f_min = 100.0
    cfg.lst.params.f_max = 300.0

    # positive value available at idx_x=0 and f index=1
    alpi_vals = np.array(
        [
            [0.0, -1.0, -1.0],
            [1.5, -1.0, -1.0],
            [0.0, -1.0, -1.0],
        ]
    )
    tp = _build_tp(alpi_vals)

    guess = tracking_mod._find_initial_guess(
        tp=tp,
        x_ini=-5.0,
        idx_betr=0,
        cfg=cfg,
        debug_path=None,
        finit=None,
    )

    assert guess["idx_x"] == 0
    assert guess["idx_f"] == 1


def test_build_and_write_case_upstream_user_nodes_and_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use upstream x-range and keep user-provided node/time overrides."""
    cfg = Config()
    cfg.lst.params.tracking_dir = 1
    cfg.lst.params.x_s = 0.0
    cfg.lst.params.x_e = 1.0
    cfg.lst.params.i_step = 1
    cfg.hpc.nodes = 1
    cfg.hpc.time = 2.0

    tp = _build_tp(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )
    x_baseflow = np.array([0.0, 0.5, 1.0])
    initial_guess = {"idx_x": 1, "idx_f": 1, "alpi": 1.0, "alpr": 2.0, "freq": 200.0}

    monkeypatch.setattr(tracking_mod, "detect", lambda: SimpleNamespace(cpus_per_node=4))
    monkeypatch.setattr(tracking_mod, "script_build", lambda *args, **kwargs: None)

    generated = {}

    def _fake_generate_lst_input_deck(*, out_path, cfg):
        generated["cfg"] = cfg

    monkeypatch.setattr(tracking_mod, "generate_lst_input_deck", _fake_generate_lst_input_deck)

    captured = {}

    def _fake_hpc_configure(cfg_obj, *, set_defaults, nodes_override, time_override):
        captured["nodes_override"] = nodes_override
        captured["time_override"] = time_override
        return SimpleNamespace(scheduler="pbs", hostname="carpenter", fname_run_script="run.pbs.carpenter")

    monkeypatch.setattr(tracking_mod, "hpc_configure", _fake_hpc_configure)

    tracking_mod._build_and_write_case(
        dir_name="kc_0010pt00",
        cfg=cfg,
        initial_guess=initial_guess,
        betr_loc=10.0,
        tp=tp,
        x_baseflow=x_baseflow,
        lst_exe="lst.x",
    )

    # upstream branch keeps x_s and sets x_e to initial guess location
    assert generated["cfg"].lst.params.x_s == 0.0
    assert generated["cfg"].lst.params.x_e == 0.5
    # user nodes/time retained when valid
    assert captured["nodes_override"] == 1
    assert captured["time_override"] == 2.0


def test_tracking_setup_auto_fill_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invoke tracking_setup with auto_fill=True and verify branch calls."""
    cfg = Config()
    cfg.geometry.theta_deg = None
    cfg.lst.params.tracking_dir = 1
    cfg.lst.params.x_e = None
    cfg.lst.params.x_s = 0.0

    tp = _build_tp(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        )
    )

    monkeypatch.setattr(tracking_mod, "resolve_config", lambda _: cfg)
    monkeypatch.setattr(tracking_mod, "auto_fill_tracking", lambda *args, **kwargs: True)
    monkeypatch.setattr(tracking_mod, "_read_parsing_solution", lambda _: tp)
    monkeypatch.setattr(tracking_mod, "_resolve_beta_values", lambda *_: np.array([10.0]))
    monkeypatch.setattr(tracking_mod, "_setup_case_directory", lambda *_: ("kc_0010pt00", "lst.x"))
    monkeypatch.setattr(tracking_mod, "_find_initial_guess", lambda *args, **kwargs: {"idx_x": 1, "idx_f": 1, "alpi": 1.0, "alpr": 2.0, "freq": 200.0})
    monkeypatch.setattr(tracking_mod, "_build_and_write_case", lambda *args, **kwargs: SimpleNamespace(scheduler="pbs", fname_run_script="run.pbs"))
    monkeypatch.setattr(tracking_mod, "write_launcher_script", lambda *args, **kwargs: Path("run_jobs.sh"))

    out = tracking_mod.tracking_setup(cfg=cfg, auto_fill=True, cfg_path="lst.cfg")

    assert out == Path("run_jobs.sh")
    assert cfg.geometry.theta_deg == 0.0
