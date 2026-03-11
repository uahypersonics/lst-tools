"""Tests for auto_fill_parsing in setup.parsing."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from lst_tools.config.schema import Config
from lst_tools.setup.parsing import auto_fill_parsing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    x_s=None,
    x_e=None,
    i_step=None,
    f_min=None,
    f_max=None,
    d_f=None,
    beta_s=None,
    beta_e=None,
    d_beta=None,
    baseflow_input="meanflow.bin",
    mach=6.0,
):
    """Build a minimal Config with specific LST params."""
    cfg = Config.from_dict({})
    cfg.lst.params.x_s = x_s
    cfg.lst.params.x_e = x_e
    cfg.lst.params.i_step = i_step
    cfg.lst.params.f_min = f_min
    cfg.lst.params.f_max = f_max
    cfg.lst.params.d_f = d_f
    cfg.lst.params.beta_s = beta_s
    cfg.lst.params.beta_e = beta_e
    cfg.lst.params.d_beta = d_beta
    cfg.lst.io.baseflow_input = baseflow_input
    cfg.flow_conditions.mach = mach
    return cfg


STATIONS_50 = np.linspace(0.1, 1.5, 50)
STATIONS_200 = np.linspace(0.05, 3.0, 200)
STATIONS_1 = np.array([0.42])


def _mock_profiles():
    """Build mock BL profiles with a clear edge at eta ≈ 0.005.

    h0 = T/(gamma-1) + 0.5*u² grows monotonically from wall to edge
    so that the 99% criterion finds a meaningful BL thickness.
    """
    eta = np.linspace(0.0, 0.01, 100)

    # velocity: tanh profile, saturates around eta ~ 0.003
    uvel = np.tanh(eta / 0.002)
    uvel = uvel / uvel[-1]  # normalise so u_e = 1

    # temperature: monotonically increasing so h0 grows wall → edge
    # (cool wall, hot freestream — exaggerated for a clear 99% crossing)
    temp = 0.5 + 0.5 * uvel

    vvel = np.zeros_like(eta)
    wvel = np.zeros_like(eta)

    return eta, uvel, vvel, wvel, temp


_eta, _uvel, _vvel, _wvel, _temp = _mock_profiles()

MOCK_SAMPLES = {
    "x": np.array([0.1, 0.5, 1.0]),
    "eta": [_eta] * 3,
    "uvel": [_uvel] * 3,
    "vvel": [_vvel] * 3,
    "wvel": [_wvel] * 3,
    "temp": [_temp] * 3,
    "stat_uvel": [1.0, 1.0, 1.0],
}

MOCK_SAMPLES_CROSSFLOW = {
    "x": np.array([0.1, 0.5, 1.0]),
    "eta": [_eta] * 3,
    "uvel": [_uvel] * 3,
    "vvel": [_vvel] * 3,
    "wvel": [np.ones(100) * 0.1] * 3,
    "temp": [_temp] * 3,
    "stat_uvel": [1.0, 1.0, 1.0],
}


# ---------------------------------------------------------------------------
# Tests: auto_fill_parsing
# ---------------------------------------------------------------------------


class TestAutoFillParsing:
    """Unit tests for auto_fill_parsing."""

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_fills_all_none_fields(self, mock_read, mock_samples, tmp_path):
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        changed = auto_fill_parsing(cfg)

        assert changed is True
        assert cfg.lst.params.x_s == pytest.approx(STATIONS_50[0])
        assert cfg.lst.params.x_e == pytest.approx(STATIONS_50[-1])
        assert cfg.lst.params.i_step == max(1, math.ceil(50 / 100))
        assert cfg.lst.params.f_max > 0.0
        assert cfg.lst.params.f_max != 1.0  # derived, not the old hardcoded default

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_200)
    def test_i_step_scales_with_stations(self, mock_read, mock_samples, tmp_path):
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg)

        assert cfg.lst.params.i_step == math.ceil(200 / 100)  # 2

    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_does_not_overwrite_existing_values(self, mock_read, tmp_path):
        cfg = _make_config(
            x_s=0.2,
            x_e=1.0,
            i_step=5,
            f_min=0.0,
            f_max=2.0,
            d_f=0.05,
            beta_s=0.0,
            beta_e=50.0,
            d_beta=5.0,
            baseflow_input=str(tmp_path / "meanflow.bin"),
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        changed = auto_fill_parsing(cfg)

        assert changed is False
        # values unchanged
        assert cfg.lst.params.x_s == 0.2
        assert cfg.lst.params.x_e == 1.0
        assert cfg.lst.params.i_step == 5
        assert cfg.lst.params.f_min == 0.0
        assert cfg.lst.params.f_max == 2.0
        assert cfg.lst.params.d_f == 0.05
        assert cfg.lst.params.beta_s == 0.0
        assert cfg.lst.params.beta_e == 50.0
        assert cfg.lst.params.d_beta == 5.0

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_fills_only_unset_fields(self, mock_read, mock_samples, tmp_path):
        """Only x_e is None; x_s and i_step are already set."""
        cfg = _make_config(
            x_s=0.3,
            x_e=None,
            i_step=2,
            baseflow_input=str(tmp_path / "meanflow.bin"),
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        changed = auto_fill_parsing(cfg)

        assert changed is True
        assert cfg.lst.params.x_s == 0.3  # unchanged
        assert cfg.lst.params.x_e == pytest.approx(STATIONS_50[-1])  # filled
        assert cfg.lst.params.i_step == 2  # unchanged

    def test_returns_false_when_meanflow_missing(self, tmp_path):
        """Space-sweep fields can't be derived, but freq/beta are still filled."""
        cfg = _make_config(baseflow_input=str(tmp_path / "nonexistent.bin"))

        changed = auto_fill_parsing(cfg)

        # freq/beta defaults were applied even though space-sweep failed
        assert changed is True
        assert cfg.lst.params.x_s is None  # still unset
        assert cfg.lst.params.f_min == 0.0
        assert cfg.lst.params.beta_e == 100.0

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=np.array([]))
    def test_returns_false_when_no_stations(self, mock_read, mock_samples, tmp_path):
        """Space-sweep fields can't be derived, but freq/beta are still filled."""
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        changed = auto_fill_parsing(cfg)

        # freq/beta defaults were applied even though space-sweep failed
        assert changed is True
        assert cfg.lst.params.x_s is None  # still unset
        assert cfg.lst.params.f_min == 0.0
        assert cfg.lst.params.beta_e == 100.0

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_1)
    def test_single_station(self, mock_read, mock_samples, tmp_path):
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg)

        assert cfg.lst.params.x_s == pytest.approx(0.42)
        assert cfg.lst.params.x_e == pytest.approx(0.42)
        assert cfg.lst.params.i_step == 1

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.write_config")
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_writes_config_back_when_cfg_path_given(self, mock_read, mock_write, mock_samples, tmp_path):
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")
        cfg_path = tmp_path / "lst.cfg"

        auto_fill_parsing(cfg, cfg_path=cfg_path)

        mock_write.assert_called_once()
        call_kwargs = mock_write.call_args
        assert call_kwargs.kwargs["path"] == cfg_path
        assert call_kwargs.kwargs["overwrite"] is True

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_no_write_when_cfg_path_is_none(self, mock_read, mock_samples, tmp_path):
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        with patch("lst_tools.setup.parsing.write_config") as mock_write:
            auto_fill_parsing(cfg, cfg_path=None)
            mock_write.assert_not_called()

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_idempotent(self, mock_read, mock_samples, tmp_path):
        """Running auto-fill twice should produce the same result."""
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg)
        x_s_1, x_e_1, i_step_1 = cfg.lst.params.x_s, cfg.lst.params.x_e, cfg.lst.params.i_step

        changed = auto_fill_parsing(cfg)

        assert changed is False
        assert cfg.lst.params.x_s == x_s_1
        assert cfg.lst.params.x_e == x_e_1
        assert cfg.lst.params.i_step == i_step_1

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_log_messages(self, mock_read, mock_samples, tmp_path, caplog):
        """Each auto-filled field should produce an [auto-fill] log line."""
        cfg = _make_config(baseflow_input=str(tmp_path / "meanflow.bin"))
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        import logging
        with caplog.at_level(logging.INFO, logger="lst_tools"):
            auto_fill_parsing(cfg)

        assert "[auto-fill] x_s" in caplog.text
        assert "[auto-fill] x_e" in caplog.text
        assert "[auto-fill] i_step" in caplog.text

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_f_max_estimated_from_baseflow(self, mock_read, mock_samples, tmp_path):
        """f_max should be derived from delta_99 and U_e, not hardcoded."""
        cfg = _make_config(
            baseflow_input=str(tmp_path / "meanflow.bin"),
            mach=6.0,
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg)

        # mach > 4: f_max = 0.9 * U_e / (2 * delta_99)
        assert cfg.lst.params.f_max > 0.0

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_f_max_uses_n_10_for_low_mach(self, mock_read, mock_samples, tmp_path):
        """For M <= 4, n=10 gives a smaller f_max than n=2."""
        cfg_hi = _make_config(
            baseflow_input=str(tmp_path / "meanflow.bin"), mach=6.0,
        )
        cfg_lo = _make_config(
            baseflow_input=str(tmp_path / "meanflow.bin"), mach=3.0,
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg_hi)
        auto_fill_parsing(cfg_lo)

        # lo-mach (n=10) should give f_max <= hi-mach (n=2)
        assert cfg_lo.lst.params.f_max <= cfg_hi.lst.params.f_max

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES_CROSSFLOW)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_f_max_fallback_on_crossflow(self, mock_read, mock_samples, tmp_path):
        """When wvel != 0, f_max falls back to 100000."""
        cfg = _make_config(
            baseflow_input=str(tmp_path / "meanflow.bin"), mach=6.0,
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg)

        assert cfg.lst.params.f_max == 100000

    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_f_max_fallback_when_mach_missing(self, mock_read, tmp_path):
        """When mach is None, f_max falls back to 100000."""
        cfg = _make_config(
            baseflow_input=str(tmp_path / "meanflow.bin"), mach=None,
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        auto_fill_parsing(cfg)

        assert cfg.lst.params.f_max == 100000

    @patch("lst_tools.setup.parsing.read_baseflow_profiles", return_value=MOCK_SAMPLES)
    @patch("lst_tools.setup.parsing.read_baseflow_stations", return_value=STATIONS_50)
    def test_force_overwrites_existing_values(self, mock_read, mock_samples, tmp_path):
        """With force=True, all fields are overwritten even if already set."""
        cfg = _make_config(
            x_s=0.2,
            x_e=1.0,
            i_step=5,
            f_min=10.0,
            f_max=2.0,
            d_f=0.05,
            beta_s=1.0,
            beta_e=50.0,
            d_beta=5.0,
            baseflow_input=str(tmp_path / "meanflow.bin"),
        )
        (tmp_path / "meanflow.bin").write_bytes(b"fake")

        changed = auto_fill_parsing(cfg, force=True)

        assert changed is True
        # x_s should now be from stations, not the original 0.2
        assert cfg.lst.params.x_s == pytest.approx(STATIONS_50[0])
        assert cfg.lst.params.x_e == pytest.approx(STATIONS_50[-1])
        # f_min forced back to 0.0
        assert cfg.lst.params.f_min == 0.0
        # beta defaults overwrite the originals
        assert cfg.lst.params.beta_s == 0.0
        assert cfg.lst.params.beta_e == 100.0
        assert cfg.lst.params.d_beta == 10.0
