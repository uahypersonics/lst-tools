"""Tests for read_baseflow_stations and read_baseflow_profiles.

Uses a real (synthetic) LASTRAC binary to exercise the readers end-to-end,
rather than mocking the I/O layer.
"""

import numpy as np
import pytest

from lst_tools.data_io import LastracWriter
from lst_tools.setup._common import read_baseflow_stations, read_baseflow_profiles


# --------------------------------------------------
# helper: write a small meanflow binary with known station coordinates
# --------------------------------------------------
def _write_meanflow(path, *, n_stations=10, n_eta=5):
    """Write a synthetic meanflow.bin and return expected station x-coords."""
    x_expected = np.linspace(0.01, 1.0, n_stations)

    writer = LastracWriter(path, endianness="<")
    writer.write_header(
        title="baseflow-test",
        n_station=n_stations,
        igas=1,
        iunit=2,
        Pr=0.71,
        stat_pres=101325.0,
        nsp=1,
    )

    for i in range(n_stations):
        # build station header
        writer.write_station_header(
            i_loc=i,
            n_eta=n_eta,
            s=float(x_expected[i]),
            lref=1.0,
            re1=1e6,
            kappa=0.0,
            rloc=0.5,
            drdx=0.0,
            stat_temp=300.0 + i,
            stat_uvel=100.0 + 10.0 * i,
            stat_dens=1.225,
        )

        # write 6 profile vectors: eta, u, v, w, temp, pres
        eta = np.linspace(0.0, 1.0, n_eta)
        u = np.sin(eta) + 0.1 * i
        v = np.zeros(n_eta)
        w = np.zeros(n_eta)
        temp = 300.0 + eta * 50.0
        pres = np.full(n_eta, 101325.0)

        for vec in (eta, u, v, w, temp, pres):
            writer.write_station_vector(vec)

    writer.close()
    return x_expected


# --------------------------------------------------
# tests for read_baseflow_stations
# --------------------------------------------------
class TestReadBaseflowStations:
    """Verify read_baseflow_stations against a synthetic binary."""

    def test_reads_all_stations(self, tmp_path):
        """All station x-coordinates are returned in order."""
        binfile = tmp_path / "meanflow.bin"
        x_expected = _write_meanflow(binfile, n_stations=8, n_eta=4)

        x = read_baseflow_stations(binfile)
        np.testing.assert_allclose(x, x_expected)

    def test_single_station(self, tmp_path):
        """File with one station produces a length-1 array."""
        binfile = tmp_path / "meanflow.bin"
        x_expected = _write_meanflow(binfile, n_stations=1, n_eta=3)

        x = read_baseflow_stations(binfile)
        assert x.shape == (1,)
        np.testing.assert_allclose(x, x_expected)

    def test_many_stations(self, tmp_path):
        """Larger file still roundtrips correctly."""
        n = 50
        binfile = tmp_path / "meanflow.bin"
        x_expected = _write_meanflow(binfile, n_stations=n, n_eta=6)

        x = read_baseflow_stations(binfile)
        assert x.shape == (n,)
        np.testing.assert_allclose(x, x_expected)


# --------------------------------------------------
# tests for read_baseflow_profiles
# --------------------------------------------------
class TestReadBaseflowProfiles:
    """Verify read_baseflow_profiles against a synthetic binary."""

    def test_returns_expected_keys(self, tmp_path):
        """Result dict contains all expected keys."""
        binfile = tmp_path / "meanflow.bin"
        _write_meanflow(binfile, n_stations=5, n_eta=4)

        result = read_baseflow_profiles(binfile)
        expected_keys = {"x", "eta", "uvel", "vvel", "wvel", "temp", "stat_uvel"}
        assert expected_keys <= set(result.keys())

    def test_samples_all_when_fewer_than_n_samples(self, tmp_path):
        """When n_stations < n_samples, all stations are read."""
        n = 5
        binfile = tmp_path / "meanflow.bin"
        x_expected = _write_meanflow(binfile, n_stations=n, n_eta=4)

        result = read_baseflow_profiles(binfile, n_samples=50)
        np.testing.assert_allclose(result["x"], x_expected)
        assert len(result["eta"]) == n
        assert len(result["uvel"]) == n

    def test_subsamples_when_many_stations(self, tmp_path):
        """When n_stations > n_samples, only n_samples are read."""
        n = 50
        n_samples = 10
        binfile = tmp_path / "meanflow.bin"
        _write_meanflow(binfile, n_stations=n, n_eta=4)

        result = read_baseflow_profiles(binfile, n_samples=n_samples)
        assert len(result["x"]) == n_samples
        assert len(result["eta"]) == n_samples
        assert len(result["uvel"]) == n_samples

    def test_profile_shapes(self, tmp_path):
        """Each profile vector has the correct wall-normal dimension."""
        n_eta = 8
        binfile = tmp_path / "meanflow.bin"
        _write_meanflow(binfile, n_stations=5, n_eta=n_eta)

        result = read_baseflow_profiles(binfile)
        for eta_arr in result["eta"]:
            assert eta_arr.shape == (n_eta,)
        for u_arr in result["uvel"]:
            assert u_arr.shape == (n_eta,)

    def test_stat_uvel_values(self, tmp_path):
        """Static u-velocity from station headers is captured."""
        n = 5
        binfile = tmp_path / "meanflow.bin"
        _write_meanflow(binfile, n_stations=n, n_eta=4)

        result = read_baseflow_profiles(binfile)
        # from _write_meanflow: stat_uvel = 100.0 + 10.0 * i
        expected_stat_uvel = [100.0 + 10.0 * i for i in range(n)]
        np.testing.assert_allclose(result["stat_uvel"], expected_stat_uvel)
