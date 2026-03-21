"""Roundtrip tests for LastracWriter / LastracReader.

Write a synthetic meanflow binary with known values, read it back,
and verify every field survives the trip.
"""

import numpy as np
import pytest

from lst_tools.data_io import LastracWriter, LastracReader


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _write_synthetic_binary(path, *, n_stations=3, n_eta=5):
    """Write a small LASTRAC binary with deterministic data.

    Returns the ground-truth dict so the reader can be verified.
    """
    truth = {
        "title": "roundtrip-test",
        "n_station": n_stations,
        "igas": 1,
        "iunit": 2,
        "Pr": 0.71,
        "stat_pres": 101325.0,
        "nsp": 1,
        "stations": [],
    }

    writer = LastracWriter(path, endianness="<")

    # write file header
    writer.write_header(
        title=truth["title"],
        n_station=truth["n_station"],
        igas=truth["igas"],
        iunit=truth["iunit"],
        Pr=truth["Pr"],
        stat_pres=truth["stat_pres"],
        nsp=truth["nsp"],
    )

    # write station data
    for i in range(n_stations):
        stn = {
            "i_loc": i,
            "n_eta": n_eta,
            "s": 0.1 * (i + 1),
            "lref": 1.0,
            "re1": 1e6,
            "kappa": 0.01 * i,
            "rloc": 0.5 + 0.1 * i,
            "drdx": 0.001 * i,
            "stat_temp": 300.0 + i,
            "stat_uvel": 100.0 + 10.0 * i,
            "stat_dens": 1.225 - 0.01 * i,
        }
        writer.write_station_header(**stn)

        # build deterministic vectors for this station
        eta = np.linspace(0.0, 1.0, n_eta)
        u = np.sin(eta) + 0.1 * i
        v = np.cos(eta) * 0.01
        w = np.zeros(n_eta)
        temp = 300.0 + eta * 50.0
        pres = np.full(n_eta, 101325.0)

        for vec in (eta, u, v, w, temp, pres):
            writer.write_station_vector(vec)

        stn["vectors"] = {
            "eta": eta,
            "uvel": u,
            "vvel": v,
            "wvel": w,
            "temp": temp,
            "pres": pres,
        }
        truth["stations"].append(stn)

    writer.close()
    return truth


# --------------------------------------------------
# tests
# --------------------------------------------------
class TestLastracBinaryRoundtrip:
    """Write → read roundtrip for the LASTRAC binary format."""

    def test_header_roundtrip(self, tmp_path):
        """File header survives a write → read cycle."""
        binfile = tmp_path / "meanflow.bin"
        truth = _write_synthetic_binary(binfile)

        reader = LastracReader(binfile, endianness="<")
        header = reader.read_header()
        reader.close()

        assert header["title"].strip() == truth["title"]
        assert header["n_station"] == truth["n_station"]
        assert header["igas"] == truth["igas"]
        assert header["iunit"] == truth["iunit"]
        np.testing.assert_allclose(header["Pr"], truth["Pr"])
        np.testing.assert_allclose(header["stat_pres"], truth["stat_pres"])
        assert header["nsp"] == truth["nsp"]

    def test_station_headers_roundtrip(self, tmp_path):
        """Station header scalars survive a write → read cycle."""
        binfile = tmp_path / "meanflow.bin"
        truth = _write_synthetic_binary(binfile, n_stations=4, n_eta=8)

        reader = LastracReader(binfile, endianness="<")
        reader.read_header()

        for stn_truth in truth["stations"]:
            stn = reader.read_station_header()
            assert stn["i_loc"] == stn_truth["i_loc"]
            assert stn["n_eta"] == stn_truth["n_eta"]
            np.testing.assert_allclose(stn["s"], stn_truth["s"])
            np.testing.assert_allclose(stn["lref"], stn_truth["lref"])
            np.testing.assert_allclose(stn["re1"], stn_truth["re1"])
            np.testing.assert_allclose(stn["kappa"], stn_truth["kappa"])
            np.testing.assert_allclose(stn["rloc"], stn_truth["rloc"])
            np.testing.assert_allclose(stn["drdx"], stn_truth["drdx"])
            np.testing.assert_allclose(stn["stat_temp"], stn_truth["stat_temp"])
            np.testing.assert_allclose(stn["stat_uvel"], stn_truth["stat_uvel"])
            np.testing.assert_allclose(stn["stat_dens"], stn_truth["stat_dens"])

            # skip the 6 vectors
            for _ in range(6):
                reader.read_station_vector()

        reader.close()

    def test_station_vectors_roundtrip(self, tmp_path):
        """Station profile vectors survive a write → read cycle."""
        n_eta = 10
        binfile = tmp_path / "meanflow.bin"
        truth = _write_synthetic_binary(binfile, n_stations=2, n_eta=n_eta)

        reader = LastracReader(binfile, endianness="<")
        reader.read_header()

        for stn_truth in truth["stations"]:
            reader.read_station_header()
            vecs = stn_truth["vectors"]

            eta = reader.read_station_vector()
            np.testing.assert_allclose(eta, vecs["eta"])

            u = reader.read_station_vector()
            np.testing.assert_allclose(u, vecs["uvel"])

            v = reader.read_station_vector()
            np.testing.assert_allclose(v, vecs["vvel"])

            w = reader.read_station_vector()
            np.testing.assert_allclose(w, vecs["wvel"])

            temp = reader.read_station_vector()
            np.testing.assert_allclose(temp, vecs["temp"])

            pres = reader.read_station_vector()
            np.testing.assert_allclose(pres, vecs["pres"])

        reader.close()

    def test_single_station(self, tmp_path):
        """Edge case: file with exactly one station."""
        binfile = tmp_path / "meanflow.bin"
        truth = _write_synthetic_binary(binfile, n_stations=1, n_eta=3)

        reader = LastracReader(binfile, endianness="<")
        header = reader.read_header()
        assert header["n_station"] == 1

        stn = reader.read_station_header()
        assert stn["i_loc"] == 0
        assert stn["n_eta"] == 3

        eta = reader.read_station_vector()
        assert eta.size == 3

        reader.close()
