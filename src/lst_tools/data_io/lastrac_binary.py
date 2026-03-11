"""LASTRAC-specific Fortran binary I/O.

Extends the generic reader/writer with the record layout
used by the LASTRAC mean-flow binary format:
  header  -> title, n_station, igas, iunit, Pr, stat_pres, nsp
  station -> header (iloc, n_eta, s, ...) + vectors (eta, u, v, w, T, p)
"""

from __future__ import annotations

import struct
import numpy as np

from cfd_io import FortranBinaryReader, FortranBinaryWriter


# --------------------------------------------------
# LASTRAC reader
# --------------------------------------------------
class LastracReader(FortranBinaryReader):
    """Fortran binary reader with LASTRAC mean-flow record helpers."""

    def read_header(self) -> dict:
        """Read LASTRAC file header records.

        Returns a dict with keys:
            title, n_station, igas, iunit, Pr, stat_pres, nsp
        """
        title = self.read_string_fixed(length=64)
        n_station_arr = self.read_ints(expected_count=1)
        n_station = int(n_station_arr[0])
        combo = self._read_record_bytes()
        if len(combo) != struct.calcsize(self._endianness + "iiddi"):
            raise IOError("Header combo record length mismatch")
        igas, iunit, Pr, stat_pres, nsp = struct.unpack(
            self._endianness + "iiddi", combo
        )
        return {
            "title": title,
            "n_station": n_station,
            "igas": int(igas),
            "iunit": int(iunit),
            "Pr": float(Pr),
            "stat_pres": float(stat_pres),
            "nsp": int(nsp),
        }

    def read_station_header(self) -> dict:
        """Read LASTRAC station header (two records).

        Returns a dict with keys:
            i_loc, n_eta, s, lref, re1, kappa, rloc, drdx,
            stat_temp, stat_uvel, stat_dens
        """
        rec1 = self._read_record_bytes()
        if len(rec1) != struct.calcsize(self._endianness + "i i d d d d d d"):
            raise IOError("Station header rec1 length mismatch")
        i_loc, n_eta, s, lref, re1, kappa, rloc, drdx = struct.unpack(
            self._endianness + "i i d d d d d d", rec1
        )
        rec2 = self._read_record_bytes()
        if len(rec2) != struct.calcsize(self._endianness + "d d d"):
            raise IOError("Station header rec2 length mismatch")
        stat_temp, stat_uvel, stat_dens = struct.unpack(
            self._endianness + "d d d", rec2
        )
        return {
            "i_loc": int(i_loc),
            "n_eta": int(n_eta),
            "s": float(s),
            "lref": float(lref),
            "re1": float(re1),
            "kappa": float(kappa),
            "rloc": float(rloc),
            "drdx": float(drdx),
            "stat_temp": float(stat_temp),
            "stat_uvel": float(stat_uvel),
            "stat_dens": float(stat_dens),
        }

    def read_station_vector(self, *, count: int | None = None) -> np.ndarray:
        """Read a station vector (1-D array of reals).

        If *count* is given, validate the record length.
        """
        arr = self._read_numpy_record(self._real)
        if count is not None and arr.size != count:
            raise IOError(f"Unexpected vector length: got {arr.size}, expected {count}")
        return arr


# --------------------------------------------------
# LASTRAC writer
# --------------------------------------------------
class LastracWriter(FortranBinaryWriter):
    """Fortran binary writer with LASTRAC mean-flow record helpers."""

    def write_header(
        self,
        title: str,
        n_station: int,
        igas: int,
        iunit: int,
        Pr: float,
        stat_pres: float,
        nsp: int,
    ) -> None:
        """Write LASTRAC file header records."""
        self.write_string_fixed(title, length=64)
        self.write_ints([n_station])
        pack = struct.pack(
            self._endianness + "iiddi",
            int(igas),
            int(iunit),
            float(Pr),
            float(stat_pres),
            int(nsp),
        )
        self._write_record_bytes(pack)

    def write_station_header(
        self,
        *,
        i_loc: int,
        n_eta: int,
        s: float,
        lref: float,
        re1: float,
        kappa: float,
        rloc: float,
        drdx: float,
        stat_temp: float,
        stat_uvel: float,
        stat_dens: float,
    ) -> None:
        """Write LASTRAC station header (two records)."""
        rec1 = struct.pack(
            self._endianness + "i i d d d d d d",
            int(i_loc),
            int(n_eta),
            float(s),
            float(lref),
            float(re1),
            float(kappa),
            float(rloc),
            float(drdx),
        )
        self._write_record_bytes(rec1)
        rec2 = struct.pack(
            self._endianness + "d d d",
            float(stat_temp),
            float(stat_uvel),
            float(stat_dens),
        )
        self._write_record_bytes(rec2)

    def write_station_vector(
        self, vec: np.ndarray | list[float], *, count: int | None = None
    ) -> None:
        """Write a station vector (1-D array of reals) as one Fortran record."""
        a = np.asarray(vec, dtype=self._real)
        if count is not None:
            a = a[:count]
        self._write_numpy_record(a, order="C")
