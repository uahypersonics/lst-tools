"""Additional coverage tests for tecplot_ascii read/write utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lst_tools.data_io.tecplot_ascii import TecplotData, TecplotZone, read_tecplot_ascii, write_tecplot_ascii


class TestTecplotDataExtra:
    """Target low-frequency branches in TecplotData helpers."""

    def test_debug_aliases_without_explicit_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exercise debug_aliases default stdout branch."""
        zone = TecplotZone("z", I=1, J=1, K=1, datapacking="POINT", dt=[])
        data = np.zeros((1, 1, 1, 1))
        tp = TecplotData(
            title="t",
            variables=["x"],
            zone=zone,
            data=data,
            var_index={"x": 0},
        )

        printed = []
        monkeypatch.setattr("builtins.print", lambda *args, **kwargs: printed.append(args[0] if args else ""))

        tp.debug_aliases()

        assert any("headers" in str(line).lower() for line in printed)

    def test_field_raises_when_resolved_header_missing_in_var_index(self) -> None:
        """Exercise defensive KeyError for inconsistent var_index maps."""
        zone = TecplotZone("z", I=1, J=1, K=1, datapacking="POINT", dt=[])
        data = np.zeros((1, 1, 1, 1))
        tp = TecplotData(
            title="t",
            variables=["x"],
            zone=zone,
            data=data,
            var_index={},
        )

        with pytest.raises(KeyError, match="Resolved header 'x' has no index"):
            tp.field("x")

    def test_fields_accepts_list_argument(self) -> None:
        """Exercise tuple/list coercion path in fields()."""
        zone = TecplotZone("z", I=2, J=1, K=1, datapacking="POINT", dt=[])
        data = np.arange(4.0).reshape(1, 1, 2, 2)
        tp = TecplotData(
            title="t",
            variables=["x", "y"],
            zone=zone,
            data=data,
            var_index={"x": 0, "y": 1},
        )

        out = tp.fields(["x", "y"])
        assert out.shape == (1, 1, 2, 2)


class TestReadTecplotAsciiExtra:
    """Target edge and fallback parsing branches in reader."""

    def test_plain_ascii_no_data_raises(self, tmp_path: Path) -> None:
        """Raise for plain-ASCII files with only comments/blank lines."""
        fpath = tmp_path / "plain.txt"
        fpath.write_text("# comment\n\n   \n", encoding="utf-8")

        with pytest.raises(ValueError, match="No data found"):
            read_tecplot_ascii(fpath)

    def test_plain_ascii_row_width_mismatch_raises(self, tmp_path: Path) -> None:
        """Raise when plain-ASCII rows have inconsistent column counts."""
        fpath = tmp_path / "plain.txt"
        fpath.write_text("1 2 3\n4 5\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Error parsing row 2"):
            read_tecplot_ascii(fpath)

    def test_fortran_style_float_repair(self, tmp_path: Path) -> None:
        """Parse tokens like 0.901560349186166-100 by inserting E."""
        content = (
            'TITLE = "Fortran Float"\n'
            'VARIABLES = "x"\n'
            'ZONE T="z", I=1, J=1, K=1\n'
            "0.901560349186166-100\n"
        )
        fpath = tmp_path / "fortran.dat"
        fpath.write_text(content, encoding="utf-8")

        tp = read_tecplot_ascii(fpath)
        assert np.isclose(tp.data[0, 0, 0, 0], 0.901560349186166e-100)

    def test_extra_numeric_tokens_are_trimmed(self, tmp_path: Path) -> None:
        """Allow extra numeric tokens and trim to expected size."""
        content = (
            'TITLE = "Extra Tokens"\n'
            'VARIABLES = "x" "y"\n'
            'ZONE T="z", I=1, J=1, K=1\n'
            "1.0 2.0 3.0 4.0\n"
        )
        fpath = tmp_path / "extra.dat"
        fpath.write_text(content, encoding="utf-8")

        tp = read_tecplot_ascii(fpath)
        assert tp.data.shape == (1, 1, 1, 2)
        assert np.allclose(tp.data[0, 0, 0, :], np.array([1.0, 2.0]))


class TestWriteTecplotAsciiExtra:
    """Target writer branches not covered by baseline tests."""

    def test_write_1d_and_read_back(self, tmp_path: Path) -> None:
        """Write 1-D zone and verify dimensions on read-back."""
        out = tmp_path / "one_d.dat"
        write_tecplot_ascii(out, {"x": np.array([1.0, 2.0, 3.0])}, title="t", zone="z")

        tp = read_tecplot_ascii(out)
        assert tp.zone.I == 3
        assert tp.zone.J == 1
        assert tp.zone.K == 1

    def test_write_shape_mismatch_raises(self, tmp_path: Path) -> None:
        """Raise when variable arrays do not share shape."""
        out = tmp_path / "bad_shape.dat"
        with pytest.raises(ValueError, match="shape mismatch"):
            write_tecplot_ascii(
                out,
                {
                    "a": np.zeros((2, 2)),
                    "b": np.zeros((2, 3)),
                },
            )

    def test_write_unsupported_ndim_raises(self, tmp_path: Path) -> None:
        """Raise when array ndim is not 1, 2, or 3."""
        out = tmp_path / "bad_ndim.dat"
        with pytest.raises(ValueError, match="expected 1-D, 2-D, or 3-D arrays"):
            write_tecplot_ascii(out, {"a": np.zeros((1, 1, 1, 1))})
