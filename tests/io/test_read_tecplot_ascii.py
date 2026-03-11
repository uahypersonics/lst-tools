import numpy as np
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from lst_tools.data_io.tecplot_ascii import (
    read_tecplot_ascii,
    TecplotData,
    TecplotZone,
    _normalize,
    _parse_variables_block,
    _parse_zone_header,
)


class TestNormalize:
    """Test suite for _normalize helper function"""

    def test_normalize_basic(self):
        """Test basic normalization"""
        assert _normalize("X-Location") == "xlocation"
        assert _normalize("x_location") == "xlocation"
        assert _normalize("x.location") == "xlocation"
        assert _normalize("X Location") == "xlocation"

    def test_normalize_complex(self):
        """Test normalization with parentheses and multiple separators"""
        assert _normalize("-im(alpha)") == "imalpha"
        assert _normalize("re(alpha)") == "realpha"
        assert _normalize("n-factor") == "nfactor"
        assert _normalize("  spaced  ") == "spaced"

    def test_normalize_empty(self):
        """Test normalization of empty or whitespace strings"""
        assert _normalize("") == ""
        assert _normalize("   ") == ""
        assert _normalize("___") == ""


class TestParseVariablesBlock:
    """Test suite for _parse_variables_block function"""

    def test_parse_single_line_variables(self):
        """Test parsing variables on a single line"""
        lines = ['VARIABLES = "x" "y" "z"', 'ZONE T="test"']
        variables, next_idx = _parse_variables_block(lines, 0)

        assert variables == ["x", "y", "z"]
        assert next_idx == 1

    def test_parse_multiline_variables(self):
        """Test parsing variables spanning multiple lines"""
        lines = ['VARIABLES = "x"', '"y"', '"z"', 'ZONE T="test"']
        variables, next_idx = _parse_variables_block(lines, 0)

        assert variables == ["x", "y", "z"]
        assert next_idx == 3

    def test_parse_variables_no_quotes(self):
        """Test parsing variables without quotes (fallback)"""
        lines = ["VARIABLES = x, y, z", 'ZONE T="test"']
        variables, next_idx = _parse_variables_block(lines, 0)

        assert variables == ["x", "y", "z"]
        assert next_idx == 1

    def test_parse_variables_mixed_format(self):
        """Test parsing variables with mixed formatting"""
        lines = ['VARIABLES = "velocity_x" "velocity_y"', '"pressure"', 'ZONE T="test"']
        variables, next_idx = _parse_variables_block(lines, 0)

        assert variables == ["velocity_x", "velocity_y", "pressure"]
        assert next_idx == 2


class TestParseZoneHeader:
    """Test suite for _parse_zone_header function"""

    def test_parse_simple_zone(self):
        """Test parsing a simple zone header"""
        lines = ['ZONE T="Test Zone", I=10, J=20, K=1', "1.0 2.0 3.0"]
        zone, next_idx = _parse_zone_header(lines, 0)

        assert zone.name == "Test Zone"
        assert zone.I == 10
        assert zone.J == 20
        assert zone.K == 1
        assert zone.datapacking == "POINT"
        assert next_idx == 1

    def test_parse_multiline_zone(self):
        """Test parsing zone header spanning multiple lines"""
        lines = [
            'ZONE T="Multi Line"',
            "I=5",
            "J=10",
            "K=2",
            "DATAPACKING=POINT",
            "DT=(DOUBLE DOUBLE)",
            "1.0 2.0",
        ]
        zone, next_idx = _parse_zone_header(lines, 0)

        assert zone.name == "Multi Line"
        assert zone.I == 5
        assert zone.J == 10
        assert zone.K == 2
        assert zone.datapacking == "POINT"
        assert zone.dt == ["DOUBLE", "DOUBLE"]
        assert next_idx == 6

    def test_parse_zone_defaults(self):
        """Test zone parsing with default values"""
        lines = ['ZONE T="Minimal"', "1.0 2.0"]
        zone, next_idx = _parse_zone_header(lines, 0)

        assert zone.name == "Minimal"
        assert zone.I == 1
        assert zone.J == 1
        assert zone.K == 1
        assert zone.datapacking == "POINT"
        assert zone.dt == []


class TestTecplotData:
    """Test suite for TecplotData class"""

    def test_tecplot_data_initialization(self):
        """Test TecplotData initialization and post_init"""
        zone = TecplotZone("test", I=2, J=2, K=1, datapacking="POINT", dt=[])
        data = np.arange(16).reshape(1, 2, 2, 4)
        variables = ["x", "y", "u", "v"]

        tp_data = TecplotData(
            title="Test",
            variables=variables,
            zone=zone,
            data=data,
            var_index={"x": 0, "y": 1, "u": 2, "v": 3},
        )

        # Check basic attributes
        assert tp_data.title == "Test"
        assert tp_data.variables == variables
        assert tp_data.data.shape == (1, 2, 2, 4)

        # Check normalized header map
        assert "x" in tp_data._norm_to_header
        assert tp_data._norm_to_header["x"] == "x"

    def test_tecplot_data_field_access(self):
        """Test field access methods"""
        zone = TecplotZone("test", I=2, J=3, K=1, datapacking="POINT", dt=[])
        data = np.arange(24).reshape(1, 3, 2, 4)
        variables = ["x", "y", "u", "v"]

        tp_data = TecplotData(
            title="Test",
            variables=variables,
            zone=zone,
            data=data,
            var_index={"x": 0, "y": 1, "u": 2, "v": 3},
        )

        # Test single field access
        x_field = tp_data.field("x")
        assert x_field.shape == (1, 3, 2)
        assert np.array_equal(x_field, data[..., 0])

        # Test multiple field access
        xy_fields = tp_data.fields("x", "y")
        assert xy_fields.shape == (1, 3, 2, 2)
        assert np.array_equal(xy_fields[..., 0], data[..., 0])
        assert np.array_equal(xy_fields[..., 1], data[..., 1])

    def test_tecplot_data_aliases(self):
        """Test alias functionality"""
        zone = TecplotZone("test", I=2, J=2, K=1, datapacking="POINT", dt=[])
        data = np.ones((1, 2, 2, 3))
        variables = ["x-location", "frequency", "-im(alpha)"]

        tp_data = TecplotData(
            title="Test",
            variables=variables,
            zone=zone,
            data=data,
            var_index={"x-location": 0, "frequency": 1, "-im(alpha)": 2},
        )

        # Test default aliases
        assert tp_data.field("s").shape == (1, 2, 2)  # should map to x-location
        assert tp_data.field("freq").shape == (1, 2, 2)  # should map to frequency
        assert tp_data.field("alpi").shape == (1, 2, 2)  # should map to -im(alpha)

        # Test custom alias
        tp_data.add_alias("custom", "x-location")
        assert tp_data.field("custom").shape == (1, 2, 2)

    def test_tecplot_data_resolve_errors(self):
        """Test error handling in field resolution"""
        zone = TecplotZone("test", I=2, J=2, K=1, datapacking="POINT", dt=[])
        data = np.ones((1, 2, 2, 2))
        variables = ["x", "y", "variable"]

        tp_data = TecplotData(
            title="Test",
            variables=variables,
            zone=zone,
            data=data,
            var_index={"x": 0, "y": 1, "variable": 2},
        )

        # Test non-existent field
        with pytest.raises(KeyError, match="Variable 'z' not found"):
            tp_data.field("z")

        # Test fuzzy matching hint
        with pytest.raises(KeyError, match="did you mean: "):
            tp_data.field("vari")  # should suggest 'x'


class TestReadTecplotAscii:
    """Test suite for read_tecplot_ascii function"""

    def create_temp_tecplot_file(self, content: str) -> Path:
        """Helper to create temporary tecplot file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False) as f:
            f.write(content)
            return Path(f.name)

    def test_read_simple_tecplot_file(self):
        """Test reading a simple tecplot file"""
        content = """TITLE = "Simple Test"
VARIABLES = "x" "y" "u"
ZONE T="Zone1", I=2, J=2, K=1
1.0 2.0 0.5
3.0 4.0 1.5
5.0 6.0 2.5
7.0 8.0 3.5
"""
        temp_file = self.create_temp_tecplot_file(content)

        try:
            tp_data = read_tecplot_ascii(temp_file)

            assert tp_data.title == "Simple Test"
            assert tp_data.variables == ["x", "y", "u"]
            assert tp_data.zone.name == "Zone1"
            assert tp_data.zone.I == 2
            assert tp_data.zone.J == 2
            assert tp_data.zone.K == 1
            assert tp_data.data.shape == (1, 2, 2, 3)

            # Check specific values
            assert tp_data.data[0, 0, 0, 0] == 1.0  # x at (0,0,0)
            assert tp_data.data[0, 1, 1, 2] == 3.5  # u at (0,1,1)

        finally:
            os.unlink(temp_file)

    def test_read_3d_tecplot_file(self):
        """Test reading a 3D tecplot file"""
        content = """TITLE = "3D Test"
VARIABLES = "x" "y" "z" "p"
ZONE T="3D Zone", I=2, J=2, K=2
1.0 2.0 3.0 4.0
5.0 6.0 7.0 8.0
9.0 10.0 11.0 12.0
13.0 14.0 15.0 16.0
17.0 18.0 19.0 20.0
21.0 22.0 23.0 24.0
25.0 26.0 27.0 28.0
29.0 30.0 31.0 32.0
"""
        temp_file = self.create_temp_tecplot_file(content)

        try:
            tp_data = read_tecplot_ascii(temp_file)

            assert tp_data.data.shape == (2, 2, 2, 4)
            assert tp_data.zone.K == 2

            # Check data ordering
            assert tp_data.data[0, 0, 0, 0] == 1.0
            assert tp_data.data[1, 1, 1, 3] == 32.0

        finally:
            os.unlink(temp_file)

    def test_read_scientific_notation(self):
        """Test reading data with scientific notation"""
        content = """TITLE = "Scientific"
VARIABLES = "x" "y"
ZONE T="Sci", I=2, J=1, K=1
1.0e-3 2.5e+2
-3.14e-1 6.02e23
"""
        temp_file = self.create_temp_tecplot_file(content)

        try:
            tp_data = read_tecplot_ascii(temp_file)

            assert np.isclose(tp_data.data[0, 0, 0, 0], 1.0e-3)
            assert np.isclose(tp_data.data[0, 0, 0, 1], 2.5e2)
            assert np.isclose(tp_data.data[0, 0, 1, 0], -3.14e-1)
            assert np.isclose(tp_data.data[0, 0, 1, 1], 6.02e23)

        finally:
            os.unlink(temp_file)

    def test_read_multiline_headers(self):
        """Test reading file with multiline headers"""
        content = """TITLE = "Multiline Test"
VARIABLES = "x"
"y"
"z"
ZONE T="Multi"
I=2
J=1
K=1
DATAPACKING=POINT
1.0 2.0 3.0
4.0 5.0 6.0
"""
        temp_file = self.create_temp_tecplot_file(content)

        try:
            tp_data = read_tecplot_ascii(temp_file)

            assert tp_data.variables == ["x", "y", "z"]
            assert tp_data.zone.I == 2
            assert tp_data.zone.datapacking == "POINT"

        finally:
            os.unlink(temp_file)

    def test_read_invalid_file_errors(self):
        """Test error handling for invalid files"""

        # Missing VARIABLES
        content1 = """TITLE = "No Variables"
ZONE T="Zone1", I=2, J=1
1.0 2.0
"""
        temp_file1 = self.create_temp_tecplot_file(content1)

        try:
            with pytest.raises(ValueError, match="VARIABLES header not found"):
                read_tecplot_ascii(temp_file1)
        finally:
            os.unlink(temp_file1)

        # Missing ZONE
        content2 = """TITLE = "No Zone"
VARIABLES = "x" "y"
1.0 2.0
"""
        temp_file2 = self.create_temp_tecplot_file(content2)

        try:
            with pytest.raises(ValueError, match="ZONE header not found"):
                read_tecplot_ascii(temp_file2)
        finally:
            os.unlink(temp_file2)

        # Not enough data
        content3 = """TITLE = "Insufficient Data"
VARIABLES = "x" "y"
ZONE T="Zone1", I=2, J=2
1.0 2.0
3.0
"""
        temp_file3 = self.create_temp_tecplot_file(content3)

        try:
            with pytest.raises(ValueError, match="Not enough data"):
                read_tecplot_ascii(temp_file3)
        finally:
            os.unlink(temp_file3)

        # Non-numeric data
        content4 = """TITLE = "Bad Data"
VARIABLES = "x" "y"
ZONE T="Zone1", I=2, J=1
1.0 2.0
abc 4.0
"""
        temp_file4 = self.create_temp_tecplot_file(content4)

        try:
            with pytest.raises(ValueError, match="Non-numeric token"):
                read_tecplot_ascii(temp_file4)
        finally:
            os.unlink(temp_file4)

    def test_read_with_debug(self):
        """Test debug output"""
        content = """TITLE = "Debug Test"
VARIABLES = "x" "y"
ZONE T="Zone1", I=2, J=1
1.0 2.0
3.0 4.0
"""
        temp_file = self.create_temp_tecplot_file(content)

        try:
            with patch("lst_tools.data_io.tecplot_ascii.logger") as mock_logger:
                read_tecplot_ascii(temp_file)

                messages = " ".join(
                    str(c) for c in mock_logger.debug.call_args_list
                    + mock_logger.info.call_args_list
                )
                assert "variables" in messages
                assert "'x'" in messages
                assert "'y'" in messages
                assert "zone" in messages.lower()

        finally:
            os.unlink(temp_file)

    def test_tecplot_data_tables(self):
        """Test table formatting methods"""
        zone = TecplotZone("test", I=2, J=1, K=1, datapacking="POINT", dt=[])
        data = np.ones((1, 1, 2, 3))
        variables = ["x-location", "y-location", "velocity"]

        tp_data = TecplotData(
            title="Test",
            variables=variables,
            zone=zone,
            data=data,
            var_index={"x-location": 0, "y-location": 1, "velocity": 2},
        )

        # Test headers table
        headers_table = tp_data.headers_table()
        assert "x-location" in headers_table
        assert "y-location" in headers_table
        assert "velocity" in headers_table

        # Test aliases table
        aliases_table = tp_data.aliases_table()
        assert "alias" in aliases_table
        assert "header" in aliases_table

        # Test debug_aliases (just ensure it doesn't crash)
        import io

        buffer = io.StringIO()
        tp_data.debug_aliases(file=buffer)
        output = buffer.getvalue()
        assert "headers:" in output
