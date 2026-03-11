import os
import struct
import numpy as np
import pytest
from lst_tools.data_io.lastrac_binary import LastracWriter


class TestFortranBinaryWriter:
    """Test cases for the LastracWriter class."""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file path for testing."""
        return tmp_path / "test.bin"

    def test_init_and_close(self, temp_file):
        """Test initialization and closing of the writer."""
        writer = LastracWriter(temp_file)
        assert os.path.exists(temp_file)  # File is created immediately
        writer.close()

    def test_write_string_fixed(self, temp_file):
        """Test writing fixed-length strings."""
        writer = LastracWriter(temp_file)
        test_string = "Hello, Fortran!"
        writer.write_string_fixed(test_string, length=64)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # Should have 4-byte marker + 64 bytes + 4-byte marker
        assert len(data) == 4 + 64 + 4
        # Check record markers
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 64
        # Check content
        content = data[4:68].decode("ascii").rstrip()
        assert content == test_string

    def test_write_ints(self, temp_file):
        """Test writing integer sequences."""
        writer = LastracWriter(temp_file)
        values = [1, 2, 3, 4, 5]
        writer.write_ints(values)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # 4-byte marker + 5*4 bytes + 4-byte marker
        assert len(data) == 4 + 20 + 4
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 20  # 5 integers * 4 bytes each

        # Check content
        content = struct.unpack("<5i", data[4:24])
        assert list(content) == values

    def test_write_reals(self, temp_file):
        """Test writing real number sequences."""
        writer = LastracWriter(temp_file)
        values = [1.1, 2.2, 3.3]
        writer.write_reals(values)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # 4-byte marker + 3*8 bytes + 4-byte marker
        assert len(data) == 4 + 24 + 4
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 24  # 3 doubles * 8 bytes each

        # Check content
        content = struct.unpack("<3d", data[4:28])
        np.testing.assert_array_almost_equal(content, values)

    def test_write_array_real_c_order(self, temp_file):
        """Test writing real arrays in C order."""
        writer = LastracWriter(temp_file)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        writer.write_array_real(arr, fortran_order=False)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # 4-byte marker + 4*8 bytes + 4-byte marker
        assert len(data) == 4 + 32 + 4
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 32  # 4 doubles * 8 bytes each

        # Check content (C order: row by row)
        content = struct.unpack("<4d", data[4:36])
        expected = [1.0, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(content, expected)

    def test_write_array_real_fortran_order(self, temp_file):
        """Test writing real arrays in Fortran order."""
        writer = LastracWriter(temp_file)
        arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        writer.write_array_real(arr, fortran_order=True)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # 4-byte marker + 4*8 bytes + 4-byte marker
        assert len(data) == 4 + 32 + 4
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 32  # 4 doubles * 8 bytes each

        # Check content (Fortran order: column by column)
        content = struct.unpack("<4d", data[4:36])
        expected = [1.0, 3.0, 2.0, 4.0]
        np.testing.assert_array_almost_equal(content, expected)

    def test_write_header(self, temp_file):
        """Test writing header records."""
        writer = LastracWriter(temp_file)
        writer.write_header(
            title="test_title",
            n_station=10,
            igas=1,
            iunit=2,
            Pr=0.71,
            stat_pres=101325.0,
            nsp=0,
        )
        writer.close()

        # Verify the file structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # First record: title (64 bytes) + markers
        assert struct.unpack("<i", data[:4])[0] == 64
        assert struct.unpack("<i", data[68:72])[0] == 64
        assert data[4:68].decode("ascii").rstrip() == "test_title"

        # Second record: n_station (4 bytes) + markers
        assert struct.unpack("<i", data[72:76])[0] == 4
        assert struct.unpack("<i", data[80:84])[0] == 4
        assert struct.unpack("<i", data[76:80])[0] == 10

        # Third record: 5 values (2 ints + 2 doubles + 1 int) + markers
        assert struct.unpack("<i", data[84:88])[0] == 28  # 2*4 + 2*8 + 4
        assert struct.unpack("<i", data[116:120])[0] == 28

        values = struct.unpack("<iiddi", data[88:116])
        assert values[0] == 1
        assert values[1] == 2
        assert abs(values[2] - 0.71) < 1e-10
        assert abs(values[3] - 101325.0) < 1e-10
        assert values[4] == 0

    def test_write_station_header(self, temp_file):
        """Test writing station header records."""
        writer = LastracWriter(temp_file)
        writer.write_station_header(
            i_loc=1,
            n_eta=150,
            s=0.5,
            lref=1.0,
            re1=1000.0,
            kappa=1.4,
            rloc=0.1,
            drdx=0.01,
            stat_temp=300.0,
            stat_uvel=10.0,
            stat_dens=1.225,
        )
        writer.close()

        # Verify the file structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # First record: 2 ints + 6 doubles (4+4+48=56 bytes) + markers
        assert struct.unpack("<i", data[:4])[0] == 56
        assert struct.unpack("<i", data[60:64])[0] == 56

        values1 = struct.unpack("<iidddddd", data[4:60])
        assert values1[0] == 1
        assert values1[1] == 150
        assert values1[2] == 0.5
        assert values1[3] == 1.0
        assert values1[4] == 1000.0
        assert abs(values1[5] - 1.4) < 1e-10
        assert values1[6] == 0.1
        assert values1[7] == 0.01

        # Second record: 3 doubles (24 bytes) + markers
        assert struct.unpack("<i", data[64:68])[0] == 24
        assert struct.unpack("<i", data[92:96])[0] == 24

        values2 = struct.unpack("<ddd", data[68:92])
        assert abs(values2[0] - 300.0) < 1e-10
        assert abs(values2[1] - 10.0) < 1e-10
        assert abs(values2[2] - 1.225) < 1e-10

    def test_write_station_vector(self, temp_file):
        """Test writing station vector data."""
        writer = LastracWriter(temp_file)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        writer.write_station_vector(vec)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # 4-byte marker + 5*8 bytes + 4-byte marker
        assert len(data) == 4 + 40 + 4
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 40  # 5 doubles * 8 bytes each

        # Check content
        content = struct.unpack("<5d", data[4:44])
        np.testing.assert_array_almost_equal(content, vec)

    def test_write_station_vector_with_count(self, temp_file):
        """Test writing station vector with limited count."""
        writer = LastracWriter(temp_file)
        vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        writer.write_station_vector(vec, count=3)
        writer.close()

        # Verify the record structure
        with open(temp_file, "rb") as f:
            data = f.read()

        # 4-byte marker + 3*8 bytes + 4-byte marker
        assert len(data) == 4 + 24 + 4
        start_marker = struct.unpack("<i", data[:4])[0]
        end_marker = struct.unpack("<i", data[-4:])[0]
        assert start_marker == end_marker == 24  # 3 doubles * 8 bytes each

        # Check content
        content = struct.unpack("<3d", data[4:28])
        np.testing.assert_array_almost_equal(content, vec[:3])
