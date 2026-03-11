import pytest
import os
from pathlib import Path
import tempfile
import shutil
import numpy as np

from lst_tools.config.write_config import (
    _serialize_for_toml,
    write_config,
)


class TestSerializeForToml:
    """Test suite for _serialize_for_toml function"""

    def test_serialize_numpy_scalar(self):
        """Test serializing numpy scalar"""
        result = _serialize_for_toml(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    def test_serialize_complex(self):
        """Test serializing complex number"""
        result = _serialize_for_toml(complex(1.5, -2.5))
        assert result == "(1.5,-2.5)"

    def test_serialize_complex_with_zero_imag(self):
        """Test serializing complex number with zero imaginary part"""
        result = _serialize_for_toml(complex(1, 0))
        # The format uses Python's default float representation
        assert result == "(1.0,0.0)"

    def test_serialize_path(self):
        """Test serializing Path object"""
        result = _serialize_for_toml(Path("/path/to/file"))
        assert result == "/path/to/file"
        assert isinstance(result, str)

    def test_serialize_dict_with_none(self):
        """Test serializing dictionary with None values"""
        input_dict = {"key1": "value1", "key2": None, "key3": 42}
        result = _serialize_for_toml(input_dict)
        assert result == {"key1": "value1", "key2": "", "key3": 42}

    def test_serialize_nested_dict(self):
        """Test serializing nested dictionary"""
        input_dict = {
            "outer": {
                "inner": complex(1, 0),
                "path": Path("test.txt"),
                "none_val": None,
            }
        }
        result = _serialize_for_toml(input_dict)
        expected = {"outer": {"inner": "(1.0,0.0)", "path": "test.txt", "none_val": ""}}
        assert result == expected

    def test_serialize_list_with_none(self):
        """Test serializing list with None values"""
        input_list = [1, None, "test", complex(0, 1)]
        result = _serialize_for_toml(input_list)
        # Fixed: complex formatting uses default float representation
        assert result == [1, "", "test", "(0.0,1.0)"]

    def test_serialize_tuple(self):
        """Test serializing tuple"""
        input_tuple = (Path("file"), None, np.int32(5))
        result = _serialize_for_toml(input_tuple)
        assert result == ["file", "", 5]


class TestWriteConfig:
    """Test suite for write_config function"""

    def setup_method(self):
        """Setup for each test - create temporary directory"""
        self.test_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.test_dir)

        # Mock DEFAULTS
        self.mock_defaults = {
            "flow_conditions": {"mach": None, "re1": None, "gamma": 1.4},
            "geometry": {"type": "flat_plate"},
        }

    def teardown_method(self):
        """Cleanup after each test"""
        os.chdir(self.original_dir)
        shutil.rmtree(self.test_dir)

    def test_write_config_default_path(self):
        """Test write_config with default path"""
        result = write_config(cfg_data=self.mock_defaults)

        assert result == Path("lst.cfg")
        assert Path("lst.cfg").exists()

    def test_write_config_custom_path(self):
        """Test write_config with custom path"""
        custom_path = Path("custom/config.toml")
        result = write_config(path=custom_path, cfg_data=self.mock_defaults)

        assert result == custom_path
        assert custom_path.exists()
        assert custom_path.parent.exists()  # Directory was created

    def test_write_config_no_overwrite(self):
        """Test write_config doesn't overwrite existing file by default"""
        # Create existing file
        Path("lst.cfg").write_text("existing content")

        result = write_config(cfg_data=self.mock_defaults)

        # Should return path but not overwrite
        assert result == Path("lst.cfg")
        assert Path("lst.cfg").read_text() == "existing content"

    def test_write_config_with_overwrite(self):
        """Test write_config overwrites when flag is set"""
        # Create existing file
        Path("lst.cfg").write_text("existing content")

        result = write_config(overwrite=True, cfg_data=self.mock_defaults)

        # Should overwrite file
        assert result == Path("lst.cfg")
        assert Path("lst.cfg").read_text() != "existing content"

    def test_write_config_with_custom_data(self):
        """Test write_config with custom configuration data"""
        custom_config = {
            "section1": {"key1": "value1", "key2": 42},
            "section2": {"complex_val": complex(1, 2), "path_val": Path("test.txt")},
        }

        result = write_config(cfg_data=custom_config)

        assert result == Path("lst.cfg")
        assert Path("lst.cfg").exists()

        # Read back and verify structure
        content = Path("lst.cfg").read_text()
        assert "[section1]" in content
        assert "[section2]" in content

    def test_write_config_requires_cfg_data(self):
        """Test write_config raises ValueError when cfg_data is None."""
        with pytest.raises(ValueError, match="cfg_data is required"):
            write_config()

