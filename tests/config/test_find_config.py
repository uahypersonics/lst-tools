from lst_tools.config.find_config import find_config


class TestFindConfig:
    def test_find_config_with_lst_cfg(self, tmp_path):
        """Test find_config finds lst.cfg with highest priority"""
        # Create config files with different names
        (tmp_path / "lst.cfg").touch()
        (tmp_path / "config.toml").touch()
        (tmp_path / "lst.toml").touch()

        result = find_config(tmp_path)
        assert result == tmp_path / "lst.cfg"

    def test_find_config_priority_order(self, tmp_path):
        """Test that find_config respects the priority order of config files"""
        # Create lowest priority file first
        (tmp_path / "config.toml").touch()

        result = find_config(tmp_path)
        assert result == tmp_path / "config.toml"

        # Add higher priority file
        (tmp_path / "lst.toml").touch()
        result = find_config(tmp_path)
        assert result == tmp_path / "lst.toml"

        # Add highest priority file
        (tmp_path / "lst.cfg").touch()
        result = find_config(tmp_path)
        assert result == tmp_path / "lst.cfg"

    def test_find_config_no_config_files(self, tmp_path):
        """Test find_config returns None when no config files are found"""
        # Create a temporary directory with no config files
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = find_config(empty_dir)
        assert result is None

    def test_find_config_with_string_path(self, tmp_path):
        """Test find_config works with string path input"""
        (tmp_path / "lst.toml").touch()

        result = find_config(str(tmp_path))
        assert result == tmp_path / "lst.toml"

    def test_find_config_default_path(self, tmp_path, monkeypatch):
        """Test find_config uses current directory when no path is provided"""
        # Change working directory to a clean temporary directory with no config files
        monkeypatch.chdir(tmp_path)
        result = find_config()
        assert result is None

    def test_find_config_single_config_file(self, tmp_path):
        """Test find_config works with only one config file present"""
        config_file = tmp_path / "config.toml"
        config_file.touch()

        result = find_config(tmp_path)
        assert result == config_file
