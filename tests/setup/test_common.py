"""Tests for lst_tools.setup._common helpers."""

import stat


from lst_tools.setup._common import scaffold_case_dir, write_launcher_script


# --------------------------------------------------
# tests for scaffold_case_dir
# --------------------------------------------------
class TestScaffoldCaseDir:
    """Test suite for scaffold_case_dir function."""

    def test_creates_directory(self, tmp_path):
        """Target directory is created if it does not exist."""
        # create a dummy meanflow file to copy
        meanflow = tmp_path / "meanflow.bin"
        meanflow.write_bytes(b"\x00" * 16)

        case_dir = tmp_path / "kc_001pt00"
        scaffold_case_dir(case_dir, meanflow, lst_exe=None)

        assert case_dir.is_dir()

    def test_copies_meanflow(self, tmp_path):
        """Meanflow binary is copied into the case directory."""
        meanflow = tmp_path / "meanflow.bin"
        payload = b"meanflow-data-1234"
        meanflow.write_bytes(payload)

        case_dir = tmp_path / "case1"
        scaffold_case_dir(case_dir, meanflow, lst_exe=None)

        copied = case_dir / "meanflow.bin"
        assert copied.exists()
        assert copied.read_bytes() == payload

    def test_copies_executable(self, tmp_path):
        """LST executable is copied and made executable."""
        meanflow = tmp_path / "meanflow.bin"
        meanflow.write_bytes(b"\x00")

        exe = tmp_path / "lst.x"
        exe.write_bytes(b"\x7fELF")

        case_dir = tmp_path / "case2"
        scaffold_case_dir(case_dir, meanflow, lst_exe=str(exe))

        copied_exe = case_dir / "lst.x"
        assert copied_exe.exists()
        mode = copied_exe.stat().st_mode
        assert mode & stat.S_IXUSR

    def test_no_exe_no_error(self, tmp_path):
        """Passing lst_exe=None is fine — no executable copied."""
        meanflow = tmp_path / "meanflow.bin"
        meanflow.write_bytes(b"\x00")

        case_dir = tmp_path / "case3"
        scaffold_case_dir(case_dir, meanflow, lst_exe=None)

        # only meanflow should be present
        files = list(case_dir.iterdir())
        assert len(files) == 1
        assert files[0].name == "meanflow.bin"

    def test_missing_exe_warns(self, tmp_path):
        """Non-existent lst_exe path logs a warning but does not raise."""
        meanflow = tmp_path / "meanflow.bin"
        meanflow.write_bytes(b"\x00")

        case_dir = tmp_path / "case4"
        # should not raise even though the exe path doesn't exist
        scaffold_case_dir(case_dir, meanflow, lst_exe="/nonexistent/lst.x")

        # meanflow still copied
        assert (case_dir / "meanflow.bin").exists()

    def test_existing_directory_ok(self, tmp_path):
        """Re-running on an existing directory succeeds (exist_ok=True)."""
        meanflow = tmp_path / "meanflow.bin"
        meanflow.write_bytes(b"\x00")

        case_dir = tmp_path / "case5"
        case_dir.mkdir()

        # should not raise
        scaffold_case_dir(case_dir, meanflow, lst_exe=None)
        assert (case_dir / "meanflow.bin").exists()


# --------------------------------------------------
# tests for write_launcher_script
# --------------------------------------------------
class TestWriteLauncherScript:
    """Test suite for write_launcher_script function."""

    def test_local_execution(self, tmp_path, monkeypatch):
        """With no submit_cmd, script uses direct execution lines."""
        monkeypatch.chdir(tmp_path)
        dirs = ["case_a", "case_b"]
        out = write_launcher_script(dirs)

        content = out.read_text()
        assert "#!/usr/bin/env bash" in content
        assert "set -euo pipefail" in content
        assert "cd case_a" in content
        assert "cd case_b" in content
        # local mode uses ./lst.x
        assert "./lst.x" in content

    def test_hpc_submit(self, tmp_path, monkeypatch):
        """With submit_cmd and fname_run_script, script emits submit lines."""
        monkeypatch.chdir(tmp_path)
        dirs = ["case_1"]
        out = write_launcher_script(
            dirs, submit_cmd="sbatch", fname_run_script="run.slurm"
        )

        content = out.read_text()
        assert "sbatch run.slurm" in content

    def test_executable_permission(self, tmp_path, monkeypatch):
        """Output script has executable permission."""
        monkeypatch.chdir(tmp_path)
        out = write_launcher_script(["d1"])

        mode = out.stat().st_mode
        assert mode & stat.S_IXUSR

    def test_empty_dirs(self, tmp_path, monkeypatch):
        """Empty dirs list produces a valid but minimal script."""
        monkeypatch.chdir(tmp_path)
        out = write_launcher_script([])

        content = out.read_text()
        assert "#!/usr/bin/env bash" in content
        # no cd lines
        assert "cd" not in content

    def test_custom_script_name(self, tmp_path, monkeypatch):
        """Custom script_name is respected."""
        monkeypatch.chdir(tmp_path)
        out = write_launcher_script(["d1"], script_name="submit_all.sh")

        assert out.name == "submit_all.sh"
        assert out.exists()
