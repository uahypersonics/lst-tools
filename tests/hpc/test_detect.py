"""Tests for lst_tools.hpc._detect — uses monkeypatch to avoid real subprocesses."""

from __future__ import annotations

from unittest.mock import patch

from lst_tools.hpc._detect import (
    DetectedEnv,
    _detect_hostname,
    _detect_launcher,
    _detect_scheduler,
)
from lst_tools.hpc._profiles import Scheduler


class TestDetectHostname:
    def test_strips_digits(self):
        with patch.dict("os.environ", {"HOSTNAME": "puma003.hpc.arizona.edu"}):
            assert _detect_hostname() == "puma"

    def test_socket_fallback(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("socket.gethostname", return_value="mybox.local"):
                assert _detect_hostname() == "mybox"


class TestDetectScheduler:
    def test_slurm_env(self):
        with patch.dict("os.environ", {"SLURM_JOB_ID": "12345"}):
            with patch("shutil.which", return_value=None):
                assert _detect_scheduler() is Scheduler.SLURM

    def test_pbs_env(self):
        with patch.dict("os.environ", {"PBS_JOBID": "12345"}, clear=True):
            with patch("shutil.which", return_value=None):
                assert _detect_scheduler() is Scheduler.PBS

    def test_unknown(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("shutil.which", return_value=None):
                assert _detect_scheduler() is Scheduler.UNKNOWN


class TestDetectLauncher:
    def test_finds_mpirun(self):
        def fake_which(name):
            return "/usr/bin/mpirun" if name == "mpirun" else None

        with patch("shutil.which", side_effect=fake_which):
            assert _detect_launcher() == "mpirun"

    def test_none_when_nothing(self):
        with patch("shutil.which", return_value=None):
            assert _detect_launcher() is None


class TestDetectedEnvFrozen:
    def test_frozen(self):
        env = DetectedEnv(
            hostname="test",
            scheduler=Scheduler.UNKNOWN,
            launcher=None,
            cpus_per_node=None,
            cpus_histogram={},
            resources=(),
            profile=None,
        )
        try:
            env.hostname = "hacked"  # type: ignore[misc]
            assert False, "should have raised"
        except AttributeError:
            pass
