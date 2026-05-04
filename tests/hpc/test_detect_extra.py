"""Additional tests for lst_tools.hpc._detect coverage gaps."""

from __future__ import annotations

import builtins
import subprocess

from lst_tools.hpc import _detect as detect_mod
from lst_tools.hpc._profiles import ClusterProfile, Scheduler


def test_detect_cpus_slurm_from_job_cpus_env(monkeypatch) -> None:
    """Use SLURM_JOB_CPUS_PER_NODE before any subprocess probing."""
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "32(x2),64")
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    cpus, hist = detect_mod._detect_cpus_slurm()

    assert cpus == 32
    assert hist == {32: 2, 64: 1}


def test_detect_cpus_slurm_from_cpus_on_node(monkeypatch) -> None:
    """Use SLURM_CPUS_ON_NODE when per-node list is absent."""
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "96")

    cpus, hist = detect_mod._detect_cpus_slurm()

    assert cpus == 96
    assert hist == {96: 1}


def test_detect_cpus_slurm_falls_back_to_sinfo_then_lscpu(monkeypatch) -> None:
    """Fall back from failed sinfo to lscpu parsing."""
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "not-an-int")

    def _fake_check_output(cmd, text=True):
        if cmd[0] == "sinfo":
            raise RuntimeError("sinfo failed")
        if cmd[0] == "lscpu":
            return "CPU(s):              128\n"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", _fake_check_output)

    cpus, hist = detect_mod._detect_cpus_slurm()

    assert cpus == 128
    assert hist == {128: 1}


def test_detect_cpus_slurm_uses_sinfo_when_env_parse_is_empty(monkeypatch) -> None:
    """Empty SLURM env parsing should continue to the sinfo probe."""
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "garbage")
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    monkeypatch.setattr(detect_mod, "parse_slurm_cpus_env", lambda raw: [])
    monkeypatch.setattr(detect_mod, "parse_sinfo_cpus", lambda out: [64, 64, 32])
    monkeypatch.setattr(
        detect_mod.subprocess,
        "check_output",
        lambda cmd, text=True: "node001 64\nnode002 64\nnode003 32\n",
    )

    cpus, hist = detect_mod._detect_cpus_slurm()

    assert cpus == 64
    assert hist == {64: 2, 32: 1}


def test_detect_cpus_slurm_returns_none_when_all_probes_fail(monkeypatch) -> None:
    """Return an empty detection result when SLURM probes yield no CPU count."""
    monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "garbage")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "still-not-an-int")

    monkeypatch.setattr(detect_mod, "parse_slurm_cpus_env", lambda raw: [])
    monkeypatch.setattr(detect_mod, "parse_sinfo_cpus", lambda out: [])
    monkeypatch.setattr(detect_mod, "parse_lscpu_cpus", lambda out: None)
    monkeypatch.setattr(
        detect_mod.subprocess,
        "check_output",
        lambda cmd, text=True: "unusable output",
    )

    cpus, hist = detect_mod._detect_cpus_slurm()

    assert cpus is None
    assert hist == {}


def test_detect_cpus_slurm_returns_none_when_lscpu_raises(monkeypatch) -> None:
    """SLURM should return an empty result when the final lscpu probe fails."""
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    def _fake_check_output(cmd, text=True):
        if cmd[0] == "sinfo":
            return "unusable sinfo output"
        if cmd[0] == "lscpu":
            raise RuntimeError("lscpu failed")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(detect_mod, "parse_sinfo_cpus", lambda out: [])
    monkeypatch.setattr(detect_mod.subprocess, "check_output", _fake_check_output)

    cpus, hist = detect_mod._detect_cpus_slurm()

    assert cpus is None
    assert hist == {}


def test_detect_cpus_pbs_from_nodefile(monkeypatch, tmp_path) -> None:
    """Use PBS_NODEFILE host repetition counts as CPU histogram."""
    nodefile = tmp_path / "pbs_nodefile"
    nodefile.write_text("nodeA\nnodeA\nnodeB\n", encoding="utf-8")

    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))

    cpus, hist = detect_mod._detect_cpus_pbs()

    assert cpus == 2
    assert hist == {2: 1, 1: 1}


def test_detect_cpus_pbs_falls_back_to_pbsnodes_then_lscpu(monkeypatch) -> None:
    """Try pbsnodes variants and finally lscpu when needed."""
    monkeypatch.delenv("PBS_NODEFILE", raising=False)

    calls = {"n": 0}

    def _fake_check_output(cmd, text=True):
        calls["n"] += 1
        if cmd[:2] == ["pbsnodes", "-ajS"]:
            raise RuntimeError("no -ajS support")
        if cmd[:2] == ["pbsnodes", "-a"]:
            return "resources_available.ncpus = 48\nresources_available.ncpus = 48\n"
        if cmd[0] == "lscpu":
            return "CPU(s): 64\n"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", _fake_check_output)

    cpus, hist = detect_mod._detect_cpus_pbs()

    assert cpus == 48
    assert hist == {48: 2}
    assert calls["n"] >= 2


def test_detect_cpus_pbs_uses_lscpu_when_other_sources_are_empty(monkeypatch, tmp_path) -> None:
    """PBS falls through to lscpu when nodefile and pbsnodes parsing are empty."""
    nodefile = tmp_path / "pbs_nodefile"
    nodefile.write_text("ignored\n", encoding="utf-8")

    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr(detect_mod, "parse_pbs_nodefile", lambda text: [])
    monkeypatch.setattr(
        detect_mod,
        "parse_pbsnodes_cpus",
        lambda out: [] if "pbsnodes" in out else [72],
    )
    monkeypatch.setattr(detect_mod, "parse_lscpu_cpus", lambda out: 72)

    def _fake_check_output(cmd, text=True):
        if cmd[:2] == ["pbsnodes", "-ajS"]:
            return "pbsnodes output"
        if cmd[:2] == ["pbsnodes", "-a"]:
            return "pbsnodes output"
        if cmd[0] == "lscpu":
            return "CPU(s): 72\n"
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", _fake_check_output)

    cpus, hist = detect_mod._detect_cpus_pbs()

    assert cpus == 72
    assert hist == {72: 1}


def test_detect_cpus_pbs_handles_bad_nodefile_and_lscpu_failure(monkeypatch, tmp_path) -> None:
    """PBS should return an empty result when nodefile read and lscpu both fail."""
    nodefile = tmp_path / "pbs_nodefile"
    nodefile.write_text("ignored\n", encoding="utf-8")

    monkeypatch.setenv("PBS_NODEFILE", str(nodefile))
    monkeypatch.setattr(
        builtins,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("nodefile unreadable")),
    )

    def _fake_check_output(cmd, text=True):
        if cmd[0] == "pbsnodes":
            raise RuntimeError("pbsnodes failed")
        if cmd[0] == "lscpu":
            raise RuntimeError("lscpu failed")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", _fake_check_output)

    cpus, hist = detect_mod._detect_cpus_pbs()

    assert cpus is None
    assert hist == {}


def test_detect_cpus_uses_profile_short_circuit() -> None:
    """Return profile CPUs directly without scheduler-specific probing."""
    profile = ClusterProfile(name="demo", scheduler=Scheduler.SLURM, cpus_per_node=94)

    cpus, hist = detect_mod._detect_cpus(Scheduler.SLURM, profile)

    assert cpus == 94
    assert hist == {94: 1}


def test_detect_cpus_unknown_scheduler_returns_empty() -> None:
    """Unknown scheduler should not attempt any scheduler-specific probe."""
    cpus, hist = detect_mod._detect_cpus(Scheduler.UNKNOWN, None)

    assert cpus is None
    assert hist == {}


def test_detect_cpus_dispatches_to_slurm_probe(monkeypatch) -> None:
    """SLURM scheduler should dispatch to the SLURM-specific CPU probe."""
    monkeypatch.setattr(detect_mod, "_detect_cpus_slurm", lambda: (88, {88: 2}))

    cpus, hist = detect_mod._detect_cpus(Scheduler.SLURM, None)

    assert cpus == 88
    assert hist == {88: 2}


def test_detect_cpus_dispatches_to_pbs_probe(monkeypatch) -> None:
    """PBS scheduler should dispatch to the PBS-specific CPU probe."""
    monkeypatch.setattr(detect_mod, "_detect_cpus_pbs", lambda: (48, {48: 3}))

    cpus, hist = detect_mod._detect_cpus(Scheduler.PBS, None)

    assert cpus == 48
    assert hist == {48: 3}


def test_detect_resources_profile_command_uses_matching_parser(monkeypatch) -> None:
    """Use parser selected by profile.resource_cmd."""
    profile = ClusterProfile(name="puma", resource_cmd="va")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", lambda *a, **k: "raw")
    monkeypatch.setattr(detect_mod, "parse_va_output", lambda out: [{"account": "A"}])

    rows = detect_mod._detect_resources(profile)

    assert rows == [{"account": "A"}]


def test_detect_resources_non_va_profile_uses_show_usage_parser(monkeypatch) -> None:
    """Non-va profile resource commands should still use the show_usage parser."""
    profile = ClusterProfile(name="delta", resource_cmd="show_usage")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", lambda *a, **k: "raw")
    monkeypatch.setattr(
        detect_mod,
        "parse_show_usage_output",
        lambda out: [{"account": "B"}],
    )

    rows = detect_mod._detect_resources(profile)

    assert rows == [{"account": "B"}]


def test_detect_resources_fallback_show_usage_and_errors(monkeypatch) -> None:
    """Fallback to show_usage and handle command failures cleanly."""
    # first: file missing
    monkeypatch.setattr(
        detect_mod.subprocess,
        "check_output",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    rows_missing = detect_mod._detect_resources(None)
    assert rows_missing == []

    # second: command returns non-zero
    def _raise_called(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["show_usage"])

    monkeypatch.setattr(detect_mod.subprocess, "check_output", _raise_called)
    rows_failed = detect_mod._detect_resources(None)
    assert rows_failed == []

    # third: command exists but is not directly executable; shell fallback works
    monkeypatch.setattr(detect_mod, "parse_show_usage_output", lambda out: [{"account": "AFOSR"}])

    def _execfmt_then_shell_ok(*args, **kwargs):
        if kwargs.get("shell"):
            return "mock shell output"
        raise OSError(8, "Exec format error")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", _execfmt_then_shell_ok)
    rows_execfmt_shell_ok = detect_mod._detect_resources(None)
    assert rows_execfmt_shell_ok == [{"account": "AFOSR"}]

    # fourth: command exists but shell fallback also fails
    monkeypatch.setattr(
        detect_mod.subprocess,
        "check_output",
        lambda *a, **k: (_ for _ in ()).throw(OSError(8, "Exec format error")),
    )
    rows_execfmt_fail = detect_mod._detect_resources(None)
    assert rows_execfmt_fail == []


def test_detect_resources_non_exec_oserror_returns_empty(monkeypatch) -> None:
    """Non-exec-format OS errors should not attempt the shell fallback."""
    monkeypatch.setattr(
        detect_mod.subprocess,
        "check_output",
        lambda *a, **k: (_ for _ in ()).throw(OSError(13, "Permission denied")),
    )

    rows = detect_mod._detect_resources(None)

    assert rows == []


def test_detect_uses_profile_over_runtime_probes(monkeypatch) -> None:
    """When hostname matches a profile, detect() should use profile settings."""
    detect_mod.detect.cache_clear()

    profile = ClusterProfile(
        name="puma",
        scheduler=Scheduler.SLURM,
        cpus_per_node=94,
        preferred_launcher="mpirun",
    )

    monkeypatch.setattr(detect_mod, "_detect_hostname", lambda: "junonia")
    monkeypatch.setattr(detect_mod, "lookup", lambda host: profile)
    monkeypatch.setattr(detect_mod, "_detect_scheduler", lambda: Scheduler.UNKNOWN)
    monkeypatch.setattr(detect_mod, "_detect_launcher", lambda: None)
    monkeypatch.setattr(detect_mod, "_detect_cpus", lambda sched, prof: (94, {94: 1}))
    monkeypatch.setattr(detect_mod, "_detect_resources", lambda prof: [])

    env = detect_mod.detect()

    assert env.hostname == "puma"
    assert env.scheduler is Scheduler.SLURM
    assert env.launcher == "mpirun"
    assert env.cpus_per_node == 94

    # clean cache for other tests
    detect_mod.detect.cache_clear()


def test_detect_without_profile_uses_runtime_functions(monkeypatch) -> None:
    """Without a profile, detect() should use scheduler/launcher probes."""
    detect_mod.detect.cache_clear()

    monkeypatch.setattr(detect_mod, "_detect_hostname", lambda: "localbox")
    monkeypatch.setattr(detect_mod, "lookup", lambda host: None)
    monkeypatch.setattr(detect_mod, "_detect_scheduler", lambda: Scheduler.PBS)
    monkeypatch.setattr(detect_mod, "_detect_launcher", lambda: "aprun")
    monkeypatch.setattr(detect_mod, "_detect_cpus", lambda sched, prof: (48, {48: 1}))
    monkeypatch.setattr(
        detect_mod,
        "_detect_resources",
        lambda prof: [{"account": "proj", "allocated": 10, "percent_remain": 50.0}],
    )

    env = detect_mod.detect()

    assert env.hostname == "localbox"
    assert env.scheduler is Scheduler.PBS
    assert env.launcher == "aprun"
    assert env.resources[0]["account"] == "proj"

    detect_mod.detect.cache_clear()
