"""Additional tests for lst_tools.hpc._detect coverage gaps."""

from __future__ import annotations

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


def test_detect_cpus_uses_profile_short_circuit() -> None:
    """Return profile CPUs directly without scheduler-specific probing."""
    profile = ClusterProfile(name="demo", scheduler=Scheduler.SLURM, cpus_per_node=94)

    cpus, hist = detect_mod._detect_cpus(Scheduler.SLURM, profile)

    assert cpus == 94
    assert hist == {94: 1}


def test_detect_resources_profile_command_uses_matching_parser(monkeypatch) -> None:
    """Use parser selected by profile.resource_cmd."""
    profile = ClusterProfile(name="puma", resource_cmd="va")

    monkeypatch.setattr(detect_mod.subprocess, "check_output", lambda *a, **k: "raw")
    monkeypatch.setattr(detect_mod, "parse_va_output", lambda out: [{"account": "A"}])

    rows = detect_mod._detect_resources(profile)

    assert rows == [{"account": "A"}]


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
