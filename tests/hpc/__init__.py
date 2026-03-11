"""Tests for lst_tools.hpc._profiles."""

from __future__ import annotations

from lst_tools.hpc._profiles import ClusterProfile, Scheduler, lookup


class TestSchedulerEnum:
    def test_str_values(self):
        assert str(Scheduler.SLURM) == "slurm"
        assert str(Scheduler.PBS) == "pbs"
        assert str(Scheduler.UNKNOWN) == "unknown"

    def test_enum_identity(self):
        assert Scheduler.SLURM is Scheduler.SLURM
        assert Scheduler("slurm") is Scheduler.SLURM


class TestLookup:
    def test_puma_canonical(self):
        p = lookup("puma")
        assert p is not None
        assert p.name == "puma"
        assert p.scheduler is Scheduler.SLURM
        assert p.cpus_per_node == 94

    def test_puma_login_alias_junonia(self):
        p = lookup("junonia")
        assert p is not None
        assert p.name == "puma"

    def test_puma_login_alias_wentletrap(self):
        p = lookup("wentletrap")
        assert p is not None
        assert p.name == "puma"

    def test_carpenter(self):
        p = lookup("carpenter")
        assert p is not None
        assert p.scheduler is Scheduler.PBS
        assert p.cpus_per_node == 48
        assert p.preferred_launcher == "aprun"

    def test_unknown_returns_none(self):
        assert lookup("totally_unknown_cluster") is None

    def test_profiles_are_frozen(self):
        p = lookup("puma")
        assert p is not None
        try:
            p.name = "hacked"  # type: ignore[misc]
            assert False, "should have raised"
        except AttributeError:
            pass


class TestClusterProfileDefaults:
    def test_default_scheduler_is_unknown(self):
        p = ClusterProfile(name="test")
        assert p.scheduler is Scheduler.UNKNOWN

    def test_default_cpus_per_node(self):
        p = ClusterProfile(name="test")
        assert p.cpus_per_node == 1

    def test_puma_modules(self):
        p = lookup("puma")
        assert p is not None
        assert len(p.modules) == 3
        assert "gnu13/13.2.0" in p.modules

    def test_puma_mem_per_cpu(self):
        p = lookup("puma")
        assert p is not None
        assert p.mem_per_cpu == "1gb"
