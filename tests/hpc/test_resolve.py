"""Tests for lst_tools.hpc._resolve — hand-built DetectedEnv → ResolvedJob."""

from __future__ import annotations

import pytest

from lst_tools.hpc._detect import DetectedEnv
from lst_tools.hpc._profiles import ClusterProfile, Scheduler
from lst_tools.hpc._resolve import ResolvedJob, resolve, _pick_best_account


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _make_env(
    *,
    hostname: str = "generic",
    scheduler: Scheduler = Scheduler.SLURM,
    launcher: str | None = "mpirun",
    cpus_per_node: int = 64,
    resources: tuple[dict, ...] = (),
    profile: ClusterProfile | None = None,
) -> DetectedEnv:
    return DetectedEnv(
        hostname=hostname,
        scheduler=scheduler,
        launcher=launcher,
        cpus_per_node=cpus_per_node,
        cpus_histogram={cpus_per_node: 1} if cpus_per_node else {},
        resources=resources,
        profile=profile,
    )


_RESOURCE_ROW = {
    "system": "puma",
    "account": "myacct",
    "allocated": 100000,
    "used": 10000,
    "remaining": 90000,
    "percent_remain": 90.0,
    "partition": "standard",
    "qos": None,
}


# ------------------------------------------------------------------
# ResolvedJob basics
# ------------------------------------------------------------------
class TestResolvedJob:
    def test_frozen(self):
        job = ResolvedJob()
        with pytest.raises(AttributeError):
            job.nodes = 42  # type: ignore[misc]

    def test_to_dict_excludes_none(self):
        job = ResolvedJob(nodes=4, account="acct")
        d = job.to_dict()
        assert "nodes" in d
        assert d["nodes"] == 4
        # None fields like constraint should be absent
        assert "constraint" not in d

    def test_get_missing(self):
        job = ResolvedJob()
        assert job.get("nonexistent", "fallback") == "fallback"


# ------------------------------------------------------------------
# _pick_best_account
# ------------------------------------------------------------------
class TestPickBestAccount:
    def test_empty_resources(self):
        assert _pick_best_account(()) is None

    def test_single_resource(self):
        best = _pick_best_account((_RESOURCE_ROW,))
        assert best is not None
        assert best["account"] == "myacct"

    def test_prefers_more_remaining(self):
        r1 = {**_RESOURCE_ROW, "account": "small", "remaining": 100}
        r2 = {**_RESOURCE_ROW, "account": "big", "remaining": 50000}
        best = _pick_best_account((r1, r2))
        assert best["account"] == "big"

    def test_prefers_high_priority(self):
        std = {**_RESOURCE_ROW, "account": "std", "remaining": 50000, "partition": "standard"}
        hp = {**_RESOURCE_ROW, "account": "hp", "remaining": 10000, "partition": "high_priority"}
        best = _pick_best_account((std, hp))
        assert best["account"] == "hp"


# ------------------------------------------------------------------
# resolve()
# ------------------------------------------------------------------
class TestResolve:
    def test_basic_no_user_config(self):
        env = _make_env(resources=(_RESOURCE_ROW,))
        job = resolve(env)
        assert job.ntasks_per_node == 64
        assert job.account == "myacct"
        assert job.scheduler == "slurm"
        assert job.nodes is None  # set_defaults=False

    def test_set_defaults(self):
        env = _make_env(resources=(_RESOURCE_ROW,))
        job = resolve(env, set_defaults=True)
        assert job.nodes == 10
        assert job.time == 1.0

    def test_nodes_override(self):
        env = _make_env(resources=(_RESOURCE_ROW,))
        job = resolve(env, nodes_override=5)
        assert job.nodes == 5

    def test_time_override(self):
        env = _make_env(resources=(_RESOURCE_ROW,))
        job = resolve(env, time_override=2.5)
        assert job.time == 2.5

    def test_user_config_account(self):
        env = _make_env(resources=(_RESOURCE_ROW,))
        job = resolve(env, {"account": "explicit"})
        assert job.account == "explicit"

    def test_partition_frontier_for_fx(self):
        # "FX" in account triggers frontier partition *only* when
        # the resource row doesn't already specify a partition
        row = {**_RESOURCE_ROW, "account": "myFXacct", "partition": None}
        env = _make_env(resources=(row,))
        job = resolve(env)
        assert job.partition == "frontier"

    def test_fname_run_script_slurm(self):
        env = _make_env(hostname="puma")
        job = resolve(env)
        assert job.fname_run_script == "run.slurm.puma"

    def test_fname_run_script_unknown(self):
        env = _make_env(scheduler=Scheduler.UNKNOWN)
        job = resolve(env)
        assert job.fname_run_script is None

    def test_profile_modules(self):
        from lst_tools.hpc._profiles import lookup

        profile = lookup("puma")
        env = _make_env(hostname="puma", profile=profile)
        job = resolve(env)
        assert len(job.modules) == 3

    def test_profile_mem_per_cpu(self):
        from lst_tools.hpc._profiles import lookup

        profile = lookup("puma")
        env = _make_env(hostname="puma", profile=profile)
        job = resolve(env)
        assert job.mem_per_cpu == "1gb"
