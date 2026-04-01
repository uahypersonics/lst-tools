"""Cluster profile registry — single source of site-specific knowledge.

Adding a new cluster = adding one ``_register()`` call below.
Nothing else in the codebase needs to change.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# --------------------------------------------------
# scheduler enum (moved here from the old scheduler.py)
# --------------------------------------------------
class Scheduler(Enum):
    """Supported job-scheduler types."""

    SLURM = "slurm"
    PBS = "pbs"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # noqa: D105
        return self.value


# --------------------------------------------------
# cluster profile dataclass
# --------------------------------------------------
@dataclass(frozen=True)
class ClusterProfile:
    """Everything we know about a specific HPC cluster."""

    name: str
    login_aliases: tuple[str, ...] = ()
    scheduler: Scheduler = Scheduler.UNKNOWN
    cpus_per_node: int = 1
    default_partition: str = "standard"
    preferred_launcher: str = "mpirun"
    resource_cmd: str | None = None
    modules: tuple[str, ...] = ()
    mem_per_cpu: str | None = None
    extra_sbatch: tuple[str, ...] = ()
    extra_pbs: tuple[str, ...] = ()


# --------------------------------------------------
# cluster profile registry
# maps the raw hostname (or login alias) to a ClusterProfile
# --------------------------------------------------
_PROFILES: dict[str, ClusterProfile] = {}

# helper to register a profile under its name and login aliases
def _register(p: ClusterProfile) -> None:
    """Register *p* under its ``name`` and every ``login_alias``."""
    _PROFILES[p.name] = p
    for alias in p.login_aliases:
        _PROFILES[alias] = p

# helper to look up a profile by hostname
def lookup(hostname: str) -> ClusterProfile | None:
    """Return the profile for *hostname*, or ``None`` for unknown clusters."""
    return _PROFILES.get(hostname)


# --------------------------------------------------
# known clusters
# --------------------------------------------------

# puma
_register(
    ClusterProfile(
        name="puma",
        login_aliases=("junonia", "wentletrap"),
        scheduler=Scheduler.SLURM,
        cpus_per_node=94,
        default_partition="standard",
        preferred_launcher="mpirun",
        resource_cmd="va",
        modules=(
            "gnu13/13.2.0",
            "openmpi5/5.0.5",
            "openblas/0.3.21",
        ),
        mem_per_cpu="1gb",
    )
)

# carpenter
_register(
    ClusterProfile(
        name="carpenter",
        login_aliases=(),
        scheduler=Scheduler.PBS,
        cpus_per_node=48,
        default_partition="standard",
        preferred_launcher="aprun",
        resource_cmd="show_usage",
        modules=(),
        mem_per_cpu=None,
    )
)

# nautilus
_register(
    ClusterProfile(
        name="nautilus",
        login_aliases=(),
        scheduler=Scheduler.PBS,
        cpus_per_node=64,
        default_partition="standard",
        preferred_launcher="mpirun",
        resource_cmd="show_usage",
        modules=(),
        mem_per_cpu=None,
    )
)

# warhawk
_register(
    ClusterProfile(
        name="warhawk",
        login_aliases=(),
        scheduler=Scheduler.PBS,
        cpus_per_node=128,
        preferred_launcher="aprun",
        resource_cmd="show_usage",
        modules=(),
        mem_per_cpu=None,
        extra_pbs=(
            'setenv LD_LIBRARY_PATH "$HOME/lib:$LD_LIBRARY_PATH"',
            "setenv UCX_WARN_UNUSED_ENV_VARS n",
            "module swap cray-mpich cray-mpich-ucx",
        ),
    )
)
