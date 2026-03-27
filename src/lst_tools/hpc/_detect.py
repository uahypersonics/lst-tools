"""Cached, one-shot environment detection.

``detect()`` probes the environment *once* and caches the result for
the lifetime of the process.  All subprocess calls live here.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

import functools
import logging
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass

from ._parsers import (
    _most_common_int,
    parse_lscpu_cpus,
    parse_pbs_nodefile,
    parse_pbsnodes_cpus,
    parse_show_usage_output,
    parse_sinfo_cpus,
    parse_slurm_cpus_env,
    parse_va_output,
)
from ._profiles import ClusterProfile, Scheduler, lookup

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# frozen result of environment probing
# --------------------------------------------------
@dataclass(frozen=True)
class DetectedEnv:
    """Immutable snapshot of everything the machine told us."""

    hostname: str
    scheduler: Scheduler
    launcher: str | None
    cpus_per_node: int | None
    cpus_histogram: dict[int, int]
    resources: tuple[dict[str, object], ...]
    profile: ClusterProfile | None


# --------------------------------------------------
# internal function to detect hostname
# --------------------------------------------------
def _detect_hostname() -> str:
    raw = os.getenv("HOSTNAME") or socket.gethostname()
    name = raw.split(".")[0]
    name = re.sub(r"\d+$", "", name).lower()
    return name


# --------------------------------------------------
# internal function to detect scheduler type
# --------------------------------------------------
def _detect_scheduler() -> Scheduler:
    if os.getenv("SLURM_JOB_ID") or shutil.which("squeue") or shutil.which("sbatch"):
        return Scheduler.SLURM
    if (
        os.getenv("PBS_JOBID")
        or shutil.which("qstat")
        or shutil.which("qsub")
        or shutil.which("pbsnodes")
    ):
        return Scheduler.PBS
    return Scheduler.UNKNOWN


# --------------------------------------------------
# internal function to detect preferred launcher
# --------------------------------------------------
def _detect_launcher() -> str | None:
    for name in ("mpirun", "mpiexec", "aprun", "srun"):
        p = shutil.which(name)
        if p is not None:
            return p.split("/")[-1]
    return None


# --------------------------------------------------
# internal functions to detect CPUs per node for slurm
# --------------------------------------------------
def _detect_cpus_slurm() -> tuple[int | None, dict[int, int]]:
    env = os.environ

    raw = env.get("SLURM_JOB_CPUS_PER_NODE")
    if raw:
        parts = parse_slurm_cpus_env(raw)
        if parts:
            return _most_common_int(parts)

    cpus_on_node = env.get("SLURM_CPUS_ON_NODE")
    if cpus_on_node:
        try:
            val = int(cpus_on_node)
            return val, {val: 1}
        except ValueError:
            pass

    try:
        out = subprocess.check_output(["sinfo", "-h", "-o", "%n %c"], text=True)
        cpus = parse_sinfo_cpus(out)
        if cpus:
            return _most_common_int(cpus)
    except Exception:
        pass

    try:
        out = subprocess.check_output(["lscpu"], text=True)
        val = parse_lscpu_cpus(out)
        if val is not None:
            return val, {val: 1}
    except Exception:
        pass

    return None, {}


# --------------------------------------------------
# internal function to detect CPUs per node for PBS
# --------------------------------------------------
def _detect_cpus_pbs() -> tuple[int | None, dict[int, int]]:
    nodefile = os.environ.get("PBS_NODEFILE")
    if nodefile and os.path.exists(nodefile):
        try:
            text = open(nodefile, encoding="utf-8").read()  # noqa: SIM115
            vals = parse_pbs_nodefile(text)
            if vals:
                return _most_common_int(vals)
        except Exception:
            pass

    for cmd in (["pbsnodes", "-ajS"], ["pbsnodes", "-a"]):
        try:
            out = subprocess.check_output(cmd, text=True)
        except Exception:
            continue
        cpus = parse_pbsnodes_cpus(out)
        if cpus:
            return _most_common_int(cpus)

    try:
        out = subprocess.check_output(["lscpu"], text=True)
        val = parse_lscpu_cpus(out)
        if val is not None:
            return val, {val: 1}
    except Exception:
        pass

    return None, {}


# --------------------------------------------------
# internal function to detect CPUs per node based on scheduler and profile
# --------------------------------------------------
def _detect_cpus(
    scheduler: Scheduler, profile: ClusterProfile | None
) -> tuple[int | None, dict[int, int]]:
    """Return ``(cpus_per_node, histogram)`` using profile or live probing."""
    # profile short-circuits expensive subprocess calls
    if profile is not None:
        known = profile.cpus_per_node
        logger.info(
            "Using known configuration for %s: %d CPUs per node",
            profile.name,
            known,
        )
        return known, {known: 1}

    if scheduler is Scheduler.SLURM:
        return _detect_cpus_slurm()
    if scheduler is Scheduler.PBS:
        return _detect_cpus_pbs()
    return None, {}


# --------------------------------------------------
# internal function to detect HPC resources (allocations)
# --------------------------------------------------
def _detect_resources(
    profile: ClusterProfile | None,
) -> list[dict[str, object]]:
    """Probe available HPC allocations."""
    cmd: list[str]
    if profile is not None and profile.resource_cmd:
        cmd = [profile.resource_cmd]
        parser = parse_va_output if profile.resource_cmd == "va" else parse_show_usage_output
    else:
        # fall back: try show_usage (more common)
        cmd = ["show_usage"]
        parser = parse_show_usage_output

    try:
        out = subprocess.check_output(cmd, text=True)
        return parser(out)
    except FileNotFoundError:
        logger.warning(
            "%s command not found => cannot detect resources", " ".join(cmd)
        )
        return []
    except subprocess.CalledProcessError as e:
        logger.warning("%s failed: %s", " ".join(cmd), e)
        return []


# --------------------------------------------------
# public entry point (cached)
# --------------------------------------------------
@functools.cache
def detect() -> DetectedEnv:
    """Probe the runtime environment (cached for the process lifetime)."""
    hostname = _detect_hostname()

    # get cluster profile (if registered)
    profile = lookup(hostname)

    # resolve hostname to known cluster name when a profile matches
    if profile is not None:
        hostname = profile.name

    scheduler = profile.scheduler if profile else _detect_scheduler()
    launcher = (
        profile.preferred_launcher if profile else _detect_launcher()
    )
    cpus, hist = _detect_cpus(scheduler, profile)
    resources = _detect_resources(profile)

    # save the detected environment in a frozen dataclass
    env = DetectedEnv(
        hostname=hostname,
        scheduler=scheduler,
        launcher=launcher,
        cpus_per_node=cpus,
        cpus_histogram=hist,
        resources=tuple(resources),
        profile=profile,
    )

    # log the detected environment for debugging purposes
    logger.info("host name: %s", env.hostname)
    logger.info("scheduler type: %s", env.scheduler)
    logger.info("number of cpus per node: %s", env.cpus_per_node)

    if resources:
        logger.info("%-15s %10s %8s", "Account", "Alloc", "%Remain")
        for row in resources:
            logger.info(
                "%-15s %10s %8.2f",
                row["account"],
                row["allocated"],
                row["percent_remain"],
            )
    else:
        logger.warning(
            "no resources detected => continuing with a dummy configuration"
        )

    return env
