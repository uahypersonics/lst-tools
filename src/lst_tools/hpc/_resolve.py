"""Merge detected environment with user configuration into a frozen job.

``resolve()`` is the single place where defaults are applied
and accounts are selected.  The resulting ``ResolvedJob`` is
immutable and carries everything ``_templates`` needs to render a
script.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Any

from ._detect import DetectedEnv
from ._profiles import Scheduler

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# frozen output
# ------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedJob:
    """All parameters required to render an HPC batch script."""

    nodes: int | None = None
    ntasks_per_node: int | None = None
    cpus_per_task: int | None = None
    time: float | str = 1.0
    partition: str | None = None
    qos: str | None = None
    account: str | None = None
    mem_per_cpu: str | None = None
    constraint: str | None = None
    job_name: str = "lst"
    output: str | None = None
    workdir: str | None = None
    hostname: str | None = None
    scheduler: str | None = None
    launcher: str | None = None
    modules: tuple[str, ...] = ()
    fname_run_script: str | None = None

    # convenience helpers (same surface as the old HPCcfg)
    def to_dict(self) -> dict[str, Any]:
        """Return a dict of all non-``None`` fields."""

        # convert dataclass to dict and filter out None values
        dict_source = asdict(self)
        dict_out = {}

        # step through all fields and only include those that are not None
        for key, value in dict_source.items():
            if value is not None:
                dict_out[key] = value

        # return filtered dict
        return dict_out

    # allow dict-like access with fallback to attributes
    def get(self, key: str, default: Any = None) -> Any:
        """Attribute access with a fallback, like ``dict.get``."""
        return getattr(self, key, default)


# ------------------------------------------------------------------
# account selection
# ------------------------------------------------------------------
def _select_account(
    resources: tuple[dict[str, object], ...],
) -> dict[str, object] | None:
    """Return the resource row with the most remaining hours."""
    if not resources:
        return None

    candidates = [r for r in resources if r.get("remaining", 0) > 0]

    if not candidates:
        logger.warning("all accounts exhausted -> selecting the least exhausted")
        return max(
            resources,
            key=lambda r: (r.get("remaining", 0), r.get("percent_remain", 0.0)),
        )

    # prefer high_priority partitions
    hp = [r for r in candidates if r.get("partition") == "high_priority"]
    if hp:
        logger.info("prioritizing high_priority partition")
        return max(
            hp, key=lambda r: (r.get("remaining", 0), r.get("percent_remain", 0.0))
        )

    return max(
        candidates,
        key=lambda r: (r.get("remaining", 0), r.get("percent_remain", 0.0)),
    )


# ------------------------------------------------------------------
# public API
# ------------------------------------------------------------------
def resolve(
    env: DetectedEnv,
    user_hpc: dict[str, Any] | None = None,
    *,
    set_defaults: bool = False,
    nodes_override: int | None = None,
    time_override: float | str | None = None,
) -> ResolvedJob:
    """Build a frozen :class:`ResolvedJob` from *env* and user settings.

    Parameters
    ----------
    env:
        Result of :func:`_detect.detect`.
    user_hpc:
        The ``[hpc]`` section from the user config file (may be ``None``).
    set_defaults:
        When ``True`` (interactive CLI), apply generous defaults for
        ``nodes`` and ``time`` so the script is immediately runnable.
    nodes_override:
        If given, override the user-config / default node count.
    time_override:
        If given, override the user-config / default wall-time.
    """
    u = user_hpc or {}

    # scheduler / launcher
    scheduler = env.scheduler
    launcher = env.launcher

    # cpus per node
    ntasks_per_node = env.cpus_per_node or 1

    # nodes
    nodes: int | None = nodes_override if nodes_override is not None else u.get("nodes")
    if nodes is None and set_defaults:
        nodes = 10

    # time
    time: float | str | None = time_override if time_override is not None else u.get("time")
    if time is None and set_defaults:
        time = 1.0

    # account
    account: str | None = u.get("account")
    partition: str | None = u.get("partition")
    qos: str | None = None

    if account is None:
        best = _select_account(env.resources)
        if best is not None:
            account = best["account"]  # type: ignore[assignment]
            if best.get("partition"):
                partition = best["partition"]  # type: ignore[assignment]
            if best.get("qos"):
                qos = best["qos"]  # type: ignore[assignment]

            part_info = f" (partition: {best.get('partition', 'unknown')})" if best.get("partition") else ""
            qos_info = f" (QOS: {best.get('qos')})" if best.get("qos") else ""
            logger.info(
                "using account '%s' (remaining: %s h, %.2f%% left)%s%s.",
                account,
                best["remaining"],
                best["percent_remain"],
                part_info,
                qos_info,
            )
        else:
            logger.warning("no accounts detected; leaving account unset")
    else:
        known = {r["account"] for r in env.resources}
        if env.resources and account not in known:
            logger.warning("account '%s' not found in detected resources.", account)

    # partition default
    if partition is None:
        acct_upper = (account or "").upper()
        partition = "frontier" if "FX" in acct_upper else "standard"

    # modules / mem_per_cpu from profile
    profile = env.profile
    modules: tuple[str, ...] = profile.modules if profile else ()
    mem_per_cpu: str | None = profile.mem_per_cpu if profile else None

    # launcher override from profile (e.g. carpenter → aprun)
    if profile is not None:
        launcher = profile.preferred_launcher

    # run-script filename
    if scheduler is not Scheduler.UNKNOWN:
        fname = f"run.{scheduler}.{env.hostname}"
    else:
        fname = None
        logger.warning("cannot determine scheduler type => likely local system")

    return ResolvedJob(
        nodes=nodes,
        ntasks_per_node=ntasks_per_node,
        time=time if time is not None else 1.0,
        partition=partition,
        qos=qos,
        account=account,
        mem_per_cpu=mem_per_cpu,
        hostname=env.hostname,
        scheduler=str(scheduler),
        launcher=launcher,
        modules=modules,
        fname_run_script=fname,
        job_name="lst",
    )
