"""Configure HPC job parameters.

This is the thin public entry point.  Heavy lifting lives in:
- :mod:`._detect`   — cached environment probing
- :mod:`._resolve`  — merge env + user config → frozen job
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from typing import Any

from ._detect import detect
from ._resolve import ResolvedJob, resolve


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function: hpc_configure
# --------------------------------------------------
def hpc_configure(
    cfg: dict[str, Any],
    set_defaults: bool = False,
    *,
    nodes_override: int | None = None,
    time_override: float | str | None = None,
) -> ResolvedJob:
    """Detect environment and build a :class:`ResolvedJob`.

    Parameters
    ----------
    cfg:
        Full application config (``Config`` dataclass *or* plain dict).
    set_defaults:
        Apply generous defaults (``nodes=10``, ``time=1.0``).
    nodes_override, time_override:
        Programmatic overrides — used by callers like *tracking_setup*
        that compute optimal values before building the script.
    """
    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()

    logger.info("configuring hpc parameters")

    env = detect()
    user_hpc = cfg.get("hpc", {})

    return resolve(
        env,
        user_hpc,
        set_defaults=set_defaults,
        nodes_override=nodes_override,
        time_override=time_override,
    )
