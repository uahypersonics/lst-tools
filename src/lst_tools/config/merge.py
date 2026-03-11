"""Utilities for merging nested configuration dictionaries."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import copy
import logging
from pathlib import Path
from lst_tools.config.schema import Config
import lst_tools.data_io


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# function to merge dictionaries
# --------------------------------------------------
def merge_dicts(base: dict, override: dict) -> dict:
    """Return a new dict with *override* merged into *base* (override wins).

    Nested sub-dicts are merged recursively; all other values from
    *override* replace those in *base*.  Neither input is mutated.

    Args:
        base (dict): Base dictionary.
        override (dict): Dictionary whose values take priority.

    Returns:
        dict: A new merged dictionary.
    """

    # start with a deep copy of base, then merge override into it
    out = copy.deepcopy(base)

    # loop over keys in override
    for key, val in override.items():
        # check if key is in out and both out[key] and val are dicts
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            # if both are dicts merge recursively
            out[key] = merge_dicts(out[key], val)
        else:
            # override value wins; copy it into out
            out[key] = copy.deepcopy(val)
    # return the merged dict
    return out


# --------------------------------------------------
# function to merge flow conditions into config defaults
# --------------------------------------------------
def merge_flow_defaults(cfg_in: dict, flow_path: Path | None) -> dict:
    """Merge ``flow_conditions.dat`` values into *cfg_in* dictionary.

    If *flow_path* exists, read it, keep only keys recognised by the
    config schema, and merge them into *cfg_in* via ``merge_dicts``.
    Neither input is mutated.

    Args:
        cfg_in (dict): Base configuration dictionary.
        flow_path (Path | None): Path to a ``flow_conditions.dat`` file.
            If *None* or the file does not exist, *cfg_in* is returned unchanged.

    Returns:
        dict: A new dictionary with flow conditions merged in.
    """

    # start with a deep copy of cfg_in
    cfg_out = copy.deepcopy(cfg_in)

    # if flow_path exists, read it and merge recognised keys
    if flow_path is not None and flow_path.exists():
        try:
            # read flow conditions from file
            flow_data = lst_tools.data_io.read_flow_conditions(flow_path)

            if isinstance(flow_data, dict) and flow_data:
                # get recognised keys from the config schema defaults
                flow_conditions_keys = Config().to_dict().get("flow_conditions", {}).keys()

                # filter to only recognised keys
                flow_conditions = {}
                for key, val in flow_data.items():
                    if key in flow_conditions_keys:
                        flow_conditions[key] = val

                # merge into cfg_in as a flow_conditions override
                cfg_out = merge_dicts(cfg_in, {"flow_conditions": flow_conditions})

        except (OSError, ValueError, KeyError) as e:
            logger.warning("failed to read flow_conditions.dat: %s", e)

    return cfg_out
