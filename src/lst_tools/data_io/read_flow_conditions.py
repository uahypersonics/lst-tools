"""Read legacy ``flow_conditions.dat`` and return a dict of SI flow conditions.

Only extracts the first numeric value on each line.  Unknown lines
are ignored gracefully.
"""

# --------------------------------------------------
# import necessary libraries
# --------------------------------------------------
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# regular expression to match a number (integer or float, with optional exponent)
# --------------------------------------------------
_NUM = re.compile(
    r"""
    [+-]?                # sign
    (?:
        (?:\d+\.\d*|\.\d+|\d+)   # decimal like 1.23 or .23 or 1
        (?:[eE][+-]?\d+)?        # optional exponent
    )
""",
    re.VERBOSE,
)


# --------------------------------------------------
# map substrings found in the flow conditions description to config keys
# --------------------------------------------------
_KEY_MAP = {

    "Gas constant, Rgas": "rgas",
    "heat capacity (p = const.), cp": "cp",
    "heat capacity (V = const.), cv": "cv",
    "heat capacity ratio, gamma": "gamma",
    "Prandtl number, Pr": "pr",
    "freestream Mach number, M": "mach",
    "stagnation pressure, ptot": "pres_0",
    "freestream pressure, pfs": "pres_inf",
    "stagnation temperature, Ttot": "temp_0",
    "freestream temperature, Tfs": "temp_inf",
    "stagnation density, rhotot": "dens_0",
    "freestream density, rhofs": "dens_inf",
    "viscosity, mu": "mu",
    "unit Reynolds number, re1": "re1",
    "freestream velocity, Ufs": "uvel_inf",
    "freestream speed of sound, cfs": "a_inf",
    "reference length scale, lref": "lref",
    "reference time scale, tref": "tref",
    "stagnation enthalpy": "h0",
    "kolmogorov length scale": "eta",
    "kolmogorov time scale": "tau_eta",
}


# map flow-state JSON keys to lst.cfg flow-condition keys
_JSON_KEY_MAP = {
    "mach": "mach",
    "re1": "re1",
    "pr": "pr",
    "gamma": "gamma",
    "cp": "cp",
    "cv": "cv",
    "r_gas": "rgas",
    "pres_stag": "pres_0",
    "temp_stag": "temp_0",
    "pres": "pres_inf",
    "temp": "temp_inf",
    "dens": "dens_inf",
    "uvel": "uvel_inf",
}


# --------------------------------------------------
# read flow-state JSON output
# --------------------------------------------------
def _read_flow_conditions_json(fpath: Path) -> dict[str, float]:
    """Read flow-state JSON values and convert them to lst.cfg keys."""

    # read the JSON document
    with fpath.open("r", encoding="utf-8") as fhandle:
        source_data = json.load(fhandle)

    # extract numeric values from scalar or [value, unit] entries
    flow_conditions: dict[str, float] = {}
    for source_key, config_key in _JSON_KEY_MAP.items():
        source_value = source_data.get(source_key)
        if isinstance(source_value, (list, tuple)) and source_value:
            source_value = source_value[0]

        if isinstance(source_value, (int, float)) and not isinstance(source_value, bool):
            flow_conditions[config_key] = float(source_value)

    # map the transport model to the legacy viscosity-law switch
    transport_model = source_data.get("transport_model")
    if isinstance(transport_model, dict):
        transport_type = transport_model.get("type")
        viscosity_laws = {
            "sutherland": 0,
            "power": 1,
            "power_law": 1,
            "power-law": 1,
        }
        if transport_type in viscosity_laws:
            flow_conditions["visc_law"] = viscosity_laws[transport_type]

    return flow_conditions


# --------------------------------------------------
# find the first numerical entry in the line
# --------------------------------------------------
def _first_number(text: str) -> float | None:

    m = _NUM.search(text)

    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


# --------------------------------------------------
# read the flow conditions file
# --------------------------------------------------
def read_flow_conditions(fpath: str | Path) -> dict[str, float]:
    """
    Parse a legacy flow_conditions.dat and return a dict with keys compatible
    with your config's `flow_conditions` section.

    We favor the first numeric value on the line (expected SI). Lines with
    multiple numbers like "ptot: <Pa> / <psi>" will yield the first (Pa).

    Parameters
    ----------
    fpath : str | Path
        Path to the flow_conditions.dat file.

    Returns
    -------
    dict[str, float]
        A mapping like:
        {
          "mach": 5.3, "re1": 1.79e7, "gamma": 1.4, "cp": 1005.0, "cv": 717.9,
          "rgas": 287.15, "pres_0": ..., "temp_0": ..., "pres_inf": ...,
          "temp_inf": ..., "dens_inf": ..., "uvel_inf": ...
        }
        Only keys that were found are returned.
    """

    # ensure fpath is a Path object
    fpath = Path(fpath)

    # read flow-state JSON files with their structured schema
    if fpath.suffix.lower() == ".json":
        return _read_flow_conditions_json(fpath)

    # generate an empty dictionary to return flow_conditions dict
    out_dict: dict[str, float] = {}

    with fpath.open("r", encoding="utf-8", errors="ignore") as fhandle:
        # read line by line
        for raw in fhandle:
            # strip all spaces
            line = raw.strip()

            # skip lines that do not have a colon (label value separator):
            if not line or ":" not in line:
                continue

            # split the line at colon symbol and strip spaces to get the label and value parts
            label = line.split(":", 1)[0].strip()
            val_str = line.split(":", 1)[1].strip()

            # find the matching config key by scanning the KEY_MAP descriptions
            key = None
            for description, config_key in _KEY_MAP.items():
                if description in label:
                    key = config_key
                    break

            if key is None:
                logger.warning("no key found for line: %s", line)
                continue

            # extract the first number from the value string
            val = _first_number(val_str)

            if val is None:
                continue

            out_dict[key] = val

    return out_dict



