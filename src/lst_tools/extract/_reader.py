"""Tecplot FE-quadrilateral BLOCK ASCII reader."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
import re
import logging

import numpy as np

from ._types import TecplotUnstructuredData


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# constants
# --------------------------------------------------

# mapping from common Tecplot variable names to canonical names
FIELD_NAME_MAP = {
    "uvel": "u", "u": "u", "u-velocity": "u",
    "vvel": "v", "v": "v", "v-velocity": "v",
    "wvel": "w", "w": "w", "w-velocity": "w",
    "temp": "t", "t": "t", "temperature": "t",
    "pres": "p", "p": "p", "pressure": "p",
    "dens": "rho", "rho": "rho", "density": "rho",
}


# --------------------------------------------------
# Tecplot FE-quad reader
# --------------------------------------------------
def read_fequad_block_tecplot(path: str | Path) -> TecplotUnstructuredData:
    """Read a FEQUADRILATERAL BLOCK Tecplot ASCII file.

    The file must have:
    - Line 0: ``VARIABLES = x, y, z, ...``
    - Line 1: ``ZONE N=<nodes>, E=<elements>, ...``
    - Lines 2-3: ``DATAPACKING=BLOCK`` and ``ZONETYPE=FEQUADRILATERAL``
    - Numeric body: nodal variables (N values each), then cell-centered
      variables (E values each), then connectivity (E rows × 4 columns).

    Args:
        path: Input Tecplot ASCII file path.

    Returns:
        Parsed nodal coordinates, cell-centered fields, and connectivity.

    Raises:
        ValueError: If the zone header cannot be parsed.
    """

    # convert to Path object
    file_path = Path(path)

    # read full file text and split into lines
    text = file_path.read_text()
    lines = text.splitlines()

    logger.debug("read %d lines from %s", len(lines), file_path)

    # parse variable names - may be comma-separated on one line or one per line
    variable_names: list[str] = []
    header_end_line = 0
    in_variables = False
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper()

        if upper.startswith("VARIABLES"):
            in_variables = True
            after_eq = stripped.split("=", 1)[1].strip() if "=" in stripped else ""
            if after_eq:
                for var in re.findall(r'"([^"]+)"', after_eq):
                    variable_names.append(var)
                if not after_eq.startswith('"'):
                    for var in after_eq.replace(",", " ").split():
                        if var.strip():
                            variable_names.append(var.strip())
            continue

        if in_variables and stripped.startswith('"'):
            for var in re.findall(r'"([^"]+)"', stripped):
                variable_names.append(var)
            continue

        if upper.startswith("ZONE"):
            in_variables = False
            header_end_line = line_num
            break

    logger.debug("parsed %d variable names: %s", len(variable_names), variable_names)

    # find zone header and parse node/element counts
    n_node = 0
    n_elem = 0
    data_start_line = header_end_line
    for line_num in range(header_end_line, min(header_end_line + 10, len(lines))):
        line = lines[line_num]
        # only update N/E counts while they haven't been found yet — prevents
        # spurious matches like the trailing 'E' in SOLUTIONTIME=5.xxx overwriting
        # the correct n_elem already parsed from the ZONE line
        if n_node == 0:
            node_match = re.search(r"(?:Nodes|N)\s*=\s*(\d+)", line, re.IGNORECASE)
            if node_match:
                n_node = int(node_match.group(1))
        if n_elem == 0:
            elem_match = re.search(r"(?:Elements|E)\s*=\s*(\d+)", line, re.IGNORECASE)
            if elem_match:
                n_elem = int(elem_match.group(1))
        # data starts after DATAPACKING line
        if "DATAPACKING" in line.upper():
            data_start_line = line_num + 1
        # skip DT line if present
        if line.strip().upper().startswith("DT"):
            data_start_line = line_num + 1

    if n_node == 0 or n_elem == 0:
        raise ValueError("Could not parse N and E from Tecplot zone header")

    # determine which variables are cell-centered (via VARLOCATION or default)
    # handle compound forms like ([1-3]=NODAL,[4-12]=CELLCENTERED) by first
    # extracting the full parenthetical value, then scanning for all CELLCENTERED ranges
    cell_centered_indices: set[int] = set()
    for line_num in range(header_end_line, min(header_end_line + 10, len(lines))):
        line = lines[line_num].upper()
        vl_match = re.search(r"VARLOCATION\s*=\s*\(([^)]+)\)", line)
        if vl_match:
            vl_content = vl_match.group(1)
            for cc_match in re.finditer(r"\[(\d+)-(\d+)\]\s*=\s*CELLCENTERED", vl_content):
                start_idx = int(cc_match.group(1))
                end_idx = int(cc_match.group(2))
                cell_centered_indices.update(range(start_idx, end_idx + 1))

    logger.debug("n_node=%d, n_elem=%d, data starts at line %d", n_node, n_elem, data_start_line)
    logger.debug("cell-centered var indices: %s", cell_centered_indices)

    # tokenize the numeric body after the header using a regex
    # that matches both fixed-point and scientific-notation numbers
    body_text = "\n".join(lines[data_start_line:])
    token_pattern = r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?"
    numeric_tokens = [float(token) for token in re.findall(token_pattern, body_text)]

    # split the token stream into nodal and cell-centered variable blocks
    # index is 1-based in VARLOCATION
    index = 0
    nodal: dict[str, np.ndarray] = {}
    cell: dict[str, np.ndarray] = {}

    for var_idx, name in enumerate(variable_names):
        tecplot_idx = var_idx + 1  # 1-based for VARLOCATION
        if tecplot_idx in cell_centered_indices:
            cell[name] = np.asarray(numeric_tokens[index:index + n_elem], dtype=float)
            index += n_elem
        else:
            nodal[name] = np.asarray(numeric_tokens[index:index + n_node], dtype=float)
            index += n_node

    # read the FE quadrilateral connectivity table (1-based node indices)
    connectivity_values = numeric_tokens[index:index + 4 * n_elem]
    connectivity = np.asarray(connectivity_values, dtype=int).reshape(n_elem, 4)

    # apply field name mapping to canonical names
    def remap_fields(fields: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        remapped: dict[str, np.ndarray] = {}
        for name, data in fields.items():
            canonical = FIELD_NAME_MAP.get(name.lower(), name)
            remapped[canonical] = data
        return remapped

    nodal = remap_fields(nodal)
    cell = remap_fields(cell)

    # add w=0 for 2D flows if not present
    if "w" not in nodal and "w" not in cell:
        if nodal:
            sample_arr = next(iter(nodal.values()))
            nodal["w"] = np.zeros(sample_arr.size, dtype=float)
        elif cell:
            sample_arr = next(iter(cell.values()))
            cell["w"] = np.zeros(sample_arr.size, dtype=float)

    return TecplotUnstructuredData(
        nodal=nodal,
        cell=cell,
        connectivity=connectivity,
    )
