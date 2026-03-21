"""CLI handler for ``lst-tools lastrac``.

Reads an HDF5 base-flow file (grid + flow solution), applies the
configuration from ``lst.cfg``, and writes a LASTRAC-compatible
``meanflow.bin`` binary file.  When ``--debug`` is active a Tecplot
ASCII snapshot of the base flow is also written.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import typer
from cfd_io import StructuredGrid, read_file as cfd_read_file

from lst_tools.config import read_config
from lst_tools.convert import convert_meanflow
from lst_tools.core import Flow, Grid
from lst_tools.data_io.tecplot_ascii import write_tecplot_ascii


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


def _to_2d(arr: np.ndarray, name: str) -> np.ndarray:
    """Normalize 2-D/3-D arrays to a 2-D plane for LST processing."""
    a = np.asarray(arr)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        # cfd-io commonly returns (nx, ny, 1) for 2-D slices; squeeze singleton axis
        if 1 in a.shape:
            b = np.squeeze(a)
            if b.ndim == 2:
                return b
        raise ValueError(f"{name}: expected a single-plane 3-D array, got {a.shape}")
    raise ValueError(f"{name}: expected 2-D or 3-D array, got {a.shape}")


def _load_with_cfd_io(fpath: Path) -> tuple[Grid, Flow, dict]:
    """Read grid/flow via cfd-io and map into lst-tools core containers.

    lst-tools uses (ny, nx) layout.  The data orientation is detected
    by checking whether x[0,0] < x[0,-1] (streamwise increases along
    axis 1).  If not, the arrays are transposed.
    """
    ds = cfd_read_file(fpath)
    grid_raw = ds.grid
    flow_raw = ds.flow
    attrs = ds.attrs or {}

    if not isinstance(grid_raw, StructuredGrid):
        raise TypeError(
            f"lastrac requires a structured grid, got {type(grid_raw).__name__}"
        )

    # get 2-D arrays before deciding on orientation
    x = _to_2d(grid_raw.x, "grid/x")
    y = _to_2d(grid_raw.y, "grid/y")

    # determine orientation: in (ny, nx) convention, x[0,:] spans the
    # streamwise direction so x[0,0] < x[0,-1].  If the opposite is
    # true, the data is (nx, ny) and needs to be transposed.
    do_transpose = x[0, 0] >= x[0, -1]

    logger.debug("%f %f -> do_transpose = %s", x[0, 0], x[0, -1], do_transpose)

    if do_transpose:
        logger.debug("transposing arrays from (nx, ny) to (ny, nx)")
        x = x.T
        y = y.T
    else:
        logger.debug("arrays already in (ny, nx) convention, no transpose needed")

    # z is optional for 2-D LST setup; keep if present
    z = None
    if grid_raw.z is not None:
        z = _to_2d(grid_raw.z, "grid/z")
        if do_transpose:
            z = z.T

    grid = Grid(x=x, y=y, z=z, attrs=attrs)

    # initialize fields dict and convert flow arrays to 2-D planes, transposing if needed
    fields: dict[str, np.ndarray] = {}

    # loop over all flow variables returned by cfd-io; convert to 2-D planes and transpose if needed
    for key, val in flow_raw.items():
        arr = np.asarray(val.data if hasattr(val, "data") else val)
        if arr.ndim in (2, 3):
            field = _to_2d(arr, f"flow/{key}")
            if do_transpose:
                field = field.T
            fields[key] = field

    flow = Flow(grid=grid, fields=fields, attrs=attrs)
    return grid, flow, attrs


# --------------------------------------------------
# main function for the 'lastrac' cli option
# --------------------------------------------------
def cmd_lastrac(
    ctx: typer.Context,
    cfg: Annotated[
        Optional[Path], typer.Option("--cfg", "-c", help="Explicit config file path.")
    ] = None,
) -> None:
    """Convert an HDF5 base-flow file to LASTRAC meanflow format.
\f

    Workflow:

    1. Load the project config (auto-discovered or via ``--cfg``).
    2. Read the HDF5 input file specified by ``config.input_file``.
    3. If ``--debug`` is active, write a Tecplot ASCII file of the
       base flow to ``./debug/debug_base_flow.dat``.
    4. Run ``convert_meanflow()`` to produce ``meanflow.bin``.

    Parameters
    ----------
    ctx : typer.Context
        Typer context carrying the ``debug`` flag from the parent callback.
    cfg : Path | None
        Explicit path to a config file.  When *None*, the standard
        config discovery logic is used (``read_config()``).
    """

    # set debug flag based on parent context (``--debug`` is a global option set in the main callback)
    debug = ctx.obj.get("debug", False) if ctx.obj else False

    # initialize debug_path
    debug_path = None

    try:
        # load config (read_config.py in /config/)
        config = read_config(path=cfg)

        # read base-flow data; prefer cfd-io canonical reader
        fname_hdf5 = Path(config.input_file)
        if not fname_hdf5.exists():
            typer.echo(f"error: {fname_hdf5} not found", err=True)
            raise typer.Exit(1)

        grid, flow, _attrs = _load_with_cfd_io(fname_hdf5)

        # debug output if --debug option is active
        # writes Tecplot ASCII base flow to ./debug/debug_base_flow.dat
        if debug:

            # set debug path
            debug_path = Path("./debug")

            # ensure debug directory exists
            debug_path.mkdir(parents=True, exist_ok=True)

            # write Tecplot ASCII file of the base flow for debugging purposes
            out_file = debug_path / "debug_base_flow.dat"
            ny, nx = grid.shape

            # set variables dict for Tecplot output
            # includes grid coordinates and flow variables
            # note: **flow.fields expands the dictionary to include
            # all flow variables (e.g. velocity components, pressure, etc.)
            variables = {"x": grid.x, "y": grid.y, **flow.fields}

            # write Tecplot ASCII file
            write_tecplot_ascii(
                out_file,
                variables,
                title="debug_base_flow",
                zone="Base Flow",
                fmt=".6e",
            )

            # output debug message for user
            logger.debug("debug file written: %s (%dx%d points)", out_file, nx, ny)

        # convert to LASTRAC meanflow format and write meanflow.bin (in /convert/)
        convert_meanflow(grid, flow, "meanflow.bin", cfg=config, debug_path=debug_path)

        # output for user
        typer.echo(f"{fname_hdf5} -> meanflow.bin")
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("lastrac conversion failed", exc_info=True)
        raise typer.Exit(1)
