"""CLI handler for ``lst-tools extract``.

Reads a Tecplot FE-quadrilateral BLOCK ASCII symmetry slice, extracts
wall-normal profiles at user-specified streamwise stations via barycentric
interpolation, and writes an HDF5 baseflow file compatible with the
``lst_next_gen`` stability solver.

Flow parameters (Mach, T_inf) are resolved in priority order:
  CLI flag > ``[flow_conditions]`` in ``lst.cfg`` > built-in default.

Output paths are resolved in priority order:
  CLI flag > ``[extract]`` in ``lst.cfg`` > derived from input file location.

Example ``lst.cfg`` snippet::

    [flow_conditions]
    mach = 6.0
    temp_inf = 50.0

    [extract]
    input_file = "symmetry_normal.dat"
    hdf5_out = "baseflow.hdf5"
    stations = [0.0025, 0.005, 0.010, 0.015, 0.020, 0.025]
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, List, Optional

import numpy as np
import typer

from lst_tools.config import read_config
from lst_tools.extract import (
    build_quad_mesh_sampler,
    compute_freestream_attrs,
    extract_lower_wall,
    read_fequad_block_tecplot,
    sample_profiles,
    write_profiles_hdf5,
    write_profiles_tecplot,
    write_wall_profile_tecplot,
)
from lst_tools.extract._fequad import N_ETA


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# default station locations (used when no cfg or CLI stations are given)
# --------------------------------------------------
_DEFAULT_STATIONS = [0.0025, 0.005, 0.010, 0.015, 0.020, 0.025]


# --------------------------------------------------
# main function for the 'extract' cli command
# --------------------------------------------------
def cmd_extract(
    ctx: typer.Context,
    input_file: Annotated[
        Optional[Path],
        typer.Argument(help="Tecplot FE-quad BLOCK ASCII input file."),
    ] = None,
    cfg: Annotated[
        Optional[Path],
        typer.Option("--cfg", "-c", help="Explicit lst.cfg path."),
    ] = None,
    hdf5_out: Annotated[
        Optional[Path],
        typer.Option("--hdf5-out", help="Output HDF5 baseflow file path."),
    ] = None,
    profiles_out: Annotated[
        Optional[Path],
        typer.Option("--profiles-out", help="Output Tecplot profiles file path."),
    ] = None,
    wall_out: Annotated[
        Optional[Path],
        typer.Option("--wall-out", help="Output Tecplot wall curve file path."),
    ] = None,
    mach: Annotated[
        Optional[float],
        typer.Option("--mach", help="Freestream Mach number (overrides lst.cfg)."),
    ] = None,
    t_inf: Annotated[
        Optional[float],
        typer.Option("--t-inf", help="Freestream static temperature [K] (overrides lst.cfg)."),
    ] = None,
    station: Annotated[
        Optional[List[float]],
        typer.Option(
            "--station",
            help="Streamwise x-coordinate for a profile station (repeatable, overrides lst.cfg).",
        ),
    ] = None,
) -> None:
    """Extract wall-normal profiles from a Tecplot FE-quadrilateral slice.
\f

    Workflow:

    1. Load project config from ``lst.cfg`` (auto-discovered or ``--cfg``).
    2. Resolve input file, output paths, Mach, T_inf, and stations from
       CLI flags → ``[extract]``/``[flow_conditions]`` in cfg → built-in defaults.
    3. Parse the Tecplot BLOCK FE-quad file.
    4. Identify the lower wall boundary and build the quad mesh sampler.
    5. Sample wall-normal profiles along rays perpendicular to the wall.
    6. Write HDF5 (``lst_next_gen``-compatible schema) and two Tecplot diagnostics.

    Parameters
    ----------
    ctx : typer.Context
        Typer context carrying the ``debug`` flag from the parent callback.
    input_file : Path | None
        Positional input file argument. Overrides ``[extract] input_file`` in cfg.
    """

    # set debug flag based on parent context
    debug = ctx.obj.get("debug", False) if ctx.obj else False

    try:
        # load config (tolerates missing file; uses defaults)
        config = read_config(path=cfg)
        ext_cfg = config.extract
        fc_cfg = config.flow_conditions

        # resolve input file
        # priority: CLI positional > [extract] input_file in cfg
        resolved_input: Path | None = input_file
        if resolved_input is None and ext_cfg.input_file is not None:
            resolved_input = Path(ext_cfg.input_file)

        if resolved_input is None:
            typer.echo(
                "error: input file required — pass it as an argument or set "
                "[extract] input_file in lst.cfg",
                err=True,
            )
            raise typer.Exit(1)

        if not resolved_input.exists():
            typer.echo(f"error: input file not found: {resolved_input}", err=True)
            raise typer.Exit(1)

        # resolve output paths
        # priority: CLI flag > [extract] field in cfg > next to input file
        def _resolve_out(cli_val: Path | None, cfg_val: str | None, default_name: str) -> Path:
            if cli_val is not None:
                return cli_val
            if cfg_val is not None:
                return Path(cfg_val)
            return resolved_input.parent / default_name

        resolved_hdf5 = _resolve_out(hdf5_out, ext_cfg.hdf5_out, "baseflow.hdf5")
        resolved_profiles = _resolve_out(profiles_out, ext_cfg.profiles_out, "extracted_profiles.dat")
        resolved_wall = _resolve_out(wall_out, ext_cfg.wall_out, "wall_profile.dat")

        # resolve flow conditions
        # priority: CLI flag > [flow_conditions] in cfg > built-in default
        resolved_mach: float = mach if mach is not None else (fc_cfg.mach if fc_cfg.mach is not None else 6.0)
        resolved_t_inf: float = t_inf if t_inf is not None else (fc_cfg.temp_inf if fc_cfg.temp_inf is not None else 50.0)

        # resolve station x-coordinates
        # priority: CLI --station (repeatable) > [extract] stations in cfg > built-in defaults
        if station:
            resolved_stations = np.asarray(sorted(station), dtype=float)
        elif ext_cfg.stations is not None:
            resolved_stations = np.asarray(ext_cfg.stations, dtype=float)
        else:
            resolved_stations = np.asarray(_DEFAULT_STATIONS, dtype=float)

        # debug output for devs
        logger.debug("input file: %s", resolved_input)
        logger.debug("hdf5 out: %s", resolved_hdf5)
        logger.debug("mach=%.4f  T_inf=%.2f K", resolved_mach, resolved_t_inf)
        logger.debug("stations: %s", resolved_stations.tolist())

        # read the Tecplot FE-quad file
        typer.echo(f"reading {resolved_input}")
        dataset = read_fequad_block_tecplot(resolved_input)

        n_cells = dataset.connectivity.shape[0]
        typer.echo(f"dataset: {dataset.nodal['x'].size} nodes, {n_cells} cells")

        nodal_x = dataset.nodal["x"]
        nodal_y = dataset.nodal["y"]

        # extract the lower wall boundary
        wall_x, wall_y = extract_lower_wall(nodal_x, nodal_y, dataset.connectivity)

        # debug output 
        logger.debug(
            "lower wall: %d points, x in [%.4e, %.4e]",
            wall_x.size,
            float(wall_x[0]),
            float(wall_x[-1]),
        )

        # build the quad mesh sampler (nodal reconstruction + spatial index)
        typer.echo("building mesh sampler (this may take a moment on large meshes)")

        # debug output for devs
        n_cells = dataset.connectivity.shape[0]
        logger.debug("building mesh sampler with %d nodes and %d cells", nodal_x.size, n_cells)

        mesh_sampler = build_quad_mesh_sampler(
            nodal_x, nodal_y, dataset.connectivity, dataset.cell,
            existing_nodal_fields=dataset.nodal,
        )

        # write the extracted wall curve diagnostic
        write_wall_profile_tecplot(resolved_wall, wall_x, wall_y)
        logger.debug("wall profile written: %s", resolved_wall)

        # sample wall-normal profiles
        typer.echo(f"sampling {resolved_stations.size} profiles ({N_ETA} points each)")
        raw_profiles = sample_profiles(wall_x, wall_y, mesh_sampler, resolved_stations)

        # compute freestream attributes
        freestream_attrs = compute_freestream_attrs(raw_profiles, resolved_mach, resolved_t_inf)

        # write Tecplot profiles diagnostic
        write_profiles_tecplot(resolved_profiles, raw_profiles)
        logger.debug("profiles tecplot written: %s", resolved_profiles)

        # write HDF5 baseflow file 
        write_profiles_hdf5(resolved_hdf5, raw_profiles, freestream_attrs)

        # print summary for the user
        typer.echo(f"{resolved_input} -> {resolved_hdf5}")
        typer.echo(f"  stations: {resolved_stations.size}")
        typer.echo(f"  points per profile: {raw_profiles.eta.size}")
        typer.echo(f"  wall x: {float(wall_x[0]):.6f} -> {float(wall_x[-1]):.6f}")
        typer.echo(f"  eta max: {float(raw_profiles.eta[-1]):.6f}")
        typer.echo(f"  Mach: {resolved_mach:.4f}  T_inf: {resolved_t_inf:.2f} K")
        typer.echo(f"  diagnostics: {resolved_profiles}  {resolved_wall}")

        if debug:
            typer.echo(f"  rho_inf: {freestream_attrs['static density']:.4e} kg/m^3")
            typer.echo(f"  mu_inf:  {freestream_attrs['freestream viscosity']:.4e} Pa·s")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("extract failed", exc_info=True)
        raise typer.Exit(1)
