"""CLI handler for ``lst-tools extract``.

Reads a Tecplot FE-quadrilateral BLOCK ASCII symmetry slice, extracts
wall-normal profiles at user-specified streamwise stations via barycentric
interpolation, and writes an HDF5 baseflow file for use with ``lst-tools
lastrac``.

Output paths are resolved from ``[extract]`` in ``lst.cfg``,
or default to ``extracted_baseflow.hdf5`` next to the input file.

Freestream metadata (Mach, T_inf) is written to the HDF5 only when
``[flow_conditions]`` provides both ``mach`` and ``temp_inf``.

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
from lst_tools.extract._fequad import DEFAULT_ETA_DISTRIBUTION, N_ETA


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# default station locations (used when no cfg or CLI stations are given)
# --------------------------------------------------
_DEFAULT_STATIONS = [0.0025, 0.005, 0.010, 0.015, 0.020, 0.025]


# --------------------------------------------------
# surface-side resolution
# --------------------------------------------------
def _resolve_surface(cli_surface: str | None, cfg_surface: str | None) -> str:
    """Resolve the requested surface side via a three-way fallback.

    Priority: explicit CLI ``--surface`` flag > ``[extract] surface`` in cfg >
    built-in ``"lower"`` default.  Using ``None`` as the CLI sentinel lets an
    explicit ``--surface lower`` win over a config ``surface = "upper"`` instead
    of being silently overridden.

    Args:
        cli_surface: Value from the ``--surface`` flag, or ``None`` if unset.
        cfg_surface: Value from ``[extract] surface`` in cfg, or ``None`` if unset.

    Returns:
        The resolved surface string.
    """

    if cli_surface is not None:
        return cli_surface
    if cfg_surface is not None:
        return cfg_surface
    return "lower"


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
    station: Annotated[
        Optional[List[float]],
        typer.Option(
            "--station",
            help="Streamwise x-coordinate for a profile station (repeatable, overrides lst.cfg).",
        ),
    ] = None,
    surface: Annotated[
        Optional[str],
        typer.Option(
            "--surface",
            help="Surface side to extract: lower or upper.",
        ),
    ] = None,
) -> None:
    """Extract wall-normal profiles from a Tecplot FE-quadrilateral slice.
\f

    Workflow:

    1. Load project config from ``lst.cfg`` (auto-discovered or ``--cfg``).
    2. Resolve input file, output paths, and stations from
       CLI flags → ``[extract]`` in cfg → built-in defaults.
       Freestream metadata written only when mach and temp_inf are in ``[flow_conditions]``.
    3. Parse the Tecplot BLOCK FE-quad file.
    4. Identify the lower wall boundary and build the quad mesh sampler.
    5. Sample wall-normal profiles along rays perpendicular to the wall.
    6. Write HDF5 baseflow file and two Tecplot diagnostics.

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

        # resolve output paths from [extract] config or defaults next to input file
        def _resolve_hdf5_out(cfg_val: str | None, default_name: str) -> Path:
            if cfg_val is not None and cfg_val.strip() != "":
                return Path(cfg_val)
            return resolved_input.parent / default_name

        resolved_hdf5 = _resolve_hdf5_out(ext_cfg.hdf5_out, "extracted_baseflow.hdf5")

        # optional outputs — default to extracted_profiles.dat; wall_out only written when
        # explicitly set in [extract] config
        resolved_profiles: Path = (
            Path(ext_cfg.profiles_out)
            if ext_cfg.profiles_out and ext_cfg.profiles_out.strip()
            else resolved_input.parent / "extracted_profiles.dat"
        )
        resolved_wall: Path | None = (
            Path(ext_cfg.wall_out)
            if ext_cfg.wall_out and ext_cfg.wall_out.strip()
            else None
        )

        # resolve freestream conditions from [flow_conditions] config only
        # skip freestream metadata if either value is missing
        resolved_rgas: float = fc_cfg.rgas
        have_freestream = fc_cfg.mach is not None and fc_cfg.temp_inf is not None
        if not have_freestream:
            typer.echo(
                "warning: mach and temp_inf not set in [flow_conditions] — "
                "HDF5 freestream metadata will not be written",
                err=True,
            )

        # resolve station x-coordinates
        # priority: CLI --station (repeatable) > [extract] stations in cfg > built-in defaults
        if station:
            resolved_stations = np.asarray(sorted(station), dtype=float)
        elif ext_cfg.stations is not None:
            resolved_stations = np.asarray(ext_cfg.stations, dtype=float)
        else:
            resolved_stations = np.asarray(_DEFAULT_STATIONS, dtype=float)

        # resolve wall-normal point count from [extract] config or built-in default
        resolved_n_eta = ext_cfg.n_eta if ext_cfg.n_eta is not None else N_ETA

        # resolve wall-normal point distribution from [extract] config or built-in default
        resolved_eta_distribution = (
            ext_cfg.eta_distribution if ext_cfg.eta_distribution is not None
            else DEFAULT_ETA_DISTRIBUTION
        )
        resolved_eta_distribution = resolved_eta_distribution.strip().lower()

        # resolve requested surface side
        # priority: explicit CLI flag > [extract] surface in cfg > built-in default
        resolved_surface = _resolve_surface(surface, ext_cfg.surface)

        surface_key = resolved_surface.strip().lower()
        if surface_key not in {"lower", "upper"}:
            typer.echo("error: --surface must be 'lower' or 'upper'", err=True)
            raise typer.Exit(1)
        target_y = 1.0 if surface_key == "upper" else -1.0

        # debug output for devs
        logger.debug("input file: %s", resolved_input)
        logger.debug("hdf5 out: %s", resolved_hdf5)
        logger.debug("freestream available: %s", have_freestream)
        logger.debug("stations: %s", resolved_stations.tolist())
        logger.debug("n_eta: %d", resolved_n_eta)
        logger.debug("eta_distribution: %s", resolved_eta_distribution)
        logger.debug("requested surface: %s", surface_key)

        # read the Tecplot FE-quad file
        typer.echo(f"reading {resolved_input}")
        dataset = read_fequad_block_tecplot(resolved_input)

        n_cells = dataset.connectivity.shape[0]
        typer.echo(f"dataset: {dataset.nodal['x'].size} nodes, {n_cells} cells")

        nodal_x = dataset.nodal["x"]
        nodal_y = dataset.nodal["y"]

        # build the quad mesh sampler (nodal reconstruction + spatial index)
        typer.echo("building mesh sampler (this may take a moment on large meshes)")

        # debug output for devs
        n_cells = dataset.connectivity.shape[0]
        logger.debug("building mesh sampler with %d nodes and %d cells", nodal_x.size, n_cells)

        mesh_sampler = build_quad_mesh_sampler(
            nodal_x, nodal_y, dataset.connectivity, dataset.cell,
            existing_nodal_fields=dataset.nodal,
        )

        # extract the lower/body wall boundary
        wall_x, wall_y = extract_lower_wall(
            nodal_x,
            nodal_y,
            dataset.connectivity,
            nodal_fields=mesh_sampler.nodal_fields,
        )

        # debug output
        logger.debug(
            "lower wall: %d points, x in [%.4e, %.4e]",
            wall_x.size,
            float(wall_x[0]),
            float(wall_x[-1]),
        )

        # write the extracted wall curve diagnostic (only if configured)
        if resolved_wall is not None:
            write_wall_profile_tecplot(resolved_wall, wall_x, wall_y)
            logger.debug("wall profile written: %s", resolved_wall)

        # sample wall-normal profiles
        typer.echo(f"sampling {resolved_stations.size} profiles ({resolved_n_eta} points each)")
        raw_profiles = sample_profiles(
            wall_x,
            wall_y,
            mesh_sampler,
            resolved_stations,
            n_eta=resolved_n_eta,
            eta_distribution=resolved_eta_distribution,
            target_y=target_y,
            rgas=resolved_rgas,
        )

        # compute freestream attributes from config if both mach and temp_inf are set
        freestream_attrs: dict = {}
        if have_freestream:
            freestream_attrs = compute_freestream_attrs(
                raw_profiles, fc_cfg.mach, fc_cfg.temp_inf, rgas=resolved_rgas
            )

        # write Tecplot profiles file (always — defaults to extracted_profiles.dat)
        write_profiles_tecplot(resolved_profiles, raw_profiles)
        logger.debug("profiles tecplot written: %s", resolved_profiles)

        # write HDF5 baseflow file
        write_profiles_hdf5(resolved_hdf5, raw_profiles, freestream_attrs)

        # determine the surface that was actually extracted (may differ from the
        # requested surface when pick_wall_branch auto-falls back on a one-sided mesh)
        actual_surface = "upper" if float(np.mean(raw_profiles.station_y)) > 0.0 else "lower"

        # print summary for the user
        typer.echo(f"{resolved_input} -> {resolved_hdf5}")
        typer.echo(f"  profiles: {resolved_profiles}")
        typer.echo(f"  stations: {resolved_stations.size}")
        typer.echo(f"  points per profile: {raw_profiles.eta.size}")
        typer.echo(f"  surface: {actual_surface}")

        if debug:
            typer.echo(f"  rho_inf: {freestream_attrs['static density']:.4e} kg/m^3")
            typer.echo(f"  mu_inf:  {freestream_attrs['freestream viscosity']:.4e} Pa·s")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("extract failed", exc_info=True)
        raise typer.Exit(1)
