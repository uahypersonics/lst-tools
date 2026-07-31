"""CLI handler for ``lst-tools extract``.

Reads a Tecplot FE-quadrilateral BLOCK ASCII symmetry slice, extracts
wall-normal profiles at user-specified streamwise stations via barycentric
interpolation, and writes an HDF5 baseflow file for use with ``lst-tools
lastrac``.

Output paths are resolved from ``[extract]`` in ``lst.cfg``,
or default to ``extracted_baseflow.hdf5`` next to the input file.

Freestream metadata is written directly from ``[flow_conditions]`` so the
HDF5 carries the case values without a separate reconstruction step.

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
    extract_lower_wall,
    read_fequad_block_tecplot,
    sample_profiles,
    write_profiles_hdf5,
    write_profiles_tecplot,
    write_wall_profile_tecplot,
)
from lst_tools.extract._normalize import detect_dimensional, normalize_profiles
from lst_tools.extract._profile import default_eta_distribution, default_n_eta

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# main function for the 'extract' cli command
# --------------------------------------------------
def cmd_extract(
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

    Parameters
    ----------
    input_file : Path | None
        Positional input file argument. Overrides ``[extract] input_file`` in cfg.
    """

    # check if verbose is enabled
    verbose = logging.getLogger("lst_tools").isEnabledFor(logging.DEBUG)

    try:
        # load config
        config = read_config(path=cfg)

        # get the [extract] section of the config file
        ext_cfg = config.extract
        # get the [flow_conditions] section of the config file
        fc_cfg = config.flow_conditions

        # get input file for extraction
        fpath_inp: Path | None = input_file
        if fpath_inp is None and ext_cfg.input_file is not None:
            fpath_inp = Path(ext_cfg.input_file)

        # if no input file is found extraction cannot be carried out -> print error and exit
        if fpath_inp is None:
            typer.echo(
                "error: input file required in config file"
                "[extract] input_file in lst.cfg",
                err=True,
            )
            raise typer.Exit(1)

        # if input file is provided but does not exist -> print error and exit
        if not fpath_inp.exists():
            typer.echo(f"error: input file not found: {fpath_inp}", err=True)
            raise typer.Exit(1)

        # resolve HDF5 output path from config or default next to the input file
        if ext_cfg.hdf5_out is not None and ext_cfg.hdf5_out.strip():
            fpath_out = Path(ext_cfg.hdf5_out)
        else:
            fpath_out = fpath_inp.parent / "extracted_baseflow.hdf5"

        # optional outputs — default to extracted_profiles.dat; wall_out only written when
        # explicitly set in [extract] config
        resolved_profiles: Path = (
            Path(ext_cfg.profiles_out)
            if ext_cfg.profiles_out and ext_cfg.profiles_out.strip()
            else fpath_inp.parent / "extracted_profiles.dat"
        )
        fpath_wall: Path | None = (
            Path(ext_cfg.wall_out)
            if ext_cfg.wall_out and ext_cfg.wall_out.strip()
            else None
        )

        # get freestream conditions from [flow_conditions] config only
        rgas: float = fc_cfg.rgas

        # get station x-coordinates
        if station:
            stations = np.asarray(sorted(station), dtype=float)
        elif (
            ext_cfg.x_s is not None
            and ext_cfg.x_e is not None
            and ext_cfg.d_x is not None
        ):
            # generate stations from range specification
            stations = np.arange(ext_cfg.x_s, ext_cfg.x_e + ext_cfg.d_x / 2.0, ext_cfg.d_x)
            logger.debug(
                "stations from x_s/x_e/d_x: %.4g to %.4g step %.4g (%d stations)",
                ext_cfg.x_s, ext_cfg.x_e, ext_cfg.d_x, len(stations),
            )
        elif ext_cfg.stations is not None:
            stations = np.asarray(ext_cfg.stations, dtype=float)
        else:
            typer.echo(
                "error: extraction stations required; pass --station or set "
                "[extract] stations or x_s/x_e/d_x in lst.cfg",
                err=True,
            )
            raise typer.Exit(1)

        # reject empty station lists before entering the extraction pipeline
        if stations.size == 0:
            typer.echo("error: extraction station list cannot be empty", err=True)
            raise typer.Exit(1)

        # resolve wall-normal point count from [extract] config or built-in default
        n_eta = ext_cfg.n_eta if ext_cfg.n_eta is not None else default_n_eta

        # resolve wall-normal point distribution from [extract] config or built-in default
        eta_distribution = (
            ext_cfg.eta_distribution if ext_cfg.eta_distribution is not None
            else default_eta_distribution
        )
        eta_distribution = eta_distribution.strip().lower()

        # resolve distribution-specific controls from [extract] config
        eta_max = ext_cfg.eta_max
        eta_stretch = ext_cfg.eta_stretch
        eta_wall_spacing = ext_cfg.eta_wall_spacing

        # resolve requested surface side with the explicit CLI flag taking priority
        surface_name = surface if surface is not None else ext_cfg.surface
        if surface_name is None:
            surface_key = "auto"
            target_y = None
        else:
            surface_key = surface_name.strip().lower()
            if surface_key not in {"lower", "upper"}:
                typer.echo("error: --surface must be 'lower' or 'upper'", err=True)
                raise typer.Exit(1)
            target_y = 1.0 if surface_key == "upper" else -1.0

        # debug output for devs
        logger.debug("input file: %s", fpath_inp)
        logger.debug("hdf5 out: %s", fpath_out)
        logger.debug("stations: %s", stations.tolist())
        logger.debug("n_eta: %d", n_eta)
        logger.debug("eta_max: %s", eta_max)
        logger.debug("eta_distribution: %s", eta_distribution)
        logger.debug("eta_stretch: %s", eta_stretch)
        logger.debug("eta_wall_spacing: %s", eta_wall_spacing)
        logger.debug("requested surface: %s", surface_key)

        # read the Tecplot FE-quad file
        typer.echo(f"reading {fpath_inp}")
        dataset = read_fequad_block_tecplot(fpath_inp)

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
        if fpath_wall is not None:
            write_wall_profile_tecplot(fpath_wall, wall_x, wall_y)
            logger.debug("wall profile written: %s", fpath_wall)

        # sample wall-normal profiles
        typer.echo(f"sampling {stations.size} profiles ({n_eta} points each)")
        raw_profiles = sample_profiles(
            wall_x,
            wall_y,
            mesh_sampler,
            stations,
            n_eta=n_eta,
            eta_max=eta_max,
            eta_distribution=eta_distribution,
            eta_stretch=eta_stretch,
            eta_wall_spacing=eta_wall_spacing,
            target_y=target_y,
            rgas=rgas,
        )

        # write freestream metadata directly from config values
        freestream_attrs: dict[str, float] = {}
        fc_attr_map = {
            "mach number": fc_cfg.mach,
            "heat capacity ratio": fc_cfg.gamma,
            "prandtl number": fc_cfg.pr,
            "gas constant": fc_cfg.rgas,
            "reference length scale": config.geometry.l_ref,
            "stagnation pressure": fc_cfg.pres_0,
            "stagnation temperature": fc_cfg.temp_0,
            "freestream pressure": fc_cfg.pres_inf,
            "freestream temperature": fc_cfg.temp_inf,
            "freestream density": fc_cfg.dens_inf,
            "freestream velocity": fc_cfg.uvel_inf,
            "viscosity law": fc_cfg.visc_law,
        }
        for attr_name, attr_value in fc_attr_map.items():
            if attr_value is not None:
                freestream_attrs[attr_name] = float(attr_value)

        # detect dimensional profiles and optionally normalize
        is_dimensional = detect_dimensional(raw_profiles)
        if is_dimensional and not ext_cfg.nondimensionalize:
            typer.echo("", err=True)
            typer.echo(
                typer.style(
                    "WARNING: profiles appear dimensional (mean edge velocity > 5 m/s)"
                    " but [extract] nondimensionalize = false."
                    " Set nondimensionalize = true to divide by edge values before writing HDF5.",
                    fg=typer.colors.YELLOW, bold=True,
                ),
                err=True,
            )
            typer.echo("", err=True)
        profiles_to_write = raw_profiles
        nondim_profiles_path: Path | None = None
        if is_dimensional and ext_cfg.nondimensionalize:
            profiles_to_write, _ = normalize_profiles(raw_profiles)
            # path for the non-dimensional Tecplot output
            nondim_profiles_path = resolved_profiles.with_stem(
                resolved_profiles.stem + "_nondimensional"
            )

        # write Tecplot profiles file — dimensional, always written for inspection
        write_profiles_tecplot(resolved_profiles, raw_profiles)
        logger.debug("profiles tecplot written: %s", resolved_profiles)

        # write non-dimensional Tecplot file when normalization was applied
        if nondim_profiles_path is not None:
            write_profiles_tecplot(nondim_profiles_path, profiles_to_write)
            logger.debug("non-dimensional profiles tecplot written: %s", nondim_profiles_path)

        # write HDF5 baseflow file
        write_profiles_hdf5(fpath_out, profiles_to_write, freestream_attrs)

        # determine the surface that was actually extracted (may differ from the
        # requested surface when pick_wall_branch auto-falls back on a one-sided mesh)
        actual_surface = "upper" if float(np.mean(raw_profiles.station_y)) > 0.0 else "lower"

        # print summary for the user
        typer.echo(f"{fpath_inp} -> {fpath_out}")
        typer.echo(f"  profiles (dimensional): {resolved_profiles}")
        if nondim_profiles_path is not None:
            typer.echo(f"  profiles (non-dim):     {nondim_profiles_path}")
        typer.echo(f"  stations: {stations.size}")
        typer.echo(f"  points per profile: {raw_profiles.eta.size}")
        typer.echo(f"  surface: {actual_surface}")

        if verbose and freestream_attrs:
            if "freestream density" in freestream_attrs:
                typer.echo(f"  rho_inf: {freestream_attrs['freestream density']:.4e} kg/m^3")
            if "freestream velocity" in freestream_attrs:
                typer.echo(f"  u_inf:   {freestream_attrs['freestream velocity']:.4e} m/s")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("extract failed", exc_info=True)
        raise typer.Exit(1)
