"""CLI handler for ``lst-tools info``.

Reads a LASTRAC ``meanflow.bin`` file and prints a summary of its
contents: file header, station count, coordinate range, grid
dimensions, and reference quantities.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from lst_tools.data_io.lastrac_binary import LastracReader


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function for the 'info' cli option
# --------------------------------------------------
def cmd_info(
    fpath: Annotated[
        Path,
        typer.Argument(help="Path to a meanflow.bin file."),
    ],
    profiles_out: Annotated[
        Path | None,
        typer.Option(
            "--profiles-out",
            "-o",
            help="Optional Tecplot ASCII output for all meanflow profiles.",
        ),
    ] = None,
) -> None:
    """Print summary information for a LASTRAC meanflow binary file."""

    # validate that the file exists
    if not fpath.is_file():
        typer.echo(f"error: {fpath} not found", err=True)
        raise typer.Exit(1)

    # default empty handles for robust cleanup in finally
    fio = None
    profile_stream = None

    try:
        # open the meanflow binary
        fio = LastracReader(fpath, endianness="<")

        # read file header
        header = fio.read_header()

        # extract station count
        n_station = int(header["n_station"])

        # print file header
        typer.echo(f"file:       {fpath}")
        typer.echo(f"title:      {header['title']}")
        typer.echo(f"n_station:  {n_station}")
        typer.echo(f"igas:       {header['igas']}")
        typer.echo(f"iunit:      {header['iunit']}")
        typer.echo(f"Pr:         {header['Pr']}")
        typer.echo(f"stat_pres:  {header['stat_pres']:.6e}")
        typer.echo(f"nsp:        {header['nsp']}")

        # prepare optional Tecplot profile export
        if profiles_out is not None:
            # convert to Path object and ensure output directory exists
            profiles_out.parent.mkdir(parents=True, exist_ok=True)
            profile_stream = profiles_out.open("w", encoding="utf-8")
            profile_stream.write('TITLE = "meanflow_profiles"\n')
            profile_stream.write('VARIABLES = "s" "eta" "u" "v" "w" "T" "p"\n')

        # read all station headers to collect summary data
        s_values = np.empty(n_station, dtype=float)
        n_eta_first = None
        re1 = None
        lref = None
        stat_temp = None
        stat_uvel = None
        stat_dens = None
        kappa_min = np.inf
        kappa_max = -np.inf

        for i in range(n_station):
            # read station header
            sh = fio.read_station_header()

            # store station coordinate
            s_values[i] = sh["s"]

            # capture values from first station
            if i == 0:
                n_eta_first = sh["n_eta"]
                re1 = sh["re1"]
                lref = sh["lref"]
                stat_temp = sh["stat_temp"]
                stat_uvel = sh["stat_uvel"]
                stat_dens = sh["stat_dens"]

            # track curvature range
            kappa_min = min(kappa_min, sh["kappa"])
            kappa_max = max(kappa_max, sh["kappa"])

            # read or skip station vectors (eta, u, v, w, temp, pres)
            if profile_stream is None:
                fio.skip_records(6)
            else:
                n_eta = int(sh["n_eta"])
                eta = fio.read_station_vector(count=n_eta)
                uvel = fio.read_station_vector(count=n_eta)
                vvel = fio.read_station_vector(count=n_eta)
                wvel = fio.read_station_vector(count=n_eta)
                temp = fio.read_station_vector(count=n_eta)
                pres = fio.read_station_vector(count=n_eta)

                zone_name = f"station_{i:04d}_iloc_{int(sh['i_loc']):04d}_s_{float(sh['s']):.6e}"
                profile_stream.write(
                    f'ZONE T="{zone_name}", I={n_eta}, DATAPACKING=POINT\n'
                )

                for j in range(n_eta):
                    profile_stream.write(
                        f"{float(sh['s']):.8e} "
                        f"{eta[j]:.8e} "
                        f"{uvel[j]:.8e} "
                        f"{vvel[j]:.8e} "
                        f"{wvel[j]:.8e} "
                        f"{temp[j]:.8e} "
                        f"{pres[j]:.8e}\n"
                    )

        # print station summary
        typer.echo("")
        typer.echo("station summary")
        typer.echo(f"n_eta:      {n_eta_first}")
        typer.echo(f"s_min:      {s_values.min():.6e}")
        typer.echo(f"s_max:      {s_values.max():.6e}")
        typer.echo(f"s_first:    {s_values[0]:.6e}")
        typer.echo(f"s_last:     {s_values[-1]:.6e}")
        typer.echo(f"kappa:      [{kappa_min:.6e}, {kappa_max:.6e}]")

        # print reference quantities
        typer.echo("")
        typer.echo("reference quantities")
        typer.echo(f"lref:       {lref:.6e}")
        typer.echo(f"re1:        {re1:.6e}")
        typer.echo(f"stat_temp:  {stat_temp:.6e}")
        typer.echo(f"stat_uvel:  {stat_uvel:.6e}")
        typer.echo(f"stat_dens:  {stat_dens:.6e}")

        # print export location when profile file was requested
        if profiles_out is not None:
            typer.echo("")
            typer.echo(f"profiles_out: {profiles_out}")

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("info command failed", exc_info=True)
        raise typer.Exit(1)
    finally:
        try:
            if profile_stream is not None:
                profile_stream.close()
        except Exception:
            pass
        try:
            if fio is not None:
                fio.close()
        except Exception:
            pass
