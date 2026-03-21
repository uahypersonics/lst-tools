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
) -> None:
    """Print summary information for a LASTRAC meanflow binary file."""

    # validate that the file exists
    if not fpath.is_file():
        typer.echo(f"error: {fpath} not found", err=True)
        raise typer.Exit(1)

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

            # skip station vectors (eta, u, v, w, temp, pres)
            fio.skip_records(6)

        # close the file
        fio.close()

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

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"error: {e}", err=True)
        logger.debug("info command failed", exc_info=True)
        raise typer.Exit(1)
