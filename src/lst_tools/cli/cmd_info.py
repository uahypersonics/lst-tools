"""CLI handler for ``lst-tools info``.

Reads a LASTRAC ``meanflow.bin`` file and prints a summary of its
contents: file header, station count, coordinate range, grid
dimensions, and reference quantities.

--profiles-out option can be used to export all station profiles to a
Tecplot ASCII file for visualization and data inspection
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

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

    # validate that the input file exists before opening the binary reader
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

        # read and report each station independently because dimensions and
        # reference quantities can vary through the meanflow file
        for station_index in range(n_station):
            # read station header
            sh = fio.read_station_header()
            n_eta = int(sh["n_eta"])

            # read eta for station dimensions and bounds
            eta = fio.read_station_vector(count=n_eta)

            # read or skip the remaining station vectors (u, v, w, temp, pres)
            if profile_stream is None:
                fio.skip_records(5)
            else:
                uvel = fio.read_station_vector(count=n_eta)
                vvel = fio.read_station_vector(count=n_eta)
                wvel = fio.read_station_vector(count=n_eta)
                temp = fio.read_station_vector(count=n_eta)
                pres = fio.read_station_vector(count=n_eta)

                zone_name = (
                    f"station_{station_index:04d}_"
                    f"iloc_{int(sh['i_loc']):04d}_"
                    f"s_{float(sh['s']):.6e}"
                )
                profile_stream.write(
                    f'ZONE T="{zone_name}", I={n_eta}, DATAPACKING=POINT\n'
                )

                for eta_index in range(n_eta):
                    profile_stream.write(
                        f"{float(sh['s']):.8e} "
                        f"{eta[eta_index]:.8e} "
                        f"{uvel[eta_index]:.8e} "
                        f"{vvel[eta_index]:.8e} "
                        f"{wvel[eta_index]:.8e} "
                        f"{temp[eta_index]:.8e} "
                        f"{pres[eta_index]:.8e}\n"
                    )

            # print station-specific dimensions, geometry, and references
            typer.echo("")
            typer.echo(f"station {station_index + 1}")
            typer.echo(f"  i_loc:      {sh['i_loc']}")
            typer.echo(f"  s:          {sh['s']:.6e}")
            typer.echo(f"  eta_0:      {eta[0]:.6e}")
            typer.echo(f"  eta_max:    {eta[-1]:.6e}")
            typer.echo(f"  n_eta:      {n_eta}")
            typer.echo(f"  kappa:      {sh['kappa']:.6e}")
            typer.echo(f"  rloc:       {sh['rloc']:.6e}")
            typer.echo(f"  drdx:       {sh['drdx']:.6e}")
            typer.echo(f"  lref:       {sh['lref']:.6e}")
            typer.echo(f"  re1:        {sh['re1']:.6e}")
            typer.echo(f"  stat_temp:  {sh['stat_temp']:.6e}")
            typer.echo(f"  stat_uvel:  {sh['stat_uvel']:.6e}")
            typer.echo(f"  stat_dens:  {sh['stat_dens']:.6e}")

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
