"""Combine cross-beta ridge curves into per-mode Tecplot surfaces."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lst_tools.process.modes import process_modes


# --------------------------------------------------
# main function for the process modes option
# --------------------------------------------------
def cmd_modes(
    directory: Annotated[
        Path,
        typer.Option(
            "--dir", help="Directory with nfac_max_mode_* and alpi_max_mode_* folders."
        ),
    ] = Path("."),
    output: Annotated[
        Path,
        typer.Option(
            "--out", "-o", help="Output directory, relative to --dir unless absolute."
        ),
    ] = Path("mode_surfaces"),
    nx: Annotated[
        int,
        typer.Option(
            "--nx", min=2, help="Points on the shared uniform s grid (default: 500)."
        ),
    ] = 500,
    force: Annotated[
        bool,
        typer.Option(
            "--force", "-f", help="Replace the output directory if it exists."
        ),
    ] = False,
) -> None:
    """Match ridge modes between beta cases and write their surfaces."""

    # check if the input directory exists and is a directory
    if not directory.is_dir():
        typer.echo(f"error: input directory not found: {directory}", err=True)
        raise typer.Exit(1)

    # assigin output directory path
    output_dir = output if output.is_absolute() else directory / output
    try:
        written = process_modes(directory, output_dir, nx=nx, force=force)
    except (OSError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error

    # report the number of surfaces and envelopes written
    surface_count = sum("_max_mode_" in path.name for path in written)
    envelope_count = sum("_envelope_mode_" in path.name for path in written)
    typer.echo(
        f"wrote {surface_count} maximum surface(s) and "
        f"{envelope_count} envelope(s) in {output_dir}"
    )
    typer.echo(f"matching report: {output_dir / 'mode_assignments.csv'}")
