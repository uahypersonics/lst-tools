"""lst-tools visualize — stage-aware wrappers for cfd-viz LST plots."""


# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Annotated

import typer


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# parse variable candidates helper
# --------------------------------------------------
def _split_candidates(raw: str) -> list[str]:
    """Split comma-separated aliases into ordered candidate names."""
    return [name.strip() for name in raw.split(",") if name.strip()]


# --------------------------------------------------
# resolve actual field name helper
# --------------------------------------------------
def _resolve_field_name(flow: dict[str, object], raw_candidates: str) -> str:
    """Resolve first matching field name from candidate aliases."""

    # check each candidate in order
    candidates = _split_candidates(raw_candidates)
    for name in candidates:
        if name in flow:
            return name

    # include available variables in error for easier debugging
    available = list(flow.keys())
    raise KeyError(
        f"none of the requested fields were found: {candidates}; available={available}"
    )


# --------------------------------------------------
# discover completed kc_* slice files
# --------------------------------------------------
def _discover_tracking_files(search_dir: Path) -> list[Path]:
    """Return sorted tracking slice files under kc_* directories."""

    # set directory pattern to discover
    dir_pattern = "kc_*"

    # discover all directories matching pattern
    dir_list = sorted(search_dir.glob(dir_pattern))

    # set expected file name
    fname = "growth_rate_with_nfact_amps.dat"

    # create empty list of Path objects to hold discovered files
    flist: list[Path] = []

    # iterate over discovered directories and collect existing solution files
    for dir in dir_list:

        # first check if it is a directory before looking for files inside, to avoid false matches
        if not dir.is_dir():
            continue

        # look for expected file inside this directory and add to list if found
        fpath = dir / fname
        if fpath.exists():
            flist.append(fpath)

    # return list of path objects
    return flist


# --------------------------------------------------
# compute shared contour bounds across multiple files
# --------------------------------------------------
def _compute_shared_bounds(
    *,
    input_files: list[Path],
    field: str,
    levels_policy: str,
) -> tuple[float, float]:
    """Compute global min/max bounds for consistent multi-file rendering."""

    # lazy import reader here so non-visualize workflows avoid this dependency
    from cfd_io import read_file

    # initialize global extrema
    global_min: float | None = None
    global_max: float | None = None

    # read each file and update extrema for resolved field values
    for fpath in input_files:
        ds = read_file(str(fpath))
        field_name = _resolve_field_name(ds.flow, field)
        values = ds.flow[field_name].data

        local_min = float(values.min())
        local_max = float(values.max())

        global_min = local_min if global_min is None else min(global_min, local_min)
        global_max = local_max if global_max is None else max(global_max, local_max)

    if global_min is None or global_max is None:
        raise ValueError("could not compute contour bounds from tracking slices")

    # apply selected contour policy to global extrema
    if levels_policy == "positive-rounded":
        level_min = 0.0
        level_max = float(math.ceil(global_max / 10.0) * 10.0)
    elif levels_policy == "global-auto":
        level_min = global_min
        level_max = global_max
    else:
        raise ValueError(
            f"unknown levels policy '{levels_policy}'. Use one of: global-auto, positive-rounded"
        )

    # avoid degenerate contour bounds
    if level_max <= level_min:
        level_max = level_min + 1.0

    return level_min, level_max


# --------------------------------------------------
# shared wrapper implementation
# --------------------------------------------------
def _visualize_data(
    *,
    stage: str,
    input_path: Path,
    out_dir: Path,
    prefix: str,
    field: str,
    xvar: str,
    yvar: str,
    kvar: str,
    all_k: bool,
    k_index: int,
    levels_policy: str,
    levels_count: int,
    level_min_override: float | None = None,
    level_max_override: float | None = None,
    clip_below: bool,
    dpi: int,
    emit_summary: bool = True,
) -> list[Path]:
    """Dispatch stage visualization to cfd-viz renderer."""

    # validate input file exists before rendering
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    # debug output for devs
    logger.debug(
        "visualize %s: input=%s, out_dir=%s, prefix=%s",
        stage,
        input_path,
        out_dir,
        prefix,
    )

    # lazily import cfd-viz renderer only when visualize commands are used
    try:
        lst_module = importlib.import_module("cfd_viz.lst")
        render_lst_contours = lst_module.render_lst_contours
    except Exception as exc:  # pragma: no cover - tested via CLI behavior
        raise RuntimeError(
            "cfd-viz is required for visualization wrappers. "
            "Install it in this environment (for example: pip install cfd-viz)."
        ) from exc

    # call cfd-viz plotting engine
    files = render_lst_contours(
        path=input_path,
        field=field,
        xvar=xvar,
        yvar=yvar,
        kvar=kvar,
        all_k=all_k,
        k_index=k_index,
        out_dir=out_dir,
        prefix=prefix,
        levels_policy=levels_policy,
        levels_count=levels_count,
        level_min_override=level_min_override,
        level_max_override=level_max_override,
        clip_below=clip_below,
        dpi=dpi,
        show=False,
    )

    # print user summary
    if emit_summary:
        typer.echo(f"visualization complete ({stage})")
        typer.echo(f"wrote {len(files)} plot(s)")
        if files:
            typer.echo(f"first: {files[0]}")
            typer.echo(f"last:  {files[-1]}")

    return files


# --------------------------------------------------
# parsing wrapper
# --------------------------------------------------
def cmd_visualize_parsing(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Parsing Tecplot input file (default: growth_rate_with_nfact_amps.dat).",
        ),
    ] = Path("growth_rate_with_nfact_amps.dat"),
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for rendered parsing plots.",
        ),
    ] = Path("alpi_contours_parsing"),
    prefix: Annotated[
        str,
        typer.Option("--prefix", "-p", help="Output filename prefix."),
    ] = "alpi_kc",
    field: Annotated[
        str,
        typer.Option("--field", "-f", help="Contour field name."),
    ] = "-im(alpha)",
    xvar: Annotated[
        str,
        typer.Option("--xvar", help="X-axis variable or aliases."),
    ] = "s",
    yvar: Annotated[
        str,
        typer.Option("--yvar", help="Y-axis variable or aliases."),
    ] = "freq,freq.",
    kvar: Annotated[
        str,
        typer.Option("--kvar", help="K-sweep variable or aliases."),
    ] = "beta",
    all_k: Annotated[
        bool,
        typer.Option("--all-k/--single-k", help="Render all k-planes or one selected plane."),
    ] = True,
    k_index: Annotated[
        int,
        typer.Option("--k-index", "-k", min=1, help="1-based k index when --single-k is used."),
    ] = 1,
    levels_policy: Annotated[
        str,
        typer.Option("--levels-policy", help="Contour levels policy."),
    ] = "positive-rounded",
    levels_count: Annotated[
        int,
        typer.Option("--levels-count", min=2, help="Number of contour levels."),
    ] = 60,
    clip_below: Annotated[
        bool,
        typer.Option("--clip-below/--no-clip-below", help="Clip values below minimum contour level."),
    ] = True,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=72, help="PNG output DPI."),
    ] = 300,
) -> None:
    """Visualize parsing results using cfd-viz defaults."""
    try:
        _visualize_data(
            stage="parsing",
            input_path=input_path,
            out_dir=out_dir,
            prefix=prefix,
            field=field,
            xvar=xvar,
            yvar=yvar,
            kvar=kvar,
            all_k=all_k,
            k_index=k_index,
            levels_policy=levels_policy,
            levels_count=levels_count,
            clip_below=clip_below,
            dpi=dpi,
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)


# --------------------------------------------------
# tracking wrapper
# --------------------------------------------------
def cmd_visualize_tracking(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            "-i",
            help="Tracking Tecplot volume input file (default: lst_vol.dat).",
        ),
    ] = Path("lst_vol.dat"),
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for rendered tracking plots.",
        ),
    ] = Path("alpi_contours_tracking"),
    prefix: Annotated[
        str,
        typer.Option("--prefix", "-p", help="Output filename prefix."),
    ] = "alpi_kc",
    field: Annotated[
        str,
        typer.Option("--field", "-f", help="Contour field name."),
    ] = "-im(alpha)",
    xvar: Annotated[
        str,
        typer.Option("--xvar", help="X-axis variable or aliases."),
    ] = "s",
    yvar: Annotated[
        str,
        typer.Option("--yvar", help="Y-axis variable or aliases."),
    ] = "freq,freq.",
    kvar: Annotated[
        str,
        typer.Option("--kvar", help="K-sweep variable or aliases."),
    ] = "beta",
    all_k: Annotated[
        bool,
        typer.Option("--all-k/--single-k", help="Render all k-planes or one selected plane."),
    ] = True,
    k_index: Annotated[
        int,
        typer.Option("--k-index", "-k", min=1, help="1-based k index when --single-k is used."),
    ] = 1,
    levels_policy: Annotated[
        str,
        typer.Option("--levels-policy", help="Contour levels policy."),
    ] = "positive-rounded",
    levels_count: Annotated[
        int,
        typer.Option("--levels-count", min=2, help="Number of contour levels."),
    ] = 60,
    clip_below: Annotated[
        bool,
        typer.Option("--clip-below/--no-clip-below", help="Clip values below minimum contour level."),
    ] = True,
    dpi: Annotated[
        int,
        typer.Option("--dpi", min=72, help="PNG output DPI."),
    ] = 300,
) -> None:
    """Visualize tracking results using cfd-viz defaults."""
    try:
        # set default input file used for tracking fallback behavior
        default_tracking_volume = Path("lst_vol.dat")

        # primary path: use consolidated tracking volume when present
        if input_path.exists():
            _visualize_data(
                stage="tracking",
                input_path=input_path,
                out_dir=out_dir,
                prefix=prefix,
                field=field,
                xvar=xvar,
                yvar=yvar,
                kvar=kvar,
                all_k=all_k,
                k_index=k_index,
                levels_policy=levels_policy,
                levels_count=levels_count,
                clip_below=clip_below,
                dpi=dpi,
            )
            return

        # fallback path: discover individual kc_* tracking slices
        if input_path != default_tracking_volume:
            raise FileNotFoundError(f"input file not found: {input_path}")

        root = Path(".").resolve()
        slice_files = _discover_tracking_files(root)
        if not slice_files:
            raise FileNotFoundError(
                f"{default_tracking_volume} not found and no kc_* tracking slices discovered"
            )

        # compute one shared contour scale across all discovered slices
        level_min, level_max = _compute_shared_bounds(
            input_files=slice_files,
            field=field,
            levels_policy=levels_policy,
        )

        # render one contour per kc directory into a common output folder
        all_outputs: list[Path] = []
        for fpath in slice_files:
            case_prefix = f"{prefix}_{fpath.parent.name}"
            outputs = _visualize_data(
                stage="tracking",
                input_path=fpath,
                out_dir=out_dir,
                prefix=case_prefix,
                field=field,
                xvar=xvar,
                yvar=yvar,
                kvar=kvar,
                all_k=False,
                k_index=1,
                levels_policy=levels_policy,
                levels_count=levels_count,
                level_min_override=level_min,
                level_max_override=level_max,
                clip_below=clip_below,
                dpi=dpi,
                emit_summary=False,
            )
            all_outputs.extend(outputs)

        # print consolidated fallback summary
        typer.echo("visualization complete (tracking fallback: kc_* slices)")
        typer.echo(f"wrote {len(all_outputs)} plot(s)")
        if all_outputs:
            typer.echo(f"first: {all_outputs[0]}")
            typer.echo(f"last:  {all_outputs[-1]}")
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1)
