"""lst-tools visualize — stage-aware wrappers for quick LST contour plots."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# workflow paths
# --------------------------------------------------
DEFAULT_CONFIG_PATH = Path("cfd-viz-lst.toml")
DEFAULT_PARSING_INPUT = Path("growth_rate_with_nfact_amps.dat")
DEFAULT_PARSING_OUTPUT = Path("alpi_contours_parsing")
DEFAULT_TRACKING_INPUT = Path("lst_vol.dat")
DEFAULT_TRACKING_OUTPUT = Path("alpi_contours_tracking")


# --------------------------------------------------
# import optional cfd-viz backend
# --------------------------------------------------
def _load_visualization_backend() -> Any:
    """Import the optional cfd-viz LST API with actionable error guidance."""

    try:
        lst_module = importlib.import_module("cfd_viz.lst")
    except Exception as exc:
        raise RuntimeError(
            "visualization support is required for visualize commands. "
            "Install lst-tools with the 'viz' extra."
        ) from exc

    return lst_module


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
# load cfd-viz configuration
# --------------------------------------------------
def _load_visualization_config(config_path: Path | None) -> tuple[Any, Any, bool]:
    """Load explicit/local configuration or cfd-viz built-in defaults."""

    # select an explicit path or discover the standard local filename
    selected_path = config_path
    if selected_path is None and DEFAULT_CONFIG_PATH.exists():
        selected_path = DEFAULT_CONFIG_PATH

    # load configured values or ask cfd-viz for its authoritative defaults
    lst_module = _load_visualization_backend()
    config_found = selected_path is not None
    if config_found:
        config = lst_module.load_lst_config(selected_path)
    else:
        config = lst_module.default_lst_config()

    return lst_module, config, config_found


# --------------------------------------------------
# render through a cfd-viz configuration
# --------------------------------------------------
def _visualize_files(
    *,
    stage: str,
    input_files: list[Path],
    out_dir: Path,
    lst_module: Any,
    config: Any,
    prefix_suffixes: list[str | None] | None = None,
    single_plane: bool = False,
) -> list[Path]:
    """Delegate a discovered file collection to cfd-viz."""

    # delegate configuration, bounds, naming, and rendering to cfd-viz
    files = lst_module.render_configured_lst_collection(
        config,
        input_files,
        output_dir=out_dir,
        prefix_suffixes=prefix_suffixes,
        single_plane=single_plane,
        show=False,
    )

    # print user summary
    typer.echo(f"visualization complete ({stage})")
    typer.echo(f"wrote {len(files)} plot(s)")
    if files:
        typer.echo(f"first: {files[0]}")
        typer.echo(f"last:  {files[-1]}")

    return files


# --------------------------------------------------
# visualization configuration initializer
# --------------------------------------------------
def cmd_visualize_init(
    output: Annotated[
        Path,
        typer.Argument(help="cfd-viz LST configuration to create."),
    ] = DEFAULT_CONFIG_PATH,
    force: Annotated[
        bool,
        typer.Option("--force", "-f", help="Replace an existing configuration."),
    ] = False,
) -> None:
    """Create an editable cfd-viz LST plotting configuration."""

    # lazily import the visualization backend
    try:
        lst_module = importlib.import_module("cfd_viz.lst")
        write_default_lst_config = lst_module.write_default_lst_config
    except Exception as exc:
        raise RuntimeError(
            "visualization support is required to create the plotting configuration"
        ) from exc

    # write the configuration without replacing user edits by default
    try:
        written = write_default_lst_config(output, force=force)
    except FileExistsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"wrote {written}")


# --------------------------------------------------
# parsing wrapper
# --------------------------------------------------
def cmd_visualize_parsing(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Parsing Tecplot input file (default: growth_rate_with_nfact_amps.dat).",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for rendered parsing plots.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="cfd-viz LST config; defaults to cfd-viz-lst.toml when present.",
        ),
    ] = None,
) -> None:
    """Visualize parsing results through cfd-viz."""
    try:
        # load cfd-viz policy and select workflow-owned paths
        lst_module, plot_config, config_found = _load_visualization_config(config_path)
        selected_input = input_path
        if selected_input is None:
            if config_found:
                selected_input = plot_config.input_path
            else:
                selected_input = DEFAULT_PARSING_INPUT

        selected_out_dir = out_dir
        if selected_out_dir is None:
            if config_found:
                selected_out_dir = plot_config.output_dir
            else:
                selected_out_dir = DEFAULT_PARSING_OUTPUT

        _visualize_files(
            stage="parsing",
            input_files=[selected_input],
            out_dir=selected_out_dir,
            lst_module=lst_module,
            config=plot_config,
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc


# --------------------------------------------------
# tracking wrapper
# --------------------------------------------------
def cmd_visualize_tracking(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            "-i",
            help="Tracking Tecplot volume input file (default: lst_vol.dat).",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Output directory for rendered tracking plots.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="cfd-viz LST config; defaults to cfd-viz-lst.toml when present.",
        ),
    ] = None,
) -> None:
    """Visualize tracking results using optional cfd-viz configuration."""
    try:
        # load cfd-viz plot configuration from the current directory or defaults
        lst_module, plot_config, config_found = _load_visualization_config(config_path)

        # set default input file used for tracking fallback behavior
        selected_input = input_path
        if selected_input is None:
            if config_found:
                selected_input = plot_config.input_path
            else:
                selected_input = DEFAULT_TRACKING_INPUT

        selected_out_dir = out_dir
        if selected_out_dir is None:
            if config_found:
                selected_out_dir = plot_config.output_dir
            else:
                selected_out_dir = DEFAULT_TRACKING_OUTPUT

        # primary path: use consolidated tracking volume when present
        if selected_input is not None and selected_input.exists():
            _visualize_files(
                stage="tracking",
                input_files=[selected_input],
                out_dir=selected_out_dir,
                lst_module=lst_module,
                config=plot_config,
            )
            return

        # fallback path: discover individual kc_* tracking slices
        if input_path is not None:
            raise FileNotFoundError(f"input file not found: {selected_input}")

        root = Path(".").resolve()
        slice_files = _discover_tracking_files(root)
        if not slice_files:
            raise FileNotFoundError(
                f"{DEFAULT_TRACKING_INPUT} not found and no kc_* tracking slices discovered"
            )

        # delegate shared bounds and rendering for the discovered collection
        prefix_suffixes = [
            input_file.parent.name.removeprefix("kc_") for input_file in slice_files
        ]
        _visualize_files(
            stage="tracking fallback: kc_* slices",
            input_files=slice_files,
            out_dir=selected_out_dir,
            lst_module=lst_module,
            config=plot_config,
            prefix_suffixes=prefix_suffixes,
            single_plane=True,
        )
    except Exception as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
