"""
Post-processing functionality for LST tracking calculations

This module orchestrates two post-processing steps:
  1. **maxima** — ridge extraction from each kc_* case directory
  2. **volume** — assembly of 2-D slices into a 3-D volume file

By default both steps run.  The ``do_maxima`` and ``do_volume`` flags
can be used as "only" selectors (e.g. ``do_maxima=True, do_volume=False``).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations
import logging
from pathlib import Path

import typer

from lst_tools.config import Config
from lst_tools.utils.progress import progress
from ._discover import discover_pattern_dirs
from .maxima import extract_maxima
from .volume import assemble_volume


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# main function to process tracking results
# --------------------------------------------------
def tracking_process(
    *,
    cfg: Config | None = None,
    do_maxima: bool = True,
    do_volume: bool = True,
    work_dir: Path | None = None,
    kc_dirs: list[Path] | None = None,
    interpolate: bool | None = None,
    plain_output: bool = False,
) -> Path:
    """
    Process LST tracking calculation results.

    Parameters
    ----------
    cfg : Optional[Config]
        Configuration object. Processing defaults come from
        ``cfg.processing`` (gate_tol, min_valid, peak_order,
        interpolate).  CLI flags override config values.
    do_maxima : bool
        If True, run ridge-line maxima extraction for every kc_* case.
    do_volume : bool
        If True, assemble 2-D slices into a 3-D volume file.
    work_dir : Optional[Path]
        Working directory containing kc_* subdirectories.
        Defaults to the current working directory.
    kc_dirs : Optional[list[Path]]
        Specific kc_* directories to process. If None, discovers all
        kc_* directories in work_dir.
    interpolate : Optional[bool]
        Use parabolic interpolation for peak refinement.
        If None, falls back to ``cfg.processing.interpolate``.
    plain_output : bool
        If True, use line-by-line text output instead of rich progress bars
        where supported.

    Returns
    -------
    Path
        Path to the working directory.
    """

    # --------------------------------------------------
    # get working directory (default to current directory if nothing provided)
    # --------------------------------------------------
    if work_dir is None:
        work_dir = Path.cwd()

    # --------------------------------------------------
    # resolve processing settings from config + CLI overrides
    # --------------------------------------------------
    proc = cfg.processing if cfg is not None else None

    # CLI flag (if explicitly set) overrides config value
    use_interpolate = interpolate if interpolate is not None else (
        proc.interpolate if proc is not None else False
    )
    gate_tol = proc.gate_tol if proc is not None else 0.10
    min_valid = proc.min_valid if proc is not None else 40

    logger.info("processing tracking results in %s", work_dir)

    # --------------------------------------------------
    # discover tracking directories (kc_*)
    # --------------------------------------------------
    if kc_dirs is None:
        kc_dirs = discover_pattern_dirs(work_dir, "kc_*")

    # nothing found to process, so just return the working directory
    if not kc_dirs:
        logger.warning("no kc_* directories found in %s", work_dir)
        return work_dir

    logger.debug("processing %d kc_* directories", len(kc_dirs))

    # --------------------------------------------------
    # processing maxima: ridge-line maxima extraction
    # --------------------------------------------------
    if do_maxima:

        typer.echo("starting maxima extraction...")

        total_files = 0

        if plain_output:
            for kc_dir in kc_dirs:
                typer.echo(f"- {kc_dir.name}")
                written = extract_maxima(
                    kc_dir,
                    interpolate=use_interpolate,
                    gate_tol=gate_tol,
                    min_valid=min_valid,
                )
                total_files += len(written)
        else:
            with progress(
                total=len(kc_dirs),
                description="process.tracking.maxima",
                persist=True,
            ) as advance:
                for kc_dir in kc_dirs:
                    written = extract_maxima(
                        kc_dir,
                        interpolate=use_interpolate,
                        gate_tol=gate_tol,
                        min_valid=min_valid,
                    )
                    total_files += len(written)
                    advance()

        typer.echo(f"maxima extraction complete: {total_files} file(s) written")

    # --------------------------------------------------
    # processing volume: assemble 3-D volume from 2-D slices
    # --------------------------------------------------

    if do_volume:

        typer.echo("starting volume assembly...")

        vol_path = assemble_volume(work_dir, plain_output=plain_output)

        if vol_path is not None:
            typer.echo(f"volume assembly complete: {vol_path.name}")
        else:
            typer.echo("volume assembly produced no output")
            logger.warning("volume assembly produced no output")

    return work_dir
