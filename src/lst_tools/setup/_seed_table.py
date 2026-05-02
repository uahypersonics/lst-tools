"""Generate ``seed_alpha.dat`` files for the Fortran tracking solver.

Reads a parsing solution (Tecplot ASCII), runs the same Hungarian-algorithm
ridge tracker that :mod:`lst_tools.process.maxima` uses for post-processing,
and emits one seed file per ``kc_*`` case directory.  Each detected ridge
corresponds to one physical mode; ``n_seeds`` evenly-spaced points along
each ridge become initial-guess seeds for the Fortran tracking solver.

Activation is via ``cfg.seed_table.enabled = true`` in ``lst.cfg``.  When
disabled (default), the file is not written and the Fortran solver falls
back to its standard marched/extrapolated initial guess.
"""

# --------------------------------------------------
# import necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# private cross-module reuse: same ridge tracker the post-processing uses
from lst_tools.process.maxima import Ridge, _track_ridges

# NOTE: smooth_contour_field is imported lazily inside
# write_seed_table_for_case() to break a circular import:
# tracking.py imports this module, and smooth_contour_field lives in tracking.py.

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# helper: filter and downsample one ridge into seed (x, f, alpha) tuples
# --------------------------------------------------
def _ridge_to_seeds(
    ridge: Ridge,
    *,
    x_arr: np.ndarray,
    freq_arr: np.ndarray,
    alpr_2d: np.ndarray,
    alpi_2d: np.ndarray,
    n_seeds: int,
    min_growth: float,
    x_range: list[float],
    f_range: list[float],
) -> list[tuple[float, float, float, float]]:
    """Pick ``n_seeds`` (x, f, alpha_r, alpha_i) tuples from a ridge.

    Args:
        ridge: A single ``Ridge`` returned by ``_track_ridges`` — list of
            (i_x, j_f) index tuples (j_f may be int or float depending on
            whether parabolic refinement was on).
        x_arr: 1-D array of streamwise coordinates from the parsing solution.
        freq_arr: 1-D array of frequencies from the parsing solution.
        alpr_2d: (nf, nx) real part of alpha at fixed beta.
        alpi_2d: (nf, nx) imaginary part of alpha at fixed beta.
        n_seeds: target number of seeds for this ridge.
        min_growth: drop ridge points where alpi < min_growth.
        x_range: optional [x_min, x_max] clipping window (empty -> no clip).
        f_range: optional [f_min, f_max] clipping window (empty -> no clip).

    Returns:
        List of (x, freq, alpha_real, alpha_imag) tuples, dimensional values
        ready to be written to ``seed_alpha.dat``.
    """

    # default empty list args -> already lists, no copy needed
    nf = freq_arr.size
    nx = x_arr.size

    # extract station and frequency indices from this ridge
    # (j may be float when parabolic interpolation was used; round to int for index lookup)
    candidate_x: list[float] = []
    candidate_f: list[float] = []
    candidate_ar: list[float] = []
    candidate_ai: list[float] = []

    # walk every (i_x, j_f) in the ridge and pull alpha at that grid point
    for i_x, j_f in ridge.indices:

        # round fractional j to nearest integer index for sampling alpha grid
        j_idx = int(round(float(j_f)))

        # bounds check
        if i_x < 0 or i_x >= nx or j_idx < 0 or j_idx >= nf:
            continue

        # pull dimensional values from the parsing solution
        x_val = float(x_arr[i_x])
        f_val = float(freq_arr[j_idx])
        alpha_real = float(alpr_2d[j_idx, i_x])

        # NOTE on sign convention:
        # The parsing tecplot column "-im(alpha)" is bound to tp.field("alpi")
        # by tecplot_ascii.py. So `alpi_2d` here is the GROWTH RATE
        # (positive when unstable), NOT the true imaginary part of alpha.
        # The Fortran tracking solver wants the true alpha_i in its seed
        # file (no implicit sign flip on read), so we:
        #   1) keep `growth_rate` for the user-facing min_growth filter
        #      (most natural: positive = unstable), and
        #   2) write `alpha_imag = -growth_rate` to disk as the true
        #      imaginary part of alpha (negative for unstable, matching
        #      alpha_0 in lst_input.dat and alpha_init internally).
        growth_rate = float(alpi_2d[j_idx, i_x])
        alpha_imag = -growth_rate

        # filter: acceptance floor on growth rate (>0 = unstable)
        if growth_rate < min_growth:
            continue

        # filter: optional x clipping window
        if x_range and not (x_range[0] <= x_val <= x_range[1]):
            continue

        # filter: optional f clipping window
        if f_range and not (f_range[0] <= f_val <= f_range[1]):
            continue

        # accept this ridge point as a candidate seed (true alpha components)
        candidate_x.append(x_val)
        candidate_f.append(f_val)
        candidate_ar.append(alpha_real)
        candidate_ai.append(alpha_imag)

    # nothing survived the filters
    if len(candidate_x) == 0:
        return []

    # if fewer candidates than requested seeds, return them all
    if len(candidate_x) <= n_seeds:
        return list(zip(candidate_x, candidate_f, candidate_ar, candidate_ai))

    # downsample evenly along the ridge in x
    # build an evenly-spaced index sequence into the candidate list
    xs = np.asarray(candidate_x)

    # pick n_seeds target x-locations spanning the candidate x-range
    x_targets = np.linspace(xs.min(), xs.max(), n_seeds)

    # for each target, pick the nearest candidate; deduplicate while preserving order
    picked_idx: list[int] = []
    seen: set[int] = set()
    for x_t in x_targets:
        k = int(np.argmin(np.abs(xs - x_t)))
        if k not in seen:
            picked_idx.append(k)
            seen.add(k)

    # assemble final tuple list
    seeds = [
        (candidate_x[k], candidate_f[k], candidate_ar[k], candidate_ai[k])
        for k in picked_idx
    ]
    return seeds


# --------------------------------------------------
# write the seed_alpha.dat file in the format expected by seed_table.f90
# --------------------------------------------------
def _write_seed_file(
    path: Path,
    seeds: list[tuple[float, float, float, float]],
    *,
    threshold: float,
    source_label: str,
    beta_label: str,
) -> None:
    """Write a seed file in the free-format ASCII layout.

    File format (matches seed_table.f90 reader):
        # comment header
        N
        x_1   f_1   alpha_real_1   alpha_imag_1
        ...

    Args:
        path: output file path (typically ``<case_dir>/seed_alpha.dat``).
        seeds: list of (x, freq, alpha_real, alpha_imag) tuples (dimensional).
        threshold: solver-side override radius (informational; written to header).
        source_label: short description of where the seeds came from.
        beta_label: beta value for this case (for header traceability).
    """

    # build the file content as a list of lines for atomic write
    lines: list[str] = []

    # header comments (all stripped by the Fortran reader as `#`/`!` comments)
    lines.append("# seed_alpha.dat -- tracking solver initial-guess seeds")
    lines.append("# generated by lst-tools (setup tracking)")
    lines.append(f"# source        : {source_label}")
    lines.append(f"# beta          : {beta_label}")
    lines.append(f"# threshold     : {threshold:g}  (normalized (x,f) override radius)")
    lines.append("# columns       : x   f   alpha_real   alpha_imag")
    lines.append("# units         : dimensional (matches alpha_0 in lst_input.dat)")
    lines.append("# convention    : alpha_imag is the TRUE imaginary part of alpha")
    lines.append("#                 (negative for unstable modes; growth rate = -alpha_imag).")
    lines.append("#                 The Fortran reader uses this value as-is; no sign flip.")
    lines.append("#")

    # number of seeds (first non-comment line)
    lines.append(f"{len(seeds)}")

    # data rows
    for x_val, f_val, ar, ai in seeds:
        lines.append(f"{x_val:25.15e} {f_val:25.15e} {ar:25.15e} {ai:25.15e}")

    # write to disk in one shot
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------
# top-level generator: build and write seed_alpha.dat for one beta case
# --------------------------------------------------
def write_seed_table_for_case(
    *,
    case_dir: str | Path,
    cfg: Any,
    tp: Any,
    idx_betr: int,
    betr_loc: float,
    source_label: str,
) -> Path | None:
    """Generate ``seed_alpha.dat`` for a single tracking case directory.

    Reads the parsing solution at ``idx_betr``, runs the ridge tracker on
    the alpi field, filters and downsamples each detected ridge to
    ``n_seeds`` evenly-spaced points, and writes the seed file inside
    ``case_dir``.

    Args:
        case_dir: path to the ``kc_*`` case directory.
        cfg: validated ``Config`` object.
        tp: parsed Tecplot solution (output of ``read_tecplot_ascii``).
        idx_betr: index into the beta dimension for this case.
        betr_loc: physical beta value for this case (header annotation).
        source_label: short description of the source file (header annotation).

    Returns:
        Path to the written seed file, or ``None`` when generation is
        disabled in config.
    """

    # convert to Path object
    case_dir = Path(case_dir)

    # short alias to the seed_table config block
    st_cfg = cfg.seed_table

    # check master switch
    if not st_cfg.enabled:
        return None

    # --- read 2-D parsing fields at this beta slice -----------------------

    # x stations (independent of beta and frequency)
    x_arr = tp.field("s")[0, 0, :].astype(float)

    # frequency line (independent of beta and station)
    freq_arr = tp.field("freq")[0, :, 0].astype(float)

    # alpha components: (nf, nx) at the requested beta index
    alpr_2d = tp.field("alpr")[idx_betr, :, :].astype(float)
    alpi_2d = tp.field("alpi")[idx_betr, :, :].astype(float)

    # debug output for devs
    logger.debug(
        "seed_table: beta=%.4f x_arr.shape=%s freq_arr.shape=%s alpi_2d.shape=%s",
        betr_loc, x_arr.shape, freq_arr.shape, alpi_2d.shape,
    )

    # --- build freq_2d for the ridge tracker ------------------------------

    # _track_ridges expects freq_2d shaped like the target field (nf, nx);
    # broadcast the 1-D freq array along the x-axis
    freq_2d = np.broadcast_to(freq_arr[:, None], alpi_2d.shape).copy()

    # --- smooth the raw alpi field for ridge detection -------------------

    # The parsing solution often has spurious local maxima (off-mode lobes,
    # noise, isolated outliers). Running the ridge tracker on the raw field
    # produces lots of short fragmentary "ridges" that are not real modes.
    # smooth_contour_field is the same de-spike + min-run + prominence
    # filter the parsing->tracking initial-guess path already uses.
    #
    # IMPORTANT: smoothing is for DETECTION only. Once the ridge tracker
    # returns ridge indices, _ridge_to_seeds samples alpha_real / alpha_imag
    # at those (i_x, j_f) locations from the ORIGINAL (unsmoothed) alpr/alpi
    # arrays so the seed values themselves stay physically faithful.

    # lazy import to break circular dependency with tracking.py
    from lst_tools.setup.tracking import smooth_contour_field

    alpi_2d_smoothed, _keep_mask = smooth_contour_field(alpi_2d, npasses=5)

    # debug output for devs
    logger.debug(
        "seed_table: beta=%.4f smoothed alpi for ridge detection (npasses=5)",
        betr_loc,
    )

    # --- run ridge tracker on the smoothed field ------------------------

    # use integer (no parabolic refinement) — sub-grid precision is overkill
    # for an initial guess and keeps the index lookup unambiguous
    ridges = _track_ridges(
        alpi_2d_smoothed,
        freq_2d,
        gate_tol=st_cfg.gate_tol,
        interpolate=False,
    )

    logger.info(
        "seed_table: beta=%.4f -> %d raw ridge(s) detected",
        betr_loc, len(ridges),
    )

    # --- filter ridges by min_valid (drop short noise ridges) ------------

    long_ridges = [r for r in ridges if len(r.indices) >= st_cfg.min_valid]

    logger.info(
        "seed_table: beta=%.4f -> %d ridge(s) after min_valid=%d filter",
        betr_loc, len(long_ridges), st_cfg.min_valid,
    )

    # --- harvest seeds from each surviving ridge -------------------------

    all_seeds: list[tuple[float, float, float, float]] = []
    for k, ridge in enumerate(long_ridges):
        ridge_seeds = _ridge_to_seeds(
            ridge,
            x_arr=x_arr,
            freq_arr=freq_arr,
            alpr_2d=alpr_2d,
            alpi_2d=alpi_2d,
            n_seeds=st_cfg.n_seeds,
            min_growth=st_cfg.min_growth,
            x_range=st_cfg.x_range,
            f_range=st_cfg.f_range,
        )
        logger.info(
            "seed_table: beta=%.4f ridge %d -> %d seed(s)",
            betr_loc, k, len(ridge_seeds),
        )
        all_seeds.extend(ridge_seeds)

    # warn if nothing was harvested
    if len(all_seeds) == 0:
        logger.warning(
            "seed_table: beta=%.4f -> no seeds harvested; writing empty file"
            " (Fortran solver will fall back to standard initial guess)",
            betr_loc,
        )

    # --- write the seed file ---------------------------------------------

    out_path = case_dir / st_cfg.output_file
    _write_seed_file(
        out_path,
        all_seeds,
        threshold=st_cfg.threshold,
        source_label=source_label,
        beta_label=f"{betr_loc:g}",
    )

    logger.info(
        "seed_table: wrote %d seed(s) to %s",
        len(all_seeds), out_path,
    )

    return out_path
