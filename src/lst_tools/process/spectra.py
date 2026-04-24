"""
Post-processing functionality for LST spectra calculations

This module provides functions to process the output data from LST spectra
calculations, including spectral analysis, frequency-wavenumber data collection,
and visualization preparation.
"""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass, field
import logging
import re
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from lst_tools.data_io import read_tecplot_ascii

logger = logging.getLogger(__name__)


# --------------------------------------------------
# constants
# --------------------------------------------------
BRANCH_CONTOUR_FILL_VALUE = -999.0
DETACHMENT_GAP_THRESHOLD = 1.5
DETACHMENT_MIN_RUN_POINTS = 3
DETACHMENT_NEIGHBOR_COUNT = 4
DETACHED_BRANCH_GATE_MIN = 0.40
DETACHED_BRANCH_GATE_FACTOR = 1.5


# --------------------------------------------------
# data structures
# --------------------------------------------------
@dataclass
class SpectraBranch:
    """A tracked branch of eigenvalues across streamwise stations."""

    points: list[dict[str, float]] = field(default_factory=list)
    score: float = 0.0
    coverage_fraction: float = 0.0
    isolation_median: float = 0.0
    isolation_max: float = 0.0
    smoothness_factor: float = 1.0
    alpha_i_penalty: float = 1.0
    alpha_r_penalty: float = 1.0
    start_neutral_factor: float = 1.0
    amplification_bonus: float = 1.0
    gap_median: float = 0.0
    gap_max: float = 0.0
    detached_fraction: float = 0.0
    longest_detached_run: int = 0
    birth_x: float | None = None
    absorption_x: float | None = None


@dataclass
class SpectraProcessingOptions:
    """Processing controls for spectra gating and branch tracking."""

    alpr_min: float | None = None
    alpr_max: float | None = None
    alpi_min: float | None = None
    alpi_max: float | None = None
    branch_gate: float = 0.25
    branch_min_points: int = 2
    isolation_k: int = 3
    isolation_threshold: float | None = None
    classify_min_points: int = 3


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _emit_status(
    reporter: Callable[[str], None] | None,
    message: str,
) -> None:
    """Send a user-facing status message when a reporter is available."""

    # write status output for interactive callers
    if reporter is not None:
        reporter(message)


def _resolve_spectra_processing_options(
    cfg: Mapping[str, Any] | None,
) -> SpectraProcessingOptions:
    """Resolve spectra processing options from config defaults."""

    # default empty dicts for optional arguments
    spectra_processing: Any = None

    # extract the spectra subsection from either a config dataclass or plain dict
    if cfg is not None:
        if hasattr(cfg, "processing"):
            spectra_processing = cfg.processing.spectra
        elif isinstance(cfg, Mapping):
            processing = cfg.get("processing", {})
            if isinstance(processing, Mapping):
                spectra_processing = processing.get("spectra")

    # fall back to built-in defaults when no spectra config is available
    if spectra_processing is None:
        return SpectraProcessingOptions()

    # read from plain mappings
    if isinstance(spectra_processing, Mapping):
        return SpectraProcessingOptions(
            alpr_min=spectra_processing.get("alpr_min"),
            alpr_max=spectra_processing.get("alpr_max"),
            alpi_min=spectra_processing.get("alpi_min"),
            alpi_max=spectra_processing.get("alpi_max"),
            branch_gate=float(spectra_processing.get("branch_gate", 0.25)),
            branch_min_points=int(spectra_processing.get("branch_min_points", 2)),
            isolation_k=int(spectra_processing.get("isolation_k", 3)),
            isolation_threshold=spectra_processing.get("isolation_threshold"),
            classify_min_points=int(spectra_processing.get("classify_min_points", 3)),
        )

    # read from typed config objects
    return SpectraProcessingOptions(
        alpr_min=getattr(spectra_processing, "alpr_min", None),
        alpr_max=getattr(spectra_processing, "alpr_max", None),
        alpi_min=getattr(spectra_processing, "alpi_min", None),
        alpi_max=getattr(spectra_processing, "alpi_max", None),
        branch_gate=float(getattr(spectra_processing, "branch_gate", 0.25)),
        branch_min_points=int(getattr(spectra_processing, "branch_min_points", 2)),
        isolation_k=int(getattr(spectra_processing, "isolation_k", 3)),
        isolation_threshold=getattr(spectra_processing, "isolation_threshold", None),
        classify_min_points=int(getattr(spectra_processing, "classify_min_points", 3)),
    )


def _discover_case_info(work_dir: Path) -> list[dict[str, Any]]:
    """Discover spectra case directories and parse their physical parameters."""

    # pattern to match: x_00pt10_m_f_0006pt00_khz_beta_pos0500pt00
    case_pattern = re.compile(
        r"^x_(\d+pt\d+)_m_f_(\d+pt\d+)_khz_beta_(pos|neg)(\d+pt\d+)$"
    )

    case_info: list[dict[str, Any]] = []

    # read all matching case directories in the working directory
    for item in sorted(work_dir.iterdir()):
        if not item.is_dir():
            continue

        match = case_pattern.match(item.name)
        if match is None:
            continue

        x_str, f_str, sign, beta_str = match.groups()

        # parse physical values from the directory name
        x_val = float(x_str.replace("pt", "."))
        f_val = float(f_str.replace("pt", ".")) * 1000.0
        beta_val = float(beta_str.replace("pt", "."))
        if sign == "neg":
            beta_val = -beta_val

        case_info.append(
            {
                "path": item,
                "name": item.name,
                "x": x_val,
                "freq": f_val,
                "beta": beta_val,
            }
        )

    return case_info


def _load_spectra_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a spectra slice as arrays of alpha_r and alpha_i."""

    # first try the Tecplot ASCII reader for fully-labeled datasets
    try:
        tecplot_data = read_tecplot_ascii(path)
        alpha_r = np.asarray(tecplot_data.field("alpr"), dtype=float).reshape(-1)
        alpha_i = np.asarray(tecplot_data.field("alpi"), dtype=float).reshape(-1)
        return alpha_r, alpha_i
    except Exception:
        pass

    # fall back to the legacy plain two-column eigenvalue format
    raw_data = np.loadtxt(path, dtype=float)
    if raw_data.ndim == 1:
        raw_data = raw_data[None, :]

    if raw_data.ndim != 2 or raw_data.shape[1] < 2:
        raise ValueError(f"spectra file {path} is not a readable 2-column dataset")

    alpha_r = np.asarray(raw_data[:, 0], dtype=float)
    alpha_i = np.asarray(raw_data[:, 1], dtype=float)
    return alpha_r, alpha_i


def _has_alpha_gating(options: SpectraProcessingOptions) -> bool:
    """Return True when any alpha-space gating bound is enabled."""

    return any(
        value is not None
        for value in (
            options.alpr_min,
            options.alpr_max,
            options.alpi_min,
            options.alpi_max,
        )
    )


def _apply_alpha_gating(
    alpha_r: np.ndarray,
    alpha_i: np.ndarray,
    options: SpectraProcessingOptions,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Apply optional alpha-space limits to one spectra slice."""

    # initialize a keep mask that accepts every point by default
    keep_mask = np.ones(alpha_r.shape, dtype=bool)

    # check alpha_r lower bound
    if options.alpr_min is not None:
        keep_mask &= alpha_r >= options.alpr_min

    # check alpha_r upper bound
    if options.alpr_max is not None:
        keep_mask &= alpha_r <= options.alpr_max

    # check alpha_i lower bound
    if options.alpi_min is not None:
        keep_mask &= alpha_i >= options.alpi_min

    # check alpha_i upper bound
    if options.alpi_max is not None:
        keep_mask &= alpha_i <= options.alpi_max

    # build the filtered point clouds
    alpha_r_filtered = alpha_r[keep_mask]
    alpha_i_filtered = alpha_i[keep_mask]

    return alpha_r_filtered, alpha_i_filtered, int(alpha_r.size), int(alpha_r_filtered.size)


def _compute_robust_scale(values: np.ndarray) -> float:
    """Compute a robust scale for alpha-space normalization."""

    # return a safe default when there is not enough data to estimate a spread
    if values.size < 2:
        return 1.0

    # use a percentile spread so extreme outliers do not collapse the density metric
    value_lo = float(np.percentile(values, 5.0))
    value_hi = float(np.percentile(values, 95.0))
    return max(value_hi - value_lo, 1.0)


def _compute_group_isolation_scores(
    cases: list[dict[str, Any]],
    *,
    isolation_k: int,
) -> None:
    """Compute per-point isolation scores for one (freq, beta) spectra group."""

    # collect all filtered points in the group to define a robust normalization scale
    alpha_r_all = [
        case["alpha_r_filtered"]
        for case in cases
        if case["alpha_r_filtered"].size > 0
    ]
    alpha_i_all = [
        case["alpha_i_filtered"]
        for case in cases
        if case["alpha_i_filtered"].size > 0
    ]

    # handle the degenerate case where no filtered points exist in this group
    if not alpha_r_all or not alpha_i_all:
        for case in cases:
            case["isolation_score_filtered"] = np.empty(0, dtype=float)
        return

    alpha_r_scale = _compute_robust_scale(np.concatenate(alpha_r_all))
    alpha_i_scale = _compute_robust_scale(np.concatenate(alpha_i_all))

    # compute a local k-nearest-neighbor distance for every station separately
    for case in cases:
        alpha_r = case["alpha_r_filtered"]
        alpha_i = case["alpha_i_filtered"]
        n_point = int(alpha_r.size)

        if n_point == 0:
            case["isolation_score_filtered"] = np.empty(0, dtype=float)
            continue

        if n_point == 1:
            case["isolation_score_filtered"] = np.array([0.0], dtype=float)
            continue

        points = np.column_stack((alpha_r / alpha_r_scale, alpha_i / alpha_i_scale))
        delta = points[:, None, :] - points[None, :, :]
        distance = np.hypot(delta[:, :, 0], delta[:, :, 1])
        np.fill_diagonal(distance, np.inf)

        effective_k = min(max(isolation_k, 1), n_point - 1)
        kth_index = effective_k - 1
        isolation_score = np.partition(distance, kth_index, axis=1)[:, kth_index]
        case["isolation_score_filtered"] = np.asarray(isolation_score, dtype=float)


def _compute_group_gap_scores(
    cases: list[dict[str, Any]],
    *,
    neighbor_count: int,
) -> None:
    """Compute a local normalized separation score for every filtered point."""

    # collect all filtered points in the group to define a robust normalization scale
    alpha_r_all = [
        case["alpha_r_filtered"]
        for case in cases
        if case["alpha_r_filtered"].size > 0
    ]
    alpha_i_all = [
        case["alpha_i_filtered"]
        for case in cases
        if case["alpha_i_filtered"].size > 0
    ]

    # handle the degenerate case where no filtered points exist in this group
    if not alpha_r_all or not alpha_i_all:
        for case in cases:
            case["gap_score_filtered"] = np.empty(0, dtype=float)
        return

    alpha_r_scale = _compute_robust_scale(np.concatenate(alpha_r_all))
    alpha_i_scale = _compute_robust_scale(np.concatenate(alpha_i_all))

    # compute a normalized gap ratio for every station separately
    for case in cases:
        alpha_r = case["alpha_r_filtered"]
        alpha_i = case["alpha_i_filtered"]
        n_point = int(alpha_r.size)

        if n_point == 0:
            case["gap_score_filtered"] = np.empty(0, dtype=float)
            continue

        if n_point == 1:
            case["gap_score_filtered"] = np.array([0.0], dtype=float)
            continue

        points = np.column_stack((alpha_r / alpha_r_scale, alpha_i / alpha_i_scale))
        delta = points[:, None, :] - points[None, :, :]
        distance = np.hypot(delta[:, :, 0], delta[:, :, 1])
        np.fill_diagonal(distance, np.inf)

        nearest_distance = np.min(distance, axis=1)

        if n_point == 2:
            case["gap_score_filtered"] = np.ones(2, dtype=float)
            continue

        effective_neighbor_count = min(max(neighbor_count, 2), n_point - 1)
        kth_index = effective_neighbor_count - 1
        neighbor_indices = np.argpartition(distance, kth_index, axis=1)[:, :effective_neighbor_count]

        local_spacing = np.empty(n_point, dtype=float)

        # build a local spacing estimate from the nearest-neighbor crowd around each point
        for point_index in range(n_point):
            local_neighbor_indices = neighbor_indices[point_index]
            local_neighbor_spacing = nearest_distance[local_neighbor_indices]

            if local_neighbor_spacing.size == 0:
                spacing_value = float(nearest_distance[point_index])
            else:
                spacing_value = float(np.median(local_neighbor_spacing))

            local_spacing[point_index] = max(spacing_value, 1.0e-12)

        gap_score = nearest_distance / local_spacing
        case["gap_score_filtered"] = np.asarray(gap_score, dtype=float)


def _compute_alpha_plane_cost(
    point_left: np.ndarray,
    point_right: np.ndarray,
) -> float:
    """Compute the normalized alpha-plane distance between two spectra points."""

    alpha_r_ref = max(abs(float(point_left[0])), abs(float(point_right[0])), 1.0)
    alpha_i_ref = max(abs(float(point_left[1])), abs(float(point_right[1])), 1.0)
    alpha_r_diff = abs(float(point_left[0]) - float(point_right[0])) / alpha_r_ref
    alpha_i_diff = abs(float(point_left[1]) - float(point_right[1])) / alpha_i_ref
    return float(np.hypot(alpha_r_diff, alpha_i_diff))


def _resolve_match_gate(
    *,
    base_gate: float,
    previous_gap_score: float,
    current_gap_score: float,
) -> float:
    """Return the effective match gate for one candidate branch continuation."""

    # keep the standard gate for ordinary embedded points
    effective_gate = base_gate

    # allow a slightly looser continuation when the branch is detached on either side
    is_detached_pair = (
        previous_gap_score >= DETACHMENT_GAP_THRESHOLD
        or current_gap_score >= DETACHMENT_GAP_THRESHOLD
    )
    if is_detached_pair:
        detached_gate = max(
            base_gate * DETACHED_BRANCH_GATE_FACTOR,
            DETACHED_BRANCH_GATE_MIN,
        )
        effective_gate = detached_gate

    return effective_gate


def _build_seed_candidate_mask(case: dict[str, Any]) -> np.ndarray:
    """Build a candidate-seed mask from local density and packet-edge heuristics."""

    alpha_r = case["alpha_r_filtered"]
    alpha_i = case["alpha_i_filtered"]
    isolation_score = case["isolation_score_filtered"]
    n_point = int(alpha_r.size)

    # return an empty mask when this station has no gated points
    if n_point == 0:
        return np.zeros(0, dtype=bool)

    # keep all points when the station is too small for density logic to help
    if n_point <= 3:
        return np.ones(n_point, dtype=bool)

    seed_mask = np.zeros(n_point, dtype=bool)

    # keep the most isolated tail of the cloud as discrete-like seed candidates
    isolated_cutoff = float(np.percentile(isolation_score, 85.0))
    seed_mask |= isolation_score >= isolated_cutoff

    # identify the dense background cloud and keep its lower alpha_i edge as candidate seeds
    dense_cutoff = float(np.percentile(isolation_score, 50.0))
    dense_indices = np.flatnonzero(isolation_score <= dense_cutoff)
    if dense_indices.size > 0:
        n_edge = min(3, dense_indices.size)
        edge_order = dense_indices[np.argsort(alpha_i[dense_indices])[:n_edge]]
        seed_mask[edge_order] = True

    return seed_mask


def _apply_seed_persistence_filter(
    cases: list[dict[str, Any]],
    *,
    persistence_gate: float,
) -> int:
    """Keep only seed candidates that persist across neighboring x-stations."""

    accepted_masks: list[np.ndarray] = []
    candidate_total = 0
    accepted_total = 0

    for case in cases:
        candidate_mask = case["seed_candidate_mask_filtered"]
        accepted_masks.append(np.zeros(candidate_mask.shape, dtype=bool))
        candidate_total += int(np.count_nonzero(candidate_mask))

    # if there is only one station, accept every candidate seed directly
    if len(cases) <= 1:
        for case in cases:
            case["seed_mask_filtered"] = np.asarray(case["seed_candidate_mask_filtered"], dtype=bool)
            accepted_total += int(np.count_nonzero(case["seed_mask_filtered"]))
        return accepted_total

    # keep candidates that can be matched to a seed in a neighboring station
    for case_index, case in enumerate(cases):
        candidate_mask = case["seed_candidate_mask_filtered"]
        alpha_r = case["alpha_r_filtered"]
        alpha_i = case["alpha_i_filtered"]

        for point_index in np.flatnonzero(candidate_mask):
            point_here = np.array([alpha_r[point_index], alpha_i[point_index]], dtype=float)
            has_neighbor = False

            for neighbor_index in (case_index - 1, case_index + 1):
                if neighbor_index < 0 or neighbor_index >= len(cases):
                    continue

                neighbor_case = cases[neighbor_index]
                neighbor_mask = neighbor_case["seed_candidate_mask_filtered"]
                if not np.any(neighbor_mask):
                    continue

                neighbor_alpha_r = neighbor_case["alpha_r_filtered"]
                neighbor_alpha_i = neighbor_case["alpha_i_filtered"]
                neighbor_points = np.column_stack((neighbor_alpha_r[neighbor_mask], neighbor_alpha_i[neighbor_mask]))

                min_cost = min(
                    _compute_alpha_plane_cost(point_here, neighbor_point)
                    for neighbor_point in neighbor_points
                )
                if min_cost <= persistence_gate:
                    has_neighbor = True
                    break

            if has_neighbor:
                accepted_masks[case_index][point_index] = True

    # fall back to all candidates if the persistence filter removed everything
    if not any(np.any(mask) for mask in accepted_masks):
        for case in cases:
            case["seed_mask_filtered"] = np.asarray(case["seed_candidate_mask_filtered"], dtype=bool)
            accepted_total += int(np.count_nonzero(case["seed_mask_filtered"]))
        return accepted_total

    for case, accepted_mask in zip(cases, accepted_masks):
        case["seed_mask_filtered"] = accepted_mask
        accepted_total += int(np.count_nonzero(accepted_mask))

    logger.info(
        "seed persistence accepted %d/%d candidate seed point(s)",
        accepted_total,
        candidate_total,
    )
    return accepted_total


def _select_branch_seeds(
    cases: list[dict[str, Any]],
    *,
    branch_gate: float,
) -> int:
    """Select persistent seed points that are allowed to start tracked branches."""

    # build local candidate seeds at each station from density and packet-edge logic
    for case in cases:
        case["seed_candidate_mask_filtered"] = _build_seed_candidate_mask(case)

    persistence_gate = max(2.0 * branch_gate, 0.40)
    return _apply_seed_persistence_filter(cases, persistence_gate=persistence_gate)


def _format_group_output_name(freq: float, beta: float) -> str:
    """Format the Tecplot output file name for one (freq, beta) pair."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_branch_output_name(freq: float, beta: float) -> str:
    """Format the tracked-branch Tecplot output file name."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_branches_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_directional_branch_output_name(
    freq: float,
    beta: float,
    direction: str,
) -> str:
    """Format the tracked-branch Tecplot output file name for one tracking direction."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_branches_{direction}_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_classified_output_name(freq: float, beta: float) -> str:
    """Format the classified-branch Tecplot output file name."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_classified_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_branch_contour_output_name(freq: float, beta: float) -> str:
    """Format the tracked-branch contour Tecplot output file name."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_branch_contours_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_directional_branch_contour_output_name(
    freq: float,
    beta: float,
    direction: str,
) -> str:
    """Format the tracked-branch contour Tecplot output file name for one direction."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_branch_contours_{direction}_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_classified_contour_output_name(freq: float, beta: float) -> str:
    """Format the classified-branch contour Tecplot output file name."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_classified_contours_f_{freq_str}_khz_beta_{beta_str}.dat"


def _format_branch_summary_output_name(freq: float, beta: float) -> str:
    """Format the tracked-branch summary output file name."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_branch_summary_f_{freq_str}_khz_beta_{beta_str}.csv"


def _format_directional_branch_summary_output_name(
    freq: float,
    beta: float,
    direction: str,
) -> str:
    """Format the tracked-branch summary output file name for one direction."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_branch_summary_{direction}_f_{freq_str}_khz_beta_{beta_str}.csv"


def _format_classified_summary_output_name(freq: float, beta: float) -> str:
    """Format the classified-branch summary output file name."""

    freq_str = f"{freq / 1000.0:07.2f}".replace(".", "pt")

    if beta >= 0.0:
        beta_str = f"pos{abs(beta):07.2f}".replace(".", "pt")
    else:
        beta_str = f"neg{abs(beta):07.2f}".replace(".", "pt")

    return f"spectra_classified_summary_f_{freq_str}_khz_beta_{beta_str}.csv"


def _write_group_animation_file(
    out_path: Path,
    cases: list[dict[str, Any]],
) -> Path:
    """Write one Tecplot animation file for a fixed (freq, beta) group."""

    # sort zones by x so Tecplot animates in streamwise order
    sorted_cases = sorted(cases, key=lambda case: case["x"])

    # write one zone per x-station, preserving the raw eigenvalue cloud
    with out_path.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "spectra_animation"\n')
        stream.write(
            'VARIABLES = "alpha_r" "alpha_i" "x" "freq" "beta"\n'
        )

        for case in sorted_cases:
            alpha_r = case["alpha_r"]
            alpha_i = case["alpha_i"]
            n_point = alpha_r.size
            zone_name = (
                f'x={case["x"]:.6e} '
                f'f={case["freq"] / 1000.0:.3f}kHz '
                f'beta={case["beta"]:.6e}'
            )

            stream.write(
                f'ZONE T="{zone_name}", I={n_point}, DATAPACKING=POINT, '
                f'STRANDID=1, SOLUTIONTIME={case["x"]:.8e}\n'
            )

            for alpha_r_value, alpha_i_value in zip(alpha_r, alpha_i):
                stream.write(
                    f"{alpha_r_value:.8e} "
                    f"{alpha_i_value:.8e} "
                    f"{case['x']:.8e} "
                    f"{case['freq']:.8e} "
                    f"{case['beta']:.8e}\n"
                )

    logger.info(
        "wrote spectra animation file %s with %d zone(s)",
        out_path,
        len(sorted_cases),
    )
    return out_path


def _track_spectra_branches(
    cases: list[dict[str, Any]],
    *,
    branch_gate: float,
    min_points: int,
    direction: str = "forward",
) -> list[SpectraBranch]:
    """Track filtered spectra points across x-stations starting only from seeds."""

    # sort the slices in the requested tracking direction before matching
    reverse_sort = direction == "backward"
    sorted_cases = sorted(cases, key=lambda case: case["x"], reverse=reverse_sort)

    # collect all filtered points to build normalization scales for matching
    alpha_r_all = [case["alpha_r_filtered"] for case in sorted_cases if case["alpha_r_filtered"].size > 0]
    alpha_i_all = [case["alpha_i_filtered"] for case in sorted_cases if case["alpha_i_filtered"].size > 0]

    # return early when no filtered candidate points exist
    if not alpha_r_all or not alpha_i_all:
        return []

    branches: list[SpectraBranch] = []
    active_branch_ids: list[int] = []

    # walk station by station and match current points to active branches
    for case in sorted_cases:
        alpha_r = case["alpha_r_filtered"]
        alpha_i = case["alpha_i_filtered"]

        # skip stations that do not retain any candidate points after gating
        if alpha_r.size == 0:
            continue

        current_points = np.column_stack((alpha_r, alpha_i))
        seed_mask = case.get("seed_mask_filtered")
        if seed_mask is None:
            seed_mask = np.ones(alpha_r.shape, dtype=bool)

        # initialize branches from the first station that contains accepted seeds
        if not active_branch_ids:
            for point_index, (alpha_r_value, alpha_i_value) in enumerate(current_points):
                if not bool(seed_mask[point_index]):
                    continue

                branch = SpectraBranch(
                    points=[
                        {
                            "x": float(case["x"]),
                            "freq": float(case["freq"]),
                            "beta": float(case["beta"]),
                            "alpha_r": float(alpha_r_value),
                            "alpha_i": float(alpha_i_value),
                            "isolation_score": float(case["isolation_score_filtered"][point_index]),
                            "gap_score": float(case["gap_score_filtered"][point_index]),
                        }
                    ]
                )
                branches.append(branch)
                active_branch_ids.append(len(branches) - 1)

            if not active_branch_ids:
                continue

            continue

        # build the set of last known active branch points
        active_points = np.array(
            [
                [
                    branches[branch_id].points[-1]["alpha_r"],
                    branches[branch_id].points[-1]["alpha_i"],
                ]
                for branch_id in active_branch_ids
            ],
            dtype=float,
        )

        # build a relative alpha-plane distance matrix using local point magnitudes
        alpha_r_ref = np.maximum(
            np.maximum(np.abs(active_points[:, None, 0]), np.abs(current_points[None, :, 0])),
            1.0,
        )
        alpha_i_ref = np.maximum(
            np.maximum(np.abs(active_points[:, None, 1]), np.abs(current_points[None, :, 1])),
            1.0,
        )
        alpha_r_diff = np.abs(active_points[:, None, 0] - current_points[None, :, 0]) / alpha_r_ref
        alpha_i_diff = np.abs(active_points[:, None, 1] - current_points[None, :, 1]) / alpha_i_ref
        cost = np.hypot(alpha_r_diff, alpha_i_diff)

        # solve the optimal assignment problem for the current station
        row_idx, col_idx = linear_sum_assignment(cost)
        matched_branch_ids: set[int] = set()
        matched_point_ids: set[int] = set()
        next_active_branch_ids: list[int] = []

        # keep only assignments that satisfy the normalized jump gate
        for row_loc, col_loc in zip(row_idx, col_idx):
            branch_id = active_branch_ids[row_loc]
            previous_gap_score = float(branches[branch_id].points[-1]["gap_score"])
            current_gap_score = float(case["gap_score_filtered"][col_loc])
            effective_gate = _resolve_match_gate(
                base_gate=branch_gate,
                previous_gap_score=previous_gap_score,
                current_gap_score=current_gap_score,
            )
            if float(cost[row_loc, col_loc]) > effective_gate:
                continue

            branches[branch_id].points.append(
                {
                    "x": float(case["x"]),
                    "freq": float(case["freq"]),
                    "beta": float(case["beta"]),
                    "alpha_r": float(current_points[col_loc, 0]),
                    "alpha_i": float(current_points[col_loc, 1]),
                    "isolation_score": float(case["isolation_score_filtered"][col_loc]),
                    "gap_score": float(case["gap_score_filtered"][col_loc]),
                }
            )
            matched_branch_ids.add(branch_id)
            matched_point_ids.add(int(col_loc))
            next_active_branch_ids.append(branch_id)

        # start new branches only from unmatched points that were selected as seeds
        for point_index, point in enumerate(current_points):
            if point_index in matched_point_ids:
                continue
            if not bool(seed_mask[point_index]):
                continue

            branch = SpectraBranch(
                points=[
                    {
                        "x": float(case["x"]),
                        "freq": float(case["freq"]),
                        "beta": float(case["beta"]),
                        "alpha_r": float(point[0]),
                        "alpha_i": float(point[1]),
                        "isolation_score": float(case["isolation_score_filtered"][point_index]),
                        "gap_score": float(case["gap_score_filtered"][point_index]),
                    }
                ]
            )
            branches.append(branch)
            next_active_branch_ids.append(len(branches) - 1)

        # replace the active set with matched and newly born branches only
        active_branch_ids = next_active_branch_ids

    # keep only branches that persist for the requested minimum number of stations
    filtered_branches = [branch for branch in branches if len(branch.points) >= min_points]

    # sort branches for deterministic output ordering
    filtered_branches.sort(
        key=lambda branch: (
            branch.points[0]["x"],
            branch.points[0]["alpha_r"],
            branch.points[0]["alpha_i"],
        )
    )

    return filtered_branches


def _compute_branch_jump_metric(branch: SpectraBranch) -> float:
    """Compute the mean normalized alpha-space jump along one branch."""

    # return a perfect smoothness metric for single-point branches
    if len(branch.points) < 2:
        return 0.0

    jump_values: list[float] = []
    for point_left, point_right in zip(branch.points[:-1], branch.points[1:]):
        alpha_r_ref = max(abs(float(point_left["alpha_r"])), abs(float(point_right["alpha_r"])), 1.0)
        alpha_i_ref = max(abs(float(point_left["alpha_i"])), abs(float(point_right["alpha_i"])), 1.0)

        alpha_r_jump = abs(float(point_right["alpha_r"]) - float(point_left["alpha_r"])) / alpha_r_ref
        alpha_i_jump = abs(float(point_right["alpha_i"]) - float(point_left["alpha_i"])) / alpha_i_ref
        jump_values.append(float(np.hypot(alpha_r_jump, alpha_i_jump)))

    return float(np.mean(jump_values)) if jump_values else 0.0


def _find_persistent_true_runs(mask: np.ndarray, *, min_run: int) -> list[tuple[int, int]]:
    """Return inclusive index spans for persistent True runs."""

    run_spans: list[tuple[int, int]] = []
    run_start: int | None = None

    for index, flag in enumerate(mask.tolist()):
        if flag and run_start is None:
            run_start = index
            continue

        if flag:
            continue

        if run_start is None:
            continue

        run_end = index - 1
        run_length = run_end - run_start + 1
        if run_length >= min_run:
            run_spans.append((run_start, run_end))
        run_start = None

    if run_start is not None:
        run_end = len(mask) - 1
        run_length = run_end - run_start + 1
        if run_length >= min_run:
            run_spans.append((run_start, run_end))

    return run_spans


def _compute_longest_true_run(mask: np.ndarray) -> int:
    """Compute the longest consecutive True run length."""

    longest_run = 0
    current_run = 0

    for flag in mask.tolist():
        if flag:
            current_run += 1
            longest_run = max(longest_run, current_run)
            continue

        current_run = 0

    return longest_run


def _score_spectra_branches(
    cases: list[dict[str, Any]],
    branches: list[SpectraBranch],
) -> list[SpectraBranch]:
    """Assign branch-level detachment scores and sort branches from strongest to weakest."""

    # return early when there are no tracked branches to score
    if not branches:
        return []

    # build group-wide reference scales for coverage metrics
    x_values = sorted({float(case["x"]) for case in cases})
    n_x_total = max(len(x_values), 1)

    # compute branch-level metrics from the tracked point histories
    for branch in branches:
        isolation_values = np.asarray([float(point["isolation_score"]) for point in branch.points], dtype=float)
        gap_values = np.asarray([float(point["gap_score"]) for point in branch.points], dtype=float)
        x_history = np.asarray([float(point["x"]) for point in branch.points], dtype=float)

        coverage_fraction = len(branch.points) / n_x_total
        isolation_median = float(np.median(isolation_values)) if isolation_values.size else 0.0
        isolation_max = float(np.max(isolation_values)) if isolation_values.size else 0.0
        mean_jump = _compute_branch_jump_metric(branch)
        smoothness_factor = 1.0 / (1.0 + mean_jump)

        gap_median = float(np.median(gap_values)) if gap_values.size else 0.0
        gap_max = float(np.max(gap_values)) if gap_values.size else 0.0
        detached_mask = gap_values >= DETACHMENT_GAP_THRESHOLD
        detached_fraction = float(np.mean(detached_mask.astype(float))) if detached_mask.size else 0.0
        longest_detached_run = _compute_longest_true_run(detached_mask)
        detached_runs = _find_persistent_true_runs(
            detached_mask,
            min_run=DETACHMENT_MIN_RUN_POINTS,
        )

        birth_x: float | None = None
        absorption_x: float | None = None

        # identify the first persistent detached interval as a birth event
        if detached_runs:
            first_detached_start, first_detached_end = detached_runs[0]
            birth_x = float(x_history[first_detached_start])

            attached_mask = np.logical_not(detached_mask[first_detached_end + 1 :])
            attached_runs = _find_persistent_true_runs(
                attached_mask,
                min_run=DETACHMENT_MIN_RUN_POINTS,
            )
            if attached_runs:
                attached_start, _ = attached_runs[0]
                absorption_x = float(x_history[first_detached_end + 1 + attached_start])

        if detached_mask.size > 0 and np.any(detached_mask):
            detached_gap_values = gap_values[detached_mask]
            detached_gap_median = float(np.median(detached_gap_values))
        else:
            detached_gap_median = 0.0

        branch.coverage_fraction = coverage_fraction
        branch.isolation_median = isolation_median
        branch.isolation_max = isolation_max
        branch.smoothness_factor = smoothness_factor
        branch.alpha_i_penalty = 1.0
        branch.alpha_r_penalty = 1.0
        branch.start_neutral_factor = 1.0
        branch.amplification_bonus = 1.0
        branch.gap_median = gap_median
        branch.gap_max = gap_max
        branch.detached_fraction = detached_fraction
        branch.longest_detached_run = longest_detached_run
        branch.birth_x = birth_x
        branch.absorption_x = absorption_x
        branch.score = float(longest_detached_run) * detached_gap_median * smoothness_factor

    # sort the branches by persistent detachment first and preserve deterministic tie breaks
    return sorted(
        branches,
        key=lambda branch: (
            -int(branch.longest_detached_run),
            -float(branch.detached_fraction),
            -float(branch.gap_median),
            -float(branch.gap_max),
            -float(branch.score),
            -float(branch.coverage_fraction),
            -float(branch.isolation_max),
            float(branch.points[0]["x"]),
            float(branch.points[0]["alpha_r"]),
            float(branch.points[0]["alpha_i"]),
        ),
    )


def _write_branch_summary_file(
    out_path: Path,
    branches: list[SpectraBranch],
) -> Path:
    """Write a branch summary table sorted by branch detachment history."""

    with out_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "branch_number,score,n_points,coverage_fraction,isolation_median,isolation_max,"
            "smoothness_factor,alpha_i_penalty,alpha_r_penalty,start_neutral_factor,amplification_bonus,"
            "gap_median,gap_max,detached_fraction,longest_detached_run,birth_x,absorption_x,"
            "x_start,x_end,alpha_r_start,alpha_r_end,alpha_i_start,alpha_i_end\n"
        )

        for branch_index, branch in enumerate(branches, start=1):
            first_point = branch.points[0]
            last_point = branch.points[-1]
            birth_x_text = "" if branch.birth_x is None else f"{branch.birth_x:.8e}"
            absorption_x_text = "" if branch.absorption_x is None else f"{branch.absorption_x:.8e}"
            stream.write(
                f"{branch_index},"
                f"{branch.score:.8e},"
                f"{len(branch.points)},"
                f"{branch.coverage_fraction:.8e},"
                f"{branch.isolation_median:.8e},"
                f"{branch.isolation_max:.8e},"
                f"{branch.smoothness_factor:.8e},"
                f"{branch.alpha_i_penalty:.8e},"
                f"{branch.alpha_r_penalty:.8e},"
                f"{branch.start_neutral_factor:.8e},"
                f"{branch.amplification_bonus:.8e},"
                f"{branch.gap_median:.8e},"
                f"{branch.gap_max:.8e},"
                f"{branch.detached_fraction:.8e},"
                f"{int(branch.longest_detached_run)},"
                f"{birth_x_text},"
                f"{absorption_x_text},"
                f"{float(first_point['x']):.8e},"
                f"{float(last_point['x']):.8e},"
                f"{float(first_point['alpha_r']):.8e},"
                f"{float(last_point['alpha_r']):.8e},"
                f"{float(first_point['alpha_i']):.8e},"
                f"{float(last_point['alpha_i']):.8e}\n"
            )

    logger.info("wrote branch summary file %s with %d branch(es)", out_path, len(branches))
    return out_path


def _write_branch_file(
    out_path: Path,
    branches: list[SpectraBranch],
) -> Path:
    """Write one Tecplot file containing tracked spectra branches."""

    with out_path.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "spectra_branches"\n')
        stream.write(
            'VARIABLES = "x" "alpha_r" "alpha_i" "isolation_score" "gap_score" "branch_score" "freq" "beta" "branch_id"\n'
        )

        for branch_index, branch in enumerate(branches, start=1):
            stream.write(
                f'ZONE T="branch_{branch_index:03d}", I={len(branch.points)}, DATAPACKING=POINT\n'
            )

            for point in branch.points:
                stream.write(
                    f"{point['x']:.8e} "
                    f"{point['alpha_r']:.8e} "
                    f"{point['alpha_i']:.8e} "
                    f"{point['isolation_score']:.8e} "
                    f"{point['gap_score']:.8e} "
                    f"{branch.score:.8e} "
                    f"{point['freq']:.8e} "
                    f"{point['beta']:.8e} "
                    f"{float(branch_index):.8e}\n"
                )

    logger.info("wrote tracked branch file %s with %d branch(es)", out_path, len(branches))
    return out_path


def _write_branch_contour_file(
    out_path: Path,
    branches: list[SpectraBranch],
    *,
    title: str,
    zone_name: str,
) -> Path:
    """Write a structured Tecplot contour file over x and branch number."""

    # build the x-grid from all available branch points in this group
    x_values = sorted(
        {
            float(point["x"])
            for branch in branches
            for point in branch.points
        }
    )

    with out_path.open("w", encoding="utf-8") as stream:
        stream.write(f'TITLE = "{title}"\n')
        stream.write(
            'VARIABLES = "x" "branch_number" "alpha_r" "alpha_i" "isolation_score" "gap_score" "branch_score" "blank" "freq" "beta"\n'
        )
        stream.write(
            f'ZONE T="{zone_name}", I={len(x_values)}, J={len(branches)}, DATAPACKING=POINT\n'
        )

        # write the structured grid with x varying fastest inside each branch row
        for branch_index, branch in enumerate(branches, start=1):
            points_by_x = {
                float(point["x"]): point
                for point in branch.points
            }

            branch_freq = float(branch.points[0]["freq"])
            branch_beta = float(branch.points[0]["beta"])

            for x_value in x_values:
                point = points_by_x.get(x_value)

                if point is None:
                    alpha_r_value = BRANCH_CONTOUR_FILL_VALUE
                    alpha_i_value = BRANCH_CONTOUR_FILL_VALUE
                    isolation_value = BRANCH_CONTOUR_FILL_VALUE
                    gap_value = BRANCH_CONTOUR_FILL_VALUE
                    branch_score_value = float(branch.score)
                    blank_value = 1.0
                else:
                    alpha_r_value = float(point["alpha_r"])
                    alpha_i_value = float(point["alpha_i"])
                    isolation_value = float(point["isolation_score"])
                    gap_value = float(point["gap_score"])
                    branch_score_value = float(branch.score)
                    blank_value = 0.0

                stream.write(
                    f"{x_value:.8e} "
                    f"{float(branch_index):.8e} "
                    f"{alpha_r_value:.8e} "
                    f"{alpha_i_value:.8e} "
                    f"{isolation_value:.8e} "
                    f"{gap_value:.8e} "
                    f"{branch_score_value:.8e} "
                    f"{blank_value:.8e} "
                    f"{branch_freq:.8e} "
                    f"{branch_beta:.8e}\n"
                )

    logger.info(
        "wrote branch contour file %s with %d branch row(s) and %d x-column(s)",
        out_path,
        len(branches),
        len(x_values),
    )
    return out_path


def _classify_branches(
    branches: list[SpectraBranch],
    *,
    isolation_threshold: float,
    min_points: int,
) -> list[SpectraBranch]:
    """Keep branches that satisfy the isolation-score criterion often enough."""

    classified_branches: list[SpectraBranch] = []

    # evaluate each tracked branch against the discrete-branch criterion
    for branch in branches:
        isolated_count = sum(
            1
            for point in branch.points
            if float(point["isolation_score"]) >= isolation_threshold
        )

        if isolated_count < min_points:
            continue

        classified_branches.append(branch)

    return classified_branches


def _emit_branch_score_status(
    reporter: Callable[[str], None] | None,
    branches: list[SpectraBranch],
) -> None:
    """Emit a short summary of the top-ranked detached branch candidates."""

    # skip status output when there are no scored branches to report
    if not branches:
        return

    _emit_status(reporter, "top detached branch candidates:")
    for branch_index, branch in enumerate(branches[:5], start=1):
        birth_text = "none" if branch.birth_x is None else f"{branch.birth_x:.5f}"
        absorption_text = "none" if branch.absorption_x is None else f"{branch.absorption_x:.5f}"
        _emit_status(
            reporter,
            (
                f"- rank {branch_index}: score = {branch.score:.3f}, "
                f"detached run = {branch.longest_detached_run}, "
                f"median gap = {branch.gap_median:.3f}, "
                f"detached fraction = {branch.detached_fraction:.3f}, "
                f"birth = {birth_text}, absorption = {absorption_text}"
            ),
        )


def _format_status_range(values: list[float], value_format: str) -> str:
    """Format a compact [min ... max] range string for status output."""

    # build a compact range string for terminal output
    range_start = format(values[0], value_format)
    range_end = format(values[-1], value_format)
    return f"[{range_start} ... {range_end}]"


def _emit_parameter_space_status(
    reporter: Callable[[str], None] | None,
    x_values: list[float],
    freq_values: list[float],
    beta_values: list[float],
) -> None:
    """Emit a compact multi-line parameter space summary."""

    # write a short parameter space block for terminal users
    _emit_status(reporter, "parameter space:")
    _emit_status(
        reporter,
        f"- {len(x_values)} x-station(s) {_format_status_range(x_values, '.5f')}",
    )
    _emit_status(
        reporter,
        (
            f"- {len(freq_values)} frequency value(s) "
            f"{_format_status_range(freq_values, '.0f')} Hz"
        ),
    )
    _emit_status(
        reporter,
        f"- {len(beta_values)} beta value(s) {_format_status_range(beta_values, '.2f')}",
    )


def _emit_alpha_gating_status(
    reporter: Callable[[str], None] | None,
    options: SpectraProcessingOptions,
) -> None:
    """Emit the configured alpha-space gating bounds."""

    # skip output when no explicit gating is active
    if not _has_alpha_gating(options):
        return

    _emit_status(reporter, "alpha-space gating:")
    _emit_status(
        reporter,
        (
                f"- alpha_r: [{options.alpr_min if options.alpr_min is not None else '-inf'} ... "
                f"{options.alpr_max if options.alpr_max is not None else 'inf'}]"
        ),
    )
    _emit_status(
        reporter,
        (
                f"- alpha_i: [{options.alpi_min if options.alpi_min is not None else '-inf'} ... "
                f"{options.alpi_max if options.alpi_max is not None else 'inf'}]"
        ),
    )


def _emit_isolation_status(
    reporter: Callable[[str], None] | None,
    options: SpectraProcessingOptions,
) -> None:
    """Emit isolation-score settings for branch analysis."""

    _emit_status(
        reporter,
        f"isolation score: {options.isolation_k}-neighbor distance in normalized alpha-space",
    )
    _emit_status(
        reporter,
        (
            "detachment score: nearest-neighbor gap normalized by nearby branch spacing "
            f"(threshold = {DETACHMENT_GAP_THRESHOLD:.2f}, min run = {DETACHMENT_MIN_RUN_POINTS})"
        ),
    )

    if options.isolation_threshold is not None:
        _emit_status(
            reporter,
            (
                f"classification criterion: isolation_score >= {options.isolation_threshold:.3f} "
                f"for at least {options.classify_min_points} point(s)"
            ),
        )


def _emit_seed_status(
    reporter: Callable[[str], None] | None,
    cases: list[dict[str, Any]],
    accepted_seed_count: int,
) -> None:
    """Emit a short summary of the accepted seed population."""

    candidate_seed_count = sum(
        int(np.count_nonzero(case["seed_candidate_mask_filtered"]))
        for case in cases
    )
    _emit_status(
        reporter,
        f"seed selection kept {accepted_seed_count}/{candidate_seed_count} candidate seed point(s)",
    )


# --------------------------------------------------
# main function to process spectra results
# --------------------------------------------------

def spectra_process(
    *,
    cfg: Mapping[str, Any] | None = None,
    reporter: Callable[[str], None] | None = None,
    do_animate: bool = True,
    do_branches: bool = True,
    do_classify: bool = False,
) -> Path:
    """
    Process LST spectra calculation results

    This function processes the output data from LST spectra calculations,
    collecting results from multiple x-location directories, analyzing
    spectral data across frequencies and wavenumbers, and preparing
    data for visualization.

    Parameters
    ----------
    cfg : Optional[Mapping[str, Any]]
        Configuration dictionary or path to config file

    Returns
    -------
    Path
        Path to the processed results file
    """

    # --------------------------------------------------
    # output for user
    # --------------------------------------------------

    logger.info("processing spectra...")

    # resolve spectra processing controls from the config
    processing_options = _resolve_spectra_processing_options(cfg)

    # validate classification settings before doing any I/O work
    if do_classify and processing_options.isolation_threshold is None:
        raise ValueError(
            "spectra classification requires processing.spectra.isolation_threshold to be set"
        )

    # resolve the working directory from the current process context
    work_dir = Path(".")

    # --------------------------------------------------
    # Step 1: Find and parse case directories
    # --------------------------------------------------

    # output for user
    logger.info("scanning for case directories...")
    case_info = _discover_case_info(work_dir)

    # output error if no case directories found
    if not case_info:

        logger.error("no case directories found matching pattern")
        _emit_status(reporter, "no spectra case directories found")
        logger.error(
            "expected pattern: x_##pt##_m_f_####pt##_khz_beta_pos####pt##"
        )
        logger.error("abort processing")
        return Path(".")

    # output for user
    logger.info("found %d case directories", len(case_info))
    _emit_status(reporter, f"found {len(case_info)} spectra case directories")

    # --------------------------------------------------
    # Step 2: Analyze parameter space for animation planning
    # --------------------------------------------------

    # collect unique values for each parameter (x, freq, beta)
    x_values = sorted(list(set(case["x"] for case in case_info)))
    freq_values = sorted(list(set(case["freq"] for case in case_info)))
    beta_values = sorted(list(set(case["beta"] for case in case_info)))

    # output for user
    logger.info("spectra parameter space:")
    logger.info(
        "  x-locations: %d values [%.3f ... %.3f]",
        len(x_values), x_values[0], x_values[-1],
    )
    logger.info(
        "  frequencies: %d values [%.0f ... %.0f Hz]",
        len(freq_values), freq_values[0], freq_values[-1],
    )
    logger.info(
        "  wavenumbers: %d values [%.2f ... %.2f]",
        len(beta_values), beta_values[0], beta_values[-1],
    )
    _emit_parameter_space_status(reporter, x_values, freq_values, beta_values)
    _emit_alpha_gating_status(reporter, processing_options)
    _emit_isolation_status(reporter, processing_options)

    # determine animation variable
    if len(x_values) > 1:
        logger.info("-> animation over x-locations (%d frames)", len(x_values))
        _emit_status(reporter, f"animation axis: x with {len(x_values)} frame(s)")
    elif len(freq_values) > 1:
        logger.info("-> animation over frequencies (%d frames)", len(freq_values))
        _emit_status(
            reporter,
            f"animation axis: frequency with {len(freq_values)} frame(s)",
        )
    elif len(beta_values) > 1:
        logger.info("-> animation over wavenumbers (%d frames)", len(beta_values))
        _emit_status(reporter, f"animation axis: beta with {len(beta_values)} frame(s)")
    else:
        logger.info("-> single case, no animation needed")
        _emit_status(reporter, "single spectra case detected")

    # --------------------------------------------------
    # Step 3: load spectra data from each case
    # --------------------------------------------------

    loaded_cases = 0
    missing_cases: list[str] = []
    total_raw_points = 0
    total_filtered_points = 0

    for case in case_info:
        # look for Eigenvalues_* files in case directory
        eigenvalue_files = list(case["path"].glob("Eigenvalues_*"))

        # take the first one if multiple found set to None if not found
        spectra_file = eigenvalue_files[0] if eigenvalue_files else None

        # if no spectra file found, log as missing case and continue to next case
        if spectra_file is None:
            missing_cases.append(case["name"])
            continue

        # load the raw eigenvalue cloud
        alpha_r, alpha_i = _load_spectra_points(spectra_file)
        alpha_r_filtered, alpha_i_filtered, n_raw, n_filtered = _apply_alpha_gating(
            alpha_r,
            alpha_i,
            processing_options,
        )

        # add spectra file metadata
        case["spectra_file"] = spectra_file
        case["alpha_r"] = alpha_r
        case["alpha_i"] = alpha_i
        case["alpha_r_filtered"] = alpha_r_filtered
        case["alpha_i_filtered"] = alpha_i_filtered

        # count points for gating summary output
        total_raw_points += n_raw
        total_filtered_points += n_filtered

        # increase loaded cases count
        loaded_cases += 1

    # output for user
    logger.info("loaded spectra data for %d/%d cases", loaded_cases, len(case_info))
    _emit_status(
        reporter,
        f"loaded spectra data for {loaded_cases}/{len(case_info)} case(s)",
    )

    # report how strongly alpha gating reduced the candidate cloud
    if _has_alpha_gating(processing_options):
        _emit_status(
            reporter,
            f"alpha gating kept {total_filtered_points}/{total_raw_points} eigenvalues",
        )

    # report missing cases
    if missing_cases:
        preview = ", ".join(missing_cases[:5])

        if len(missing_cases) > 5:
            preview += ", ..."

        logger.warning("missing spectra files for cases: %s", preview)
        _emit_status(reporter, f"missing spectra files for {len(missing_cases)} case(s)")

    # stop early if nothing can actually be combined
    if loaded_cases == 0:
        logger.warning("no spectra files were loaded successfully")
        _emit_status(reporter, "no readable spectra files were found")
        return Path(".")

    # --------------------------------------------------
    # Step 4: group loaded cases by (frequency, beta)
    # --------------------------------------------------
    grouped_cases: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for case in case_info:
        if "alpha_r" not in case or "alpha_i" not in case:
            continue

        group_key = (float(case["freq"]), float(case["beta"]))
        grouped_cases.setdefault(group_key, []).append(case)

    logger.info("grouped loaded cases into %d spectra set(s)", len(grouped_cases))
    _emit_status(reporter, f"grouped loaded cases into {len(grouped_cases)} spectra set(s)")

    # --------------------------------------------------
    # Step 5: write one Tecplot animation file per (freq, beta) pair
    # --------------------------------------------------
    output_dir = work_dir / "spectra_animation"
    branch_dir = work_dir / "spectra_branches"
    classified_dir = work_dir / "spectra_classified"

    if do_animate:
        output_dir.mkdir(parents=True, exist_ok=True)
    if do_branches:
        branch_dir.mkdir(parents=True, exist_ok=True)
    if do_classify:
        classified_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    written_branch_files: list[Path] = []
    written_branch_contour_files: list[Path] = []
    written_branch_summary_files: list[Path] = []
    written_classified_files: list[Path] = []
    written_classified_contour_files: list[Path] = []
    written_classified_summary_files: list[Path] = []
    for (freq, beta), grouped in sorted(grouped_cases.items()):
        _emit_status(
            reporter,
            (
                f"writing group: f = {freq / 1000.0:.3f} kHz, "
                f"beta = {beta:.2f}, x-stations = {len(grouped)}"
            ),
        )

        # write the raw spectra cloud animation file
        if do_animate:
            out_name = _format_group_output_name(freq, beta)
            out_path = output_dir / out_name
            written_path = _write_group_animation_file(out_path, grouped)
            written_files.append(written_path)

        # compute local spectral isolation before branch matching
        _compute_group_isolation_scores(
            grouped,
            isolation_k=processing_options.isolation_k,
        )
        _compute_group_gap_scores(
            grouped,
            neighbor_count=DETACHMENT_NEIGHBOR_COUNT,
        )

        accepted_seed_count = _select_branch_seeds(
            grouped,
            branch_gate=processing_options.branch_gate,
        )
        _emit_seed_status(reporter, grouped, accepted_seed_count)

        tracked_branches: list[SpectraBranch] = []
        if do_branches or do_classify:
            # track forward branch candidates across x for the main branch workflow
            tracked_branches = _track_spectra_branches(
                grouped,
                branch_gate=processing_options.branch_gate,
                min_points=processing_options.branch_min_points,
                direction="forward",
            )
            tracked_branches = _score_spectra_branches(grouped, tracked_branches)

            _emit_status(
                reporter,
                (
                    f"tracked {len(tracked_branches)} forward branch(es): "
                    f"f = {freq / 1000.0:.3f} kHz, beta = {beta:.2f}"
                ),
            )
            _emit_branch_score_status(reporter, tracked_branches)

        if do_branches:
            directional_branches: dict[str, list[SpectraBranch]] = {
                "forward": tracked_branches,
            }

            backward_branches = _track_spectra_branches(
                grouped,
                branch_gate=processing_options.branch_gate,
                min_points=processing_options.branch_min_points,
                direction="backward",
            )
            backward_branches = _score_spectra_branches(grouped, backward_branches)
            directional_branches["backward"] = backward_branches

            _emit_status(
                reporter,
                (
                    f"tracked {len(backward_branches)} backward branch(es): "
                    f"f = {freq / 1000.0:.3f} kHz, beta = {beta:.2f}"
                ),
            )

            for direction_name, direction_branches in directional_branches.items():
                if not direction_branches:
                    continue

                branch_name = _format_directional_branch_output_name(freq, beta, direction_name)
                branch_path = branch_dir / branch_name
                written_branch_files.append(_write_branch_file(branch_path, direction_branches))

                branch_contour_name = _format_directional_branch_contour_output_name(freq, beta, direction_name)
                branch_contour_path = branch_dir / branch_contour_name
                written_branch_contour_files.append(
                    _write_branch_contour_file(
                        branch_contour_path,
                        direction_branches,
                        title=f"spectra_branch_contours_{direction_name}",
                        zone_name=f"tracked_branches_{direction_name}",
                    )
                )

                branch_summary_name = _format_directional_branch_summary_output_name(freq, beta, direction_name)
                branch_summary_path = branch_dir / branch_summary_name
                written_branch_summary_files.append(
                    _write_branch_summary_file(branch_summary_path, direction_branches)
                )

        if do_classify:
            classified_branches = _classify_branches(
                tracked_branches,
                isolation_threshold=float(processing_options.isolation_threshold),
                min_points=processing_options.classify_min_points,
            )
            _emit_status(
                reporter,
                (
                    f"classified {len(classified_branches)} branch(es): "
                    f"f = {freq / 1000.0:.3f} kHz, beta = {beta:.2f}"
                ),
            )

            if classified_branches:
                classified_name = _format_classified_output_name(freq, beta)
                classified_path = classified_dir / classified_name
                written_classified_files.append(_write_branch_file(classified_path, classified_branches))

                classified_contour_name = _format_classified_contour_output_name(freq, beta)
                classified_contour_path = classified_dir / classified_contour_name
                written_classified_contour_files.append(
                    _write_branch_contour_file(
                        classified_contour_path,
                        classified_branches,
                        title="spectra_classified_contours",
                        zone_name="classified_branches",
                    )
                )

                classified_summary_name = _format_classified_summary_output_name(freq, beta)
                classified_summary_path = classified_dir / classified_summary_name
                written_classified_summary_files.append(
                    _write_branch_summary_file(classified_summary_path, classified_branches)
                )


    # output for user
    logger.info(
        "spectra processing complete (%d animation file(s) written)",
        len(written_files),
    )
    if do_animate:
        _emit_status(
            reporter,
            f"wrote {len(written_files)} Tecplot animation file(s) to {output_dir}",
        )
    if do_branches:
        _emit_status(
            reporter,
            f"wrote {len(written_branch_files)} tracked-branch file(s) to {branch_dir}",
        )
        _emit_status(
            reporter,
            f"wrote {len(written_branch_contour_files)} tracked-branch contour file(s) to {branch_dir}",
        )
        _emit_status(
            reporter,
            f"wrote {len(written_branch_summary_files)} tracked-branch summary file(s) to {branch_dir}",
        )
    if do_classify:
        _emit_status(
            reporter,
            f"wrote {len(written_classified_files)} classified-branch file(s) to {classified_dir}",
        )
        _emit_status(
            reporter,
            f"wrote {len(written_classified_contour_files)} classified-branch contour file(s) to {classified_dir}",
        )
        _emit_status(
            reporter,
            f"wrote {len(written_classified_summary_files)} classified-branch summary file(s) to {classified_dir}",
        )

    if do_animate:
        return output_dir
    if do_branches:
        return branch_dir
    if do_classify:
        return classified_dir
    return work_dir
