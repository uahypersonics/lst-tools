"""Match frequency ridges across beta values for per-mode surfaces."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from lst_tools.data_io import read_tecplot_ascii, write_tecplot_ascii

MODE_FAMILIES = {
    "nfac": ("nfac_max_mode_*", "nfac3"),
    "alpi": ("alpi_max_mode_*", "alpi"),
}


@dataclass(frozen=True)
class ModeCurve:
    """One frequency ridge from a single beta case."""

    path: Path
    beta: float
    s: np.ndarray
    frequency: np.ndarray
    value: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    value_header: str = ""


@dataclass(frozen=True)
class ModeAssignment:
    """A ridge assigned to a global mode with its matching evidence."""

    mode: int
    curve: ModeCurve
    match_cost: float | None


def load_mode_curves(root: Path, family: str) -> dict[float, list[ModeCurve]]:
    """Read all ridge files in one maximum-value family, grouped by beta."""

    folder_pattern, value_name = MODE_FAMILIES[family]
    curves_by_beta: dict[float, list[ModeCurve]] = {}
    for mode_dir in sorted(root.glob(folder_pattern)):
        if not mode_dir.is_dir():
            continue
        for file_path in sorted(mode_dir.glob("kc_*.dat")):
            tecplot = read_tecplot_ascii(file_path)
            s = np.asarray(tecplot.field("s"), dtype=float).ravel()
            frequency = np.asarray(tecplot.field("freq"), dtype=float).ravel()
            beta_values = np.asarray(tecplot.field("beta"), dtype=float).ravel()
            value = np.asarray(tecplot.field(value_name), dtype=float).ravel()
            s_header = tecplot.aliases["s"]
            beta_header = tecplot.aliases["beta"]
            value_header = tecplot.aliases[value_name]
            fields = {
                name: np.asarray(tecplot.field(name), dtype=float).ravel()
                for name in tecplot.variables
                if name not in {s_header, beta_header}
            }
            if s.size < 2 or np.any(np.diff(s) <= 0):
                raise ValueError(f"ridge s coordinates must increase in {file_path}")
            if not np.allclose(beta_values, beta_values[0]):
                raise ValueError(f"ridge beta must be constant in {file_path}")
            if not all(
                np.all(np.isfinite(field))
                for field in (s, beta_values, *fields.values())
            ):
                raise ValueError(f"ridge contains non-finite values: {file_path}")
            beta = float(beta_values[0])
            curves_by_beta.setdefault(beta, []).append(
                ModeCurve(
                    file_path,
                    beta,
                    s,
                    frequency,
                    value,
                    fields,
                    value_header,
                )
            )

    return curves_by_beta


def process_modes(
    root: Path, output_dir: Path, *, nx: int = 500, force: bool = False
) -> list[Path]:
    """Build per-mode surfaces on one uniformly spaced streamwise grid."""

    if nx < 2:
        raise ValueError("nx must be at least 2")
    root_resolved = root.resolve()
    output_resolved = output_dir.resolve()
    if root_resolved == output_resolved or root_resolved.is_relative_to(
        output_resolved
    ):
        raise ValueError("output directory cannot contain the input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; "
                "use --force to replace it"
            )

    family_assignments: dict[str, list[ModeAssignment]] = {}
    for family in MODE_FAMILIES:
        curves_by_beta = load_mode_curves(root, family)
        if curves_by_beta:
            family_assignments[family] = match_mode_curves(curves_by_beta)

    if not family_assignments:
        raise ValueError(
            f"no nfac_max_mode_* or alpi_max_mode_* curves found in {root}"
        )

    if force and output_dir.exists():
        shutil.rmtree(output_dir)

    all_curves = [
        assignment.curve
        for assignments in family_assignments.values()
        for assignment in assignments
    ]
    s_grid = np.linspace(
        min(float(curve.s[0]) for curve in all_curves),
        max(float(curve.s[-1]) for curve in all_curves),
        nx,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    report_path = output_dir / "mode_assignments.csv"
    with report_path.open("w", newline="", encoding="utf-8") as stream:
        report = csv.writer(stream)
        report.writerow(["family", "global_mode", "beta", "source", "match_cost"])
        for family, assignments in family_assignments.items():
            beta_values = sorted({assignment.curve.beta for assignment in assignments})
            field_names = list(assignments[0].curve.fields)
            value_header = assignments[0].curve.value_header
            for assignment in assignments:
                if set(assignment.curve.fields) != set(field_names):
                    raise ValueError(
                        f"ridge variables do not match in {assignment.curve.path}"
                    )
            for assignment in assignments:
                report.writerow(
                    [
                        family,
                        assignment.mode,
                        assignment.curve.beta,
                        str(assignment.curve.path.relative_to(root)),
                        "" if assignment.match_cost is None else assignment.match_cost,
                    ]
                )
            for mode in sorted({assignment.mode for assignment in assignments}):
                mode_assignments = [
                    assignment for assignment in assignments if assignment.mode == mode
                ]
                field_shape = (len(beta_values), s_grid.size)
                surface_fields = {name: np.zeros(field_shape) for name in field_names}
                valid = np.zeros(field_shape)
                for assignment in mode_assignments:
                    beta_index = beta_values.index(assignment.curve.beta)
                    field_rows = {
                        name: values[beta_index]
                        for name, values in surface_fields.items()
                    }
                    _interpolate_curve(
                        assignment.curve,
                        s_grid,
                        field_rows,
                        valid[beta_index],
                    )

                output_name = f"{family}_max_mode_{mode:03d}"
                surface_path = output_dir / f"{output_name}.dat"
                surface_variables = {
                    "s": np.broadcast_to(s_grid, field_shape),
                    "Beta": np.broadcast_to(
                        np.asarray(beta_values)[:, None], field_shape
                    ),
                }
                surface_variables.update(surface_fields)
                surface_variables["valid"] = valid
                write_tecplot_ascii(
                    surface_path,
                    surface_variables,
                    title=output_name,
                    zone=output_name,
                )
                written.append(surface_path)

                envelope_fields, envelope_betas, envelope_valid = _extract_envelope(
                    surface_fields, value_header, valid, beta_values
                )
                envelope_name = f"{family}_envelope_mode_{mode:03d}"
                envelope_path = output_dir / f"{envelope_name}.dat"
                envelope_variables = {
                    "s": s_grid,
                    "Beta": envelope_betas,
                }
                envelope_variables.update(envelope_fields)
                envelope_variables["valid"] = envelope_valid
                write_tecplot_ascii(
                    envelope_path,
                    envelope_variables,
                    title=envelope_name,
                    zone=envelope_name,
                )
                written.append(envelope_path)

    return written


def _extract_envelope(
    surface_fields: dict[str, np.ndarray],
    value_header: str,
    valid: np.ndarray,
    beta_values: list[float],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Select all fields at the largest valid value in each streamwise column."""

    values = surface_fields[value_header]
    has_support = np.any(valid > 0.0, axis=0)
    eligible_values = np.where(valid > 0.0, values, -np.inf)
    beta_indices = np.argmax(eligible_values, axis=0)
    s_indices = np.arange(values.shape[1])

    envelope_fields = {
        name: np.where(has_support, field_values[beta_indices, s_indices], 0.0)
        for name, field_values in surface_fields.items()
    }
    betas = np.asarray(beta_values)
    envelope_betas = np.where(has_support, betas[beta_indices], 0.0)
    envelope_valid = has_support.astype(float)

    return envelope_fields, envelope_betas, envelope_valid


def _interpolate_curve(
    curve: ModeCurve,
    s_grid: np.ndarray,
    field_rows: dict[str, np.ndarray],
    valid: np.ndarray,
) -> None:
    """Interpolate every field within contiguous measured ridge segments."""

    spacing = np.diff(curve.s)
    breaks = np.flatnonzero(spacing > 2.5 * np.median(spacing)) + 1
    for segment in np.split(np.arange(curve.s.size), breaks):
        if segment.size < 2:
            continue
        s_segment = curve.s[segment]
        within = (s_grid >= s_segment[0]) & (s_grid <= s_segment[-1])
        for name, source_values in curve.fields.items():
            field_rows[name][within] = np.interp(
                s_grid[within], s_segment, source_values[segment]
            )
        valid[within] = 1.0


def match_mode_curves(
    curves_by_beta: dict[float, list[ModeCurve]],
    *,
    max_relative_frequency_difference: float = 0.2,
    ambiguity_margin: float = 0.03,
) -> list[ModeAssignment]:
    """Track curves between neighboring beta cases without trusting local mode labels."""

    assignments: list[ModeAssignment] = []
    previous: list[ModeAssignment] = []
    next_mode = 1

    for beta in sorted(curves_by_beta):
        current_curves = curves_by_beta[beta]
        scores = np.full((len(previous), len(current_curves)), np.inf)
        for row, previous_assignment in enumerate(previous):
            for col, curve in enumerate(current_curves):
                scores[row, col] = _frequency_distance(previous_assignment.curve, curve)

        matches: dict[int, tuple[int, float]] = {}
        if np.any(np.isfinite(scores)):
            eligible = np.where(np.isfinite(scores), scores, 1e6)
            for row, col in zip(*linear_sum_assignment(eligible)):
                cost = scores[row, col]
                other_scores = [
                    scores[other_row, col]
                    for other_row in range(len(previous))
                    if other_row != row
                ] + [
                    scores[row, other_col]
                    for other_col in range(len(current_curves))
                    if other_col != col
                ]
                closest_alternative = min(other_scores, default=np.inf)
                if (
                    cost <= max_relative_frequency_difference
                    and closest_alternative - cost >= ambiguity_margin
                ):
                    matches[col] = (previous[row].mode, float(cost))

        current: list[ModeAssignment] = []
        for index, curve in enumerate(current_curves):
            if index in matches:
                mode, cost = matches[index]
            else:
                mode, cost = next_mode, None
                next_mode += 1
            assignment = ModeAssignment(mode=mode, curve=curve, match_cost=cost)
            current.append(assignment)
            assignments.append(assignment)
        previous = current

    return assignments


def _frequency_distance(previous: ModeCurve, current: ModeCurve) -> float:
    """Measure relative frequency separation only where the ridges overlap in s."""

    overlap_start = max(float(previous.s[0]), float(current.s[0]))
    overlap_end = min(float(previous.s[-1]), float(current.s[-1]))
    sample_s = current.s[(current.s >= overlap_start) & (current.s <= overlap_end)]
    if sample_s.size < 5 or sample_s.size < 0.25 * min(previous.s.size, current.s.size):
        return np.inf

    previous_frequency = np.interp(sample_s, previous.s, previous.frequency)
    current_frequency = np.interp(sample_s, current.s, current.frequency)
    scale = np.maximum(
        np.maximum(np.abs(previous_frequency), np.abs(current_frequency)), 1000.0
    )
    return float(np.median(np.abs(previous_frequency - current_frequency) / scale))
