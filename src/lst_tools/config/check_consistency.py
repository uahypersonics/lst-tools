"""Validate configuration consistency.

Defines a registry of named checks that inspect a config dictionary
for contradictions (e.g. geometry type vs. solver switch).  Results
are returned as typed ``Issue`` objects split into errors and warnings.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable
import logging


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# helper: dotted-path access, e.g. get(cfg, "lst.solver.type")
# --------------------------------------------------
def get(d: dict[str, Any], path: str, default: Any = None) -> Any:
    """Retrieve a value from a nested dict using a dotted path.

    Parameters
    ----------
    d : dict
        Nested dictionary to search.
    path : str
        Dot-separated key path, e.g. ``"lst.solver.type"``.
    default : Any
        Value returned when any segment is missing.
    """

    cur: Any = d

    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return default
        cur = cur[seg]
    return cur


# --------------------------------------------------
# issue model
# --------------------------------------------------
class IssueLevel(Enum):
    """Severity level for a consistency issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    """Single consistency problem found in a configuration."""

    level: IssueLevel
    path: str  # dotted config path most related to the issue
    message: str  # explanation of the issue for the user
    hint: str = ""  # optional suggestion to fix the issue

    def __str__(self) -> str:
        tag = self.level.name  # "ERROR" or "WARNING"
        base = f"[{tag}] {self.path}: {self.message}"
        return f"{base}\n {self.hint}" if self.hint else base



# --------------------------------------------------
# registry of checks
# --------------------------------------------------
CheckFn = Callable[[dict[str, Any]], Iterable[Issue]]

# set up empty registry (_ prefix -> only visible within this module)
_REGISTRY: list[tuple[str, CheckFn]] = []

def register_check(name: str | None = None) -> Callable[[CheckFn], CheckFn]:
    """Decorator that adds a check function to the module registry."""
    def deco(fn: CheckFn) -> CheckFn:

        _REGISTRY.append((name or fn.__name__, fn))

        return fn

    return deco


# --------------------------------------------------
# checks (add more as needed)
# --------------------------------------------------
@register_check("geometry_switch_check")

def _check_geometry_type_vs_switch(cfg: dict[str, Any]) -> Iterable[Issue]:

    # check that geometry.type and lst.options.geometry_switch agree
    # example:
    # if geometry.type=0 (flat plate) then geometry_switch=0 (flat palte)
    # if geometry.type=2 (cone) then geometry_switch=1 (conical coordinate system)

    # set empty list of issues
    issues: list[Issue] = []

    # get geometry.type from cfg dictionary
    geometry_type = get(cfg, "geometry.type")

    # get lst.options.geometry_switch from cfg dictionary
    geometry_switch = get(cfg, "lst.options.geometry_switch")

    # check for different cases

    if geometry_type is None and geometry_switch is None:

        # case 1: geometry_type and geometry_switch are both missing (not set) -> cannot check consistency -> error

        # record issue for error/warning report
        issues.append(
            Issue(
                level=IssueLevel.ERROR,
                path="geometry.type",
                message="both geometry.type and lst.options.geometry_switch are missing; cannot check consistency.",
                hint="set geometry.type to 0 (flat plate), 1 (cylinder), 2 (cone), or 3 (generalized axisymmetric)",
            )
        )

    elif geometry_type is None and geometry_switch is not None:

        # case 2: geometry_type is missing but geometry_switch is set -> cannot check consistency -> error

        # record issue for error/warning report
        issues.append(
            Issue(
                level=IssueLevel.ERROR,
                path="geometry.type",
                message="geometry.type is missing; cannot check consistency with lst.options.geometry_switch",
                hint="set geometry.type to 0 (flat plate), 1 (cylinder), 2 (cone), or 3 (generalized axisymmetric)",
            )
        )

    elif geometry_type is not None and geometry_switch is None:

        # case 3: geometry_type is set but geometry_switch is missing -> try to set geometry_switch according to geometry_type

        # record issue for error/warning report
        issues.append(
            Issue(
                level=IssueLevel.ERROR,
                path="lst.options.geometry_switch",
                message="lst.options.geometry_switch was not set",
                hint="set lst.options.geometry_switch to 0 (flat plate), or 1 (conical coordinate system)",
            )
        )

    else:

        # case 4: both geometry_type and geometry_switch are set -> check consistency

        if geometry_type == 0 and geometry_switch != 0:

            # flat plate -> geometry_switch must be 0 (flat plate coordinate system)
            issues.append(
                Issue(
                    level=IssueLevel.ERROR,
                    path="lst.options.geometry_switch",
                    message=f"geometry.type is {geometry_type} (flat plate) but lst.options.geometry_switch is {geometry_switch}; inconsistent",
                    hint="set lst.options.geometry_switch to 0 (flat plate)",
                )
            )

        elif geometry_type == 1 and geometry_switch != 0:

            # cylinder -> geometry_switch must be 0 (no conical coordinate system)
            issues.append(
                Issue(
                    level=IssueLevel.ERROR,
                    path="lst.options.geometry_switch",
                    message=f"geometry.type is {geometry_type} (cylinder) but lst.options.geometry_switch is {geometry_switch}; inconsistent",
                    hint="set lst.options.geometry_switch to 0 (cylinder)",
                )
            )

        elif geometry_type == 2 and geometry_switch != 1:

            # cone -> geometry_switch must be 1 (conical coordinate system)
            issues.append(
                Issue(
                    level=IssueLevel.ERROR,
                    path="lst.options.geometry_switch",
                    message=f"geometry.type is {geometry_type} (cone) but lst.options.geometry_switch is {geometry_switch}; inconsistent",
                    hint="set lst.options.geometry_switch to 1 (conical coordinate system)",
                )
            )

        elif geometry_type == 3 and geometry_switch != 1:

            # generalized axisymmetric -> geometry_switch must be 1 (generalized axisymmetric coordinate system)
            issues.append(
                Issue(
                    level=IssueLevel.ERROR,
                    path="lst.options.geometry_switch",
                    message=f"geometry.type is {geometry_type} (generalized axisymmetric) but lst.options.geometry_switch is {geometry_switch}; inconsistent",
                    hint="set lst.options.geometry_switch to 1 (generalized axisymmetric)",
                )
            )



    return issues


@register_check("generalized_flag_check")
def _check_geometry_type_vs_generalized(cfg: dict[str, Any]) -> Iterable[Issue]:

    # check that geometry.type and lst.solver.generalized agree
    # flat plate (geometry.type=0) requires generalized=1 for orthogonal coordinate system
    # conical geometries should not use generalized=1

    # set empty list of issues
    issues: list[Issue] = []

    # get geometry.type from cfg dictionary
    geometry_type = get(cfg, "geometry.type")

    # get lst.solver.generalized from cfg dictionary
    generalized_flag = get(cfg, "lst.solver.generalized")

    # check for different cases

    if geometry_type is None and generalized_flag is None:

        # case 1: both are missing -> cannot check consistency (but this is not critical)
        pass

    elif geometry_type is None and generalized_flag is not None:

        # case 2: geometry_type is missing but generalized is set -> warn user
        issues.append(
            Issue(
                level=IssueLevel.WARNING,
                path="geometry.type",
                message="geometry.type is missing but lst.solver.generalized is set; cannot verify consistency",
                hint="set geometry.type to 0 (flat plate), 1 (cylinder), 2 (cone), or 3 (generalized axisymmetric)",
            )
        )

    elif geometry_type is not None and generalized_flag is None:

        # case 3: geometry_type is set but generalized is missing -> suggest appropriate value
        if geometry_type == 0:
            issues.append(
                Issue(
                    level=IssueLevel.ERROR,
                    path="lst.solver.generalized",
                    message="lst.solver.generalized is not set for flat plate geometry",
                    hint="set lst.solver.generalized to 1 (orthogonal coordinate system for flat plate)",
                )
            )

    else:

        # case 4: both are set -> check consistency

        if geometry_type == 0 and generalized_flag != 1:

            # flat plate -> generalized must be 1 (orthogonal coordinate system)
            issues.append(
                Issue(
                    level=IssueLevel.ERROR,
                    path="lst.solver.generalized",
                    message=f"geometry.type is {geometry_type} (flat plate) but lst.solver.generalized is {generalized_flag}; inconsistent",
                    hint="set lst.solver.generalized to 1 (orthogonal coordinate system for flat plate)",
                )
            )

        elif geometry_type in [2, 3] and generalized_flag == 1:

            # conical geometries (cone, generalized axisymmetric) -> generalized should be 0
            geometry_names = {2: "cone", 3: "generalized axisymmetric"}
            geometry_name = geometry_names.get(geometry_type, "unknown")
            issues.append(
                Issue(
                    level=IssueLevel.WARNING,
                    path="lst.solver.generalized",
                    message=f"geometry.type is {geometry_type} ({geometry_name}) but lst.solver.generalized is 1; may be inconsistent",
                    hint="consider setting lst.solver.generalized to 0 for conical coordinate systems",
                )
            )



    return issues


@register_check("tracking_geometry")
def _check_tracking_geometry(cfg: dict[str, Any]) -> Iterable[Issue]:
    """Validate that cone geometries have theta_deg set (required for tracking)."""
    issues: list[Issue] = []

    geometry_type = get(cfg, "geometry.type")
    theta_deg = get(cfg, "geometry.theta_deg")

    if theta_deg is None and geometry_type == 2:
        issues.append(
            Issue(
                level=IssueLevel.ERROR,
                path="geometry.theta_deg",
                message="theta_deg is required for cone geometries (geometry.type=2) but is not set",
                hint="set geometry.theta_deg in the configuration file",
            )
        )

    return issues


# --------------------------------------------------
# check consistency function: sanity checks for select config parameters
# --------------------------------------------------
def check_consistency(
    cfg: dict[str, Any],
    *,
    enabled: Iterable[str] | None = None,
) -> tuple[list[Issue], list[Issue]]:
    """Run all registered checks and return ``(errors, warnings)``.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary (or ``Config`` dataclass with ``to_dict``).
    enabled : Iterable[str] | None
        If provided, only run checks whose names are in this set.

    Returns
    -------
    tuple[list[Issue], list[Issue]]
        ``(errors, warnings)`` collected from all executed checks.
    """

    # --------------------------------------------------
    # normalise input: accept Config dataclass or plain dict
    # --------------------------------------------------

    if hasattr(cfg, "to_dict"):
        cfg = cfg.to_dict()

    # --------------------------------------------------
    # convert enabled to a set for fast lookup (if provided) -> enables selective checking
    # --------------------------------------------------

    selected = set(enabled) if enabled is not None else None

    # --------------------------------------------------
    # set up empty lists to collect errors and warnings
    # --------------------------------------------------

    # set empty lists for errors
    errors: list[Issue] = []

    # set empty lists for warnings
    warnings: list[Issue] = []

    # --------------------------------------------------
    # run all registered checks
    # --------------------------------------------------

    logger.info("selected consistency checks = %s", selected)

    for name, fn in _REGISTRY:

        # check if the slected is not None and if the name is not in selected -> skip this check
        if selected is not None and name not in selected:

            continue

        # output
        logger.info("running consistency check %s ...", name)

        # run the check function and collect issues
        for issue in fn(cfg):

            if issue.level is IssueLevel.ERROR:
                errors.append(issue)
            elif issue.level is IssueLevel.WARNING:
                warnings.append(issue)



    # return all collected issues (errors and warnings)
    return errors, warnings


# --------------------------------------------------
# print report on terminal for user
# --------------------------------------------------
def format_report(errors: list[Issue], warnings: list[Issue]) -> str:
    """Format errors and warnings into a human-readable string."""

    # set empty list of strings called lines
    lines: list[str] = []

    logger.info(
        "formatting report with %d errors and %d warnings",
        len(errors),
        len(warnings),
    )

    # add all errors to lines
    if errors:
        lines.append("Consistency errors:")
        lines.extend(str(e) for e in errors)

    # add all warnigns to lines, with a blank line before if there were errors
    if warnings:
        if lines:
            lines.append("")
        lines.append("Consistency warnings:")
        lines.extend(str(w) for w in warnings)

    # if lines is empty, return "no consistency issues found"
    # otherwise, join lines with newline characters and return the resulting string

    return "\n".join(lines) if lines else "no consistency issues found"
