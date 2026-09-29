"""Read legacy LST run logs without modifying job artifacts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_STATION = re.compile(
    rf"Initialization Stage,\s*Station LST#:\s*(\d+)\s*/\s*\d+\s*\(x\s*=\s*({_NUMBER})\s*\)"
)
_NONCONVERGENCE = re.compile(
    rf"Method did not converge.*?freq\.\s*({_NUMBER}).*?x-station\s*(\d+)",
    re.IGNORECASE,
)
_TOTAL_TIME = re.compile(r"\bTotal Time\s+" + _NUMBER + r"\s*sec\b", re.IGNORECASE)


@dataclass(frozen=True)
class ConvergenceIssue:
    """A solver non-convergence and its physical station, when known."""

    station: int
    frequency: float
    x: float | None


@dataclass(frozen=True)
class RunLogStatus:
    """Completion evidence and convergence issues found in one solver log."""

    finished: bool
    issues: list[ConvergenceIssue]


def read_run_log(log_path: Path) -> RunLogStatus:
    """Read solver completion evidence and match warnings to logged x-coordinates."""

    station_x: dict[int, float] = {}
    issues: list[ConvergenceIssue] = []
    finished = False

    with log_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if _TOTAL_TIME.search(line):
                finished = True
            progress = _STATION.search(line)
            if progress:
                station_x[int(progress.group(1))] = float(
                    progress.group(2).replace("D", "E")
                )

            failure = _NONCONVERGENCE.search(line)
            if failure:
                station = int(failure.group(2))
                frequency = float(failure.group(1).replace("D", "E"))
                issues.append(
                    ConvergenceIssue(
                        station=station, frequency=frequency, x=station_x.get(station)
                    )
                )

    return RunLogStatus(finished=finished, issues=issues)
