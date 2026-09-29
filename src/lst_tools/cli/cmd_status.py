"""Report local solver run status without contacting the scheduler."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from lst_tools.status import read_run_log

status_app = typer.Typer(
    help="Inspect solver logs and convergence issues in case directories.",
    no_args_is_help=True,
)


@status_app.command("tracking")
def cmd_status_tracking(
    directory: Annotated[
        Path,
        typer.Argument(help="Directory containing kc_* tracking cases."),
    ] = Path("."),
    all_issues: Annotated[
        bool,
        typer.Option(
            "--all-issues",
            help="List every convergence issue instead of the first ten.",
        ),
    ] = False,
) -> None:
    """Summarize tracking cases and list stations with convergence issues."""

    if not directory.is_dir():
        typer.echo(f"error: directory not found: {directory}", err=True)
        raise typer.Exit(1)

    case_dirs = sorted(path for path in directory.glob("kc_*") if path.is_dir())
    if not case_dirs:
        typer.echo(f"no kc_* tracking cases found in {directory}")
        return

    finished_count = 0
    incomplete_count = 0
    no_log_count = 0
    issue_count = 0

    for case_dir in case_dirs:
        log_path = case_dir / "run.log"
        if not log_path.is_file():
            no_log_count += 1
            typer.echo(f"[no log] {case_dir.name}")
            continue

        run_status = read_run_log(log_path)
        issue_count += len(run_status.issues)
        if run_status.finished:
            finished_count += 1
            label = "finished"
        else:
            incomplete_count += 1
            label = (
                "stopped after nonconvergence"
                if run_status.issues
                else "incomplete/unknown"
            )

        typer.echo(
            f"[{label}] {case_dir.name}: {len(run_status.issues)} convergence issue(s)"
        )
        displayed_issues = run_status.issues if all_issues else run_status.issues[:10]
        for issue in displayed_issues:
            x_text = f"{issue.x:.10g}" if issue.x is not None else "unknown"
            typer.echo(
                f"  station {issue.station}: x={x_text}, f={issue.frequency:g} Hz"
            )
        if not all_issues and len(run_status.issues) > len(displayed_issues):
            typer.echo("  ... use --all-issues to list every x and frequency")

    typer.echo(
        f"Summary: {len(case_dirs)} cases, {finished_count} finished, "
        f"{incomplete_count} incomplete/unknown, {no_log_count} without log, "
        f"{issue_count} convergence issue(s)"
    )
