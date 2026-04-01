"""Data-driven batch-script rendering.

One SLURM renderer, one PBS renderer, one dispatcher.  Host-specific
sections (module loads, ``--mem-per-cpu``, etc.) are driven entirely by
the fields on :class:`~._resolve.ResolvedJob` — no per-host ``if``
branches.
"""

from __future__ import annotations

import os

from ._parsers import coerce_time_to_hms
from ._profiles import Scheduler
from ._resolve import ResolvedJob


# ------------------------------------------------------------------
# SLURM
# ------------------------------------------------------------------
def render_slurm(
    job: ResolvedJob,
    *,
    lst_exe: str = "lst.x",
    args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Render a SLURM batch script from *job*."""
    if job.nodes is None or job.ntasks_per_node is None:
        raise ValueError(
            "ResolvedJob.nodes and .ntasks_per_node must be set for SLURM"
        )

    wall = coerce_time_to_hms(job.time)
    total_ranks = job.nodes * job.ntasks_per_node

    lines = [
        "#!/bin/sh -l",
        "#",
        f"#SBATCH -J {job.job_name or 'lst'}",
        f"#SBATCH --nodes={job.nodes}",
        f"#SBATCH --ntasks-per-node={job.ntasks_per_node}",
        f"#SBATCH --time={wall}",
    ]

    if job.mem_per_cpu:
        lines.append(f"#SBATCH --mem-per-cpu={job.mem_per_cpu}")
    if job.account:
        lines.append(f"#SBATCH --account={job.account}")
    if job.partition:
        lines.append(f"#SBATCH --partition={job.partition}")
    if job.qos and job.qos != "standard":
        lines.append(f"#SBATCH --qos={job.qos}")
    if job.constraint:
        lines.append(f"#SBATCH --constraint={job.constraint}")
    if job.extra_sbatch:
        lines.extend(job.extra_sbatch)

    # working directory
    workdir = job.workdir or "${SLURM_SUBMIT_DIR}"
    lines.append(f"cd {workdir}")

    # module loads (profile-driven — may be empty)
    if job.modules:
        lines.append("")
        lines.append('echo "=== Loaded modules (before) ==="')
        lines.append("module list")
        lines.append('echo "====================="')
        for mod in job.modules:
            lines.append(f"module load {mod}")
        lines.append('echo "=== Loaded modules ==="')
        lines.append("module list")
        lines.append('echo "====================="')

    # standard env
    lines.extend(
        [
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "",
        ]
    )

    # extra env
    if extra_env:
        for k, v in extra_env.items():
            lines.append(f'export {k}="{v}"')
        lines.append("")

    # launch command
    launcher = job.launcher or "mpirun"
    flag = "-n" if launcher == "aprun" else "-np"

    # clean args (strip redirections — we add our own >run.log)
    clean_args: list[str] = []
    if args:
        for a in args:
            if not a.startswith(">") and not a.startswith(">>"):
                clean_args.append(a)
    else:
        clean_args = ["lst_input.dat"]

    exe_args = " ".join(clean_args)
    lines.append(f"{launcher} {flag} {total_ranks} {lst_exe} {exe_args} >run.log")
    lines.append(f"{launcher} {flag} 1 rm -f log_proc*")

    return "\n".join(lines)


# ------------------------------------------------------------------
# PBS
# ------------------------------------------------------------------
def render_pbs(
    job: ResolvedJob,
    *,
    lst_exe: str = "lst.x",
    args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Render a PBS batch script from *job*."""
    if job.nodes is None or job.ntasks_per_node is None:
        raise ValueError(
            "ResolvedJob.nodes and .ntasks_per_node must be set for PBS"
        )

    shell = os.getenv("SHELL", "/bin/bash")
    shebang = "#!/bin/csh" if shell.endswith("csh") else "#!/usr/bin/env bash"

    mpiprocs = job.ntasks_per_node
    threads = job.cpus_per_task or 1
    ncpus = mpiprocs * threads
    wall = coerce_time_to_hms(job.time)
    total_ranks = job.nodes * job.ntasks_per_node

    lines = [shebang]

    if job.account:
        lines.append(f"#PBS -A {job.account}")
    if job.partition:
        lines.append(f"#PBS -q {job.partition}")
    lines.append(f"#PBS -l select={job.nodes}:ncpus={ncpus}:mpiprocs={mpiprocs}")
    lines.append(f"#PBS -l walltime={wall}")

    if job.job_name:
        lines.append(f"#PBS -N {job.job_name}")
    if job.output:
        lines.append(f"#PBS -o {job.output}")
    lines.append(f"#PBS -S {shell}")
    lines.append("#PBS -V")
    lines.append("#PBS -j oe")

    lines.append(f"cd {job.workdir or '$PBS_O_WORKDIR'}")

    # module loads (profile-driven)
    if job.modules:
        lines.append("")
        for mod in job.modules:
            if shell.endswith("csh"):
                lines.append(f"module load {mod}")
            else:
                lines.append(f"module load {mod}")
        lines.append("")

    # profile-provided setup commands
    if job.extra_pbs:
        for line in job.extra_pbs:
            lines.append(line)
        lines.append("")

    # extra env
    if extra_env:
        for k, v in extra_env.items():
            if shell.endswith("csh"):
                lines.append(f'setenv {k} "{v}"')
            else:
                lines.append(f'export {k}="{v}"')

    # launch command
    launcher = job.launcher or "mpirun"
    flag = "-n" if launcher == "aprun" else "-np"

    cmd = f"{launcher} {flag} {total_ranks} {lst_exe}"
    for arg in args or []:
        cmd += f" {arg}"

    lines.extend(["", cmd, ""])
    lines.append(f"{launcher} {flag} 1 rm -f log_proc*")

    return "\n".join(lines)


# ------------------------------------------------------------------
# dispatcher
# ------------------------------------------------------------------
def render(
    job: ResolvedJob,
    *,
    lst_exe: str = "lst.x",
    args: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    """Dispatch to the correct renderer based on ``job.scheduler``."""
    if job.scheduler == str(Scheduler.SLURM):
        return render_slurm(job, lst_exe=lst_exe, args=args, extra_env=extra_env)
    if job.scheduler == str(Scheduler.PBS):
        return render_pbs(job, lst_exe=lst_exe, args=args, extra_env=extra_env)
    raise ValueError(f"Unknown scheduler '{job.scheduler}'; cannot render script.")
