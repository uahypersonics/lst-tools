"""Tests for lst_tools.hpc._templates — hand-built ResolvedJob → script text."""

from __future__ import annotations

import pytest

from lst_tools.hpc._resolve import ResolvedJob
from lst_tools.hpc._templates import render, render_pbs, render_slurm


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _slurm_job(**kw) -> ResolvedJob:
    defaults = dict(
        nodes=2,
        ntasks_per_node=94,
        time=1.0,
        partition="standard",
        account="myacct",
        scheduler="slurm",
        launcher="mpirun",
        hostname="generic",
        fname_run_script="run.slurm.generic",
    )
    defaults.update(kw)
    return ResolvedJob(**defaults)


def _pbs_job(**kw) -> ResolvedJob:
    defaults = dict(
        nodes=2,
        ntasks_per_node=48,
        time=1.0,
        partition="standard",
        account="myacct",
        scheduler="pbs",
        launcher="mpirun",
        hostname="carpenter",
        fname_run_script="run.pbs.carpenter",
    )
    defaults.update(kw)
    return ResolvedJob(**defaults)


# ------------------------------------------------------------------
# SLURM rendering
# ------------------------------------------------------------------
class TestRenderSlurm:
    def test_shebang(self):
        text = render_slurm(_slurm_job())
        assert text.startswith("#!/bin/sh -l")

    def test_nodes_directive(self):
        text = render_slurm(_slurm_job(nodes=4))
        assert "#SBATCH --nodes=4" in text

    def test_ntasks_directive(self):
        text = render_slurm(_slurm_job(ntasks_per_node=94))
        assert "#SBATCH --ntasks-per-node=94" in text

    def test_time_conversion(self):
        text = render_slurm(_slurm_job(time=2.5))
        assert "#SBATCH --time=02:30:00" in text

    def test_account(self):
        text = render_slurm(_slurm_job(account="myacct"))
        assert "#SBATCH --account=myacct" in text

    def test_mem_per_cpu(self):
        text = render_slurm(_slurm_job(mem_per_cpu="1gb"))
        assert "#SBATCH --mem-per-cpu=1gb" in text

    def test_no_mem_per_cpu_when_none(self):
        text = render_slurm(_slurm_job(mem_per_cpu=None))
        assert "--mem-per-cpu" not in text

    def test_module_loads(self):
        text = render_slurm(
            _slurm_job(modules=("gnu13/13.2.0", "openmpi5/5.0.5"))
        )
        assert "module load gnu13/13.2.0" in text
        assert "module load openmpi5/5.0.5" in text

    def test_no_module_loads_when_empty(self):
        text = render_slurm(_slurm_job(modules=()))
        assert "module load" not in text

    def test_launch_command(self):
        text = render_slurm(_slurm_job(nodes=2, ntasks_per_node=94))
        assert "mpirun -np 188 lst.x" in text

    def test_cleanup_line(self):
        text = render_slurm(_slurm_job())
        assert "rm -f log_proc*" in text

    def test_extra_env(self):
        text = render_slurm(_slurm_job(), extra_env={"FOO": "bar"})
        assert 'export FOO="bar"' in text

    def test_custom_exe(self):
        text = render_slurm(_slurm_job(), lst_exe="my_solver")
        assert "my_solver" in text

    def test_qos_not_shown_when_standard(self):
        text = render_slurm(_slurm_job(qos="standard"))
        assert "--qos" not in text

    def test_qos_shown_when_not_standard(self):
        text = render_slurm(_slurm_job(qos="high_priority"))
        assert "#SBATCH --qos=high_priority" in text

    def test_raises_without_nodes(self):
        with pytest.raises(ValueError, match="nodes"):
            render_slurm(_slurm_job(nodes=None))

    def test_redirect_stripped_from_args(self):
        text = render_slurm(_slurm_job(), args=["input.dat", ">run.log"])
        # should not have double >run.log
        assert text.count(">run.log") == 1

    def test_aprun_flag(self):
        text = render_slurm(_slurm_job(launcher="aprun"))
        assert "aprun -n" in text


# ------------------------------------------------------------------
# PBS rendering
# ------------------------------------------------------------------
class TestRenderPbs:
    def test_account_directive(self):
        text = render_pbs(_pbs_job())
        assert "#PBS -A myacct" in text

    def test_select_line(self):
        text = render_pbs(_pbs_job(nodes=3, ntasks_per_node=48))
        assert "#PBS -l select=3:ncpus=48:mpiprocs=48" in text

    def test_walltime(self):
        text = render_pbs(_pbs_job(time=0.5))
        assert "#PBS -l walltime=00:30:00" in text

    def test_job_name(self):
        text = render_pbs(_pbs_job(job_name="myrun"))
        assert "#PBS -N myrun" in text

    def test_join_oe(self):
        text = render_pbs(_pbs_job())
        assert "#PBS -j oe" in text

    def test_launch_command(self):
        text = render_pbs(_pbs_job(nodes=2, ntasks_per_node=48, launcher="aprun"))
        assert "aprun -n 96" in text

    def test_raises_without_nodes(self):
        with pytest.raises(ValueError, match="nodes"):
            render_pbs(_pbs_job(nodes=None))


# ------------------------------------------------------------------
# dispatcher
# ------------------------------------------------------------------
class TestRender:
    def test_dispatch_slurm(self):
        text = render(_slurm_job())
        assert "#SBATCH" in text

    def test_dispatch_pbs(self):
        text = render(_pbs_job())
        assert "#PBS" in text

    def test_unknown_raises(self):
        job = ResolvedJob(scheduler="unknown", nodes=1, ntasks_per_node=1)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            render(job)
