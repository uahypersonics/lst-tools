"""Tests for hpc_configure thin wrapper."""

from __future__ import annotations

from unittest.mock import patch

from lst_tools.config.schema import Config
from lst_tools.hpc.configure import hpc_configure
from lst_tools.hpc._resolve import ResolvedJob


def _job() -> ResolvedJob:
    return ResolvedJob(
        nodes=1,
        ntasks_per_node=1,
        time=1.0,
        partition="standard",
        account="acct",
        scheduler="slurm",
        launcher="mpirun",
        hostname="puma",
        fname_run_script="run.slurm.puma",
    )


@patch("lst_tools.hpc.configure.resolve")
@patch("lst_tools.hpc.configure.detect")
def test_hpc_configure_with_dict_cfg(mock_detect, mock_resolve):
    mock_detect.return_value = object()
    mock_resolve.return_value = _job()

    cfg = {"hpc": {"nodes": 2, "time": 2.0}}
    out = hpc_configure(cfg, set_defaults=True, nodes_override=3, time_override=4.0)

    assert out.scheduler == "slurm"
    mock_resolve.assert_called_once_with(
        mock_detect.return_value,
        {"nodes": 2, "time": 2.0},
        set_defaults=True,
        nodes_override=3,
        time_override=4.0,
    )


@patch("lst_tools.hpc.configure.resolve")
@patch("lst_tools.hpc.configure.detect")
def test_hpc_configure_with_dataclass_cfg(mock_detect, mock_resolve):
    mock_detect.return_value = object()
    mock_resolve.return_value = _job()

    cfg = Config()
    cfg.hpc.nodes = 5

    out = hpc_configure(cfg)

    assert out.scheduler == "slurm"
    # ensure cfg.to_dict() path was used by checking dict argument
    args, kwargs = mock_resolve.call_args
    assert isinstance(args[1], dict)
    assert args[1].get("nodes") == 5
