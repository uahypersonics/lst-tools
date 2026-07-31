"""Tests for extraction CLI configuration forwarding."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import typer

from lst_tools.cli.cmd_extract import cmd_extract
from lst_tools.extract._types import SampledProfiles


def test_cmd_extract_requires_stations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit before mesh processing when no extraction stations are configured."""
    # build a valid input path with no CLI or configuration stations
    input_path = tmp_path / "slice.dat"
    input_path.write_text("fixture", encoding="utf-8")
    extract_config = SimpleNamespace(
        input_file=None,
        hdf5_out=None,
        profiles_out=None,
        wall_out=None,
        stations=None,
        x_s=None,
        x_e=None,
        d_x=None,
    )
    config = SimpleNamespace(
        extract=extract_config,
        flow_conditions=SimpleNamespace(rgas=287.15),
    )

    # execute and validate the station requirement
    with patch("lst_tools.cli.cmd_extract.read_config", return_value=config):
        with pytest.raises(typer.Exit) as exc:
            cmd_extract(input_path)

    assert exc.value.exit_code == 1
    captured = capsys.readouterr()
    assert "extraction stations required" in captured.err


def test_cmd_extract_forwards_eta_controls(tmp_path: Path) -> None:
    """Forward all configured eta controls to the profile sampler."""
    # build a minimal input path and extraction configuration
    input_path = tmp_path / "slice.dat"
    input_path.write_text("fixture", encoding="utf-8")

    extract_config = SimpleNamespace(
        input_file=None,
        hdf5_out=str(tmp_path / "baseflow.hdf5"),
        profiles_out=str(tmp_path / "profiles.dat"),
        wall_out=None,
        surface="lower",
        n_eta=81,
        eta_max=0.012,
        eta_distribution="geometric",
        eta_stretch=2.0,
        eta_wall_spacing=1.0e-6,
        stations=[0.1],
        x_s=None,
        x_e=None,
        d_x=None,
        nondimensionalize=False,
    )
    flow_config = SimpleNamespace(
        mach=None,
        gamma=1.4,
        pr=0.71,
        rgas=287.15,
        pres_0=None,
        temp_0=None,
        pres_inf=None,
        temp_inf=None,
        dens_inf=None,
        uvel_inf=None,
        visc_law=0,
    )
    config = SimpleNamespace(
        extract=extract_config,
        flow_conditions=flow_config,
        geometry=SimpleNamespace(l_ref=1.0),
    )

    dataset = SimpleNamespace(
        nodal={"x": np.array([0.0, 1.0]), "y": np.array([0.0, 0.0])},
        cell={},
        connectivity=np.zeros((1, 4), dtype=int),
    )
    mesh_sampler = SimpleNamespace(nodal_fields={})
    profiles = SampledProfiles(
        station_x=np.array([0.1]),
        station_y=np.array([-0.1]),
        station_s=np.array([0.1]),
        eta=np.array([0.0, 0.012]),
        sample_x=np.array([[0.1, 0.1]]),
        sample_y=np.array([[-0.1, -0.088]]),
        uvel=np.array([[0.0, 1.0]]),
        vvel=np.zeros((1, 2)),
        wvel=np.zeros((1, 2)),
        temp=np.ones((1, 2)),
        pres=np.ones((1, 2)),
        rho=np.ones((1, 2)),
    )

    # execute with mesh I/O and interpolation isolated
    with (
        patch("lst_tools.cli.cmd_extract.read_config", return_value=config),
        patch(
            "lst_tools.cli.cmd_extract.read_fequad_block_tecplot",
            return_value=dataset,
        ),
        patch(
            "lst_tools.cli.cmd_extract.build_quad_mesh_sampler",
            return_value=mesh_sampler,
        ),
        patch(
            "lst_tools.cli.cmd_extract.extract_lower_wall",
            return_value=(np.array([0.0, 1.0]), np.array([-0.1, -0.1])),
        ),
        patch(
            "lst_tools.cli.cmd_extract.sample_profiles",
            return_value=profiles,
        ) as sample_mock,
        patch("lst_tools.cli.cmd_extract.detect_dimensional", return_value=False),
        patch("lst_tools.cli.cmd_extract.write_profiles_tecplot"),
        patch("lst_tools.cli.cmd_extract.write_profiles_hdf5"),
    ):
        cmd_extract(input_path, surface="upper")

    # validate every profile-grid control at the CLI boundary
    sample_kwargs = sample_mock.call_args.kwargs
    assert sample_kwargs["n_eta"] == 81
    assert sample_kwargs["eta_max"] == 0.012
    assert sample_kwargs["eta_distribution"] == "geometric"
    assert sample_kwargs["eta_stretch"] == 2.0
    assert sample_kwargs["eta_wall_spacing"] == 1.0e-6
    assert sample_kwargs["target_y"] == 1.0