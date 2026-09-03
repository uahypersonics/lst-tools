"""Tests for wall-state handling during profile sampling."""

from __future__ import annotations

import numpy as np
import pytest

from lst_tools.extract._mesh import build_quad_mesh_sampler
from lst_tools.extract._profile import _sample_one_station, build_station_normals


def test_build_station_normals_intersects_exact_wall_edge() -> None:
    """A requested station should lie on its bracketing FE wall edge."""

    # build a piecewise-linear upper wall with a sloped first edge
    wall_x = np.array([0.0, 1.0, 2.0])
    wall_y = np.array([0.0, 1.0, 1.0])
    station_x = np.array([0.25])

    station_y, station_s, normal_x, normal_y, _ = build_station_normals(
        wall_x,
        wall_y,
        station_x,
        target_y=1.0,
        body_centroid=(1.0, -1.0),
    )

    expected_component = 1.0 / np.sqrt(2.0)
    assert station_y == pytest.approx([0.25])
    assert station_s == pytest.approx([0.25 * np.sqrt(2.0)])
    assert normal_x == pytest.approx([-expected_component])
    assert normal_y == pytest.approx([expected_component])


def test_sample_one_station_preserves_interpolated_wall_thermodynamics() -> None:
    """A valid wall stencil should not be replaced by the first interior state."""

    # build one unit quad with different wall and outer thermodynamic states
    nodal_x = np.array([0.0, 1.0, 1.0, 0.0])
    nodal_y = np.array([0.0, 0.0, 1.0, 1.0])
    connectivity = np.array([[1, 2, 3, 4]])
    nodal_fields = {
        "u": np.array([2.0, 2.0, 4.0, 4.0]),
        "v": np.zeros(4),
        "w": np.zeros(4),
        "t": np.array([10.0, 10.0, 20.0, 20.0]),
        "p": np.array([100.0, 100.0, 200.0, 200.0]),
        "rho": np.array([1.0, 1.0, 2.0, 2.0]),
    }
    mesh_sampler = build_quad_mesh_sampler(
        nodal_x,
        nodal_y,
        connectivity,
        cell_fields={},
        existing_nodal_fields=nodal_fields,
    )

    # sample at the wall and halfway through the cell
    eta = np.array([0.0, 0.5])
    uvel, vvel, wvel, temp, pres, rho = _sample_one_station(
        x0=0.5,
        y0=0.0,
        normal_x=0.0,
        normal_y=1.0,
        eta=eta,
        mesh_sampler=mesh_sampler,
        rgas=287.15,
    )

    # enforce no-slip velocity while retaining the interpolated wall state
    assert uvel[0] == pytest.approx(0.0)
    assert vvel[0] == pytest.approx(0.0)
    assert wvel[0] == pytest.approx(0.0)
    assert temp == pytest.approx([10.0, 15.0])
    assert pres == pytest.approx([100.0, 150.0])
    assert rho == pytest.approx([1.0, 1.5])
