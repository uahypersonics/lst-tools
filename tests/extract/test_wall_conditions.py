"""Tests for the wall conditions applied to nodal fields reconstructed from cell-centred data."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest

from lst_tools.extract._mesh import build_quad_mesh_sampler
from lst_tools.extract._profile import _sample_one_station
from lst_tools.extract._wall_state import apply_wall_conditions

# --------------------------------------------------
# test case: boundary-layer-like mesh over a flat wall at y = 0
# --------------------------------------------------
RGAS = 287.15
TEMP_WALL = 300.0
TEMP_GRADIENT = -400.0  # dT/dy [K/m]
VELOCITY_GRADIENT = 4000.0  # du/dy [1/s]
PRESSURE = 1000.0

# stretched wall-normal rows, as in a boundary-layer mesh; uniform columns
X_NODES = np.linspace(0.0, 1.0, 6)
Y_NODES = np.array([0.0, 0.01, 0.025, 0.05, 0.09, 0.15, 0.25])


def build_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodes and 1-based counter-clockwise quad connectivity of the structured mesh."""

    # nodes, row by row (node index = j * nx + i)
    n_x = X_NODES.size
    grid_x, grid_y = np.meshgrid(X_NODES, Y_NODES)
    nodal_x = grid_x.ravel()
    nodal_y = grid_y.ravel()

    # one quad per (i, j) cell, corners counter-clockwise, 1-based
    cells = []
    for j in range(Y_NODES.size - 1):
        for i in range(n_x - 1):
            lower_left = j * n_x + i
            cells.append([lower_left, lower_left + 1, lower_left + 1 + n_x, lower_left + n_x])
    connectivity = np.asarray(cells, dtype=int) + 1

    return nodal_x, nodal_y, connectivity


def linear_cell_fields(nodal_x: np.ndarray, nodal_y: np.ndarray, connectivity: np.ndarray) -> dict:
    """Cell-centred fields with linear u(y) and T(y), constant p, and rho from the gas law."""

    # cell centroids
    corners = connectivity - 1
    cell_y = nodal_y[corners].mean(axis=1)

    # linear profiles: u = 0 and T = TEMP_WALL at the wall
    temp = TEMP_WALL + TEMP_GRADIENT * cell_y
    fields = {
        "u": VELOCITY_GRADIENT * cell_y,
        "v": np.zeros_like(cell_y),
        "w": np.zeros_like(cell_y),
        "t": temp,
        "p": np.full_like(cell_y, PRESSURE),
        "rho": PRESSURE / (RGAS * temp),
    }
    return fields


@pytest.fixture
def mesh():
    """Mesh, cell fields, reconstructed sampler and wall coordinates."""

    # build the mesh and reconstruct nodal fields from the cell data
    nodal_x, nodal_y, connectivity = build_mesh()
    cell_fields = linear_cell_fields(nodal_x, nodal_y, connectivity)
    sampler = build_quad_mesh_sampler(nodal_x, nodal_y, connectivity, cell_fields)

    # the wall is the bottom row of nodes, in arc order
    wall_x = X_NODES.copy()
    wall_y = np.zeros_like(X_NODES)

    return sampler, cell_fields, wall_x, wall_y


# --------------------------------------------------
# the problem: reconstruction alone loses the wall state
# --------------------------------------------------
def test_reconstruction_alone_misses_the_wall_state(mesh) -> None:
    """Averaging cell centres gives first-cell values at the wall nodes, not the wall state."""

    # reconstructed wall-node values
    sampler, _, _, _ = mesh
    wall_nodes = np.arange(X_NODES.size)

    # check: flagged as reconstructed, and the wall velocity and temperature are off
    assert sampler.reconstructed_from_cells is True
    assert np.all(sampler.nodal_fields["u"][wall_nodes] > 0.0)
    assert np.all(np.abs(sampler.nodal_fields["t"][wall_nodes] - TEMP_WALL) > 1.0)


# --------------------------------------------------
# the fix
# --------------------------------------------------
def test_wall_nodes_get_the_wall_state(mesh) -> None:
    """No-slip, the exact wall temperature (linear T), wall pressure, and gas-law density."""

    # apply the wall conditions
    sampler, cell_fields, wall_x, wall_y = mesh
    corrected = apply_wall_conditions(sampler, cell_fields, wall_x, wall_y, RGAS)
    wall_nodes = np.arange(X_NODES.size)
    fields = corrected.nodal_fields

    # check every wall node, including the two corners
    np.testing.assert_array_equal(fields["u"][wall_nodes], 0.0)
    np.testing.assert_array_equal(fields["v"][wall_nodes], 0.0)
    np.testing.assert_array_equal(fields["w"][wall_nodes], 0.0)
    np.testing.assert_allclose(fields["t"][wall_nodes], TEMP_WALL, rtol=1e-12)
    np.testing.assert_allclose(fields["p"][wall_nodes], PRESSURE, rtol=1e-12)
    np.testing.assert_allclose(fields["rho"][wall_nodes], PRESSURE / (RGAS * TEMP_WALL), rtol=1e-12)


def test_only_wall_nodes_change_and_input_is_untouched(mesh) -> None:
    """Interior nodes keep their values, and the original sampler isn't modified."""

    # apply the wall conditions
    sampler, cell_fields, wall_x, wall_y = mesh
    original_t = sampler.nodal_fields["t"].copy()
    corrected = apply_wall_conditions(sampler, cell_fields, wall_x, wall_y, RGAS)
    interior = np.arange(X_NODES.size, X_NODES.size * Y_NODES.size)

    # check
    np.testing.assert_array_equal(corrected.nodal_fields["t"][interior], original_t[interior])
    np.testing.assert_array_equal(sampler.nodal_fields["t"], original_t)


def test_adiabatic_wall_keeps_the_first_cell_temperature() -> None:
    """With zero wall-normal temperature gradient the wall temperature equals the cell value."""

    # uniform temperature: an adiabatic wall in the limit
    nodal_x, nodal_y, connectivity = build_mesh()
    cell_fields = linear_cell_fields(nodal_x, nodal_y, connectivity)
    cell_fields["t"] = np.full_like(cell_fields["t"], 250.0)
    sampler = build_quad_mesh_sampler(nodal_x, nodal_y, connectivity, cell_fields)

    # apply
    corrected = apply_wall_conditions(
        sampler, cell_fields, X_NODES.copy(), np.zeros_like(X_NODES), RGAS
    )

    # check
    np.testing.assert_allclose(corrected.nodal_fields["t"][: X_NODES.size], 250.0, rtol=1e-12)


def sample_middle_station(sampler, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Velocity and temperature profiles at x = 0.5 along the wall normal."""

    # sample one station in the middle of the wall
    uvel, _, _, temp, _, _ = _sample_one_station(
        x0=0.5,
        y0=0.0,
        normal_x=0.0,
        normal_y=1.0,
        eta=eta,
        mesh_sampler=sampler,
        rgas=RGAS,
    )
    return uvel, temp


def test_sampled_profile_has_no_kink_at_the_wall(mesh) -> None:
    """Inside the first cell the velocity rises linearly from exactly zero at the wall.

    Without the fix the wall nodes carry the first-cell velocity, so the profile jumps
    from the enforced u = 0 at eta = 0 to almost the first-cell value at the next point.
    """

    # sample before and after applying the wall conditions (first cell is 0 < eta < 0.01)
    sampler, cell_fields, wall_x, wall_y = mesh
    corrected = apply_wall_conditions(sampler, cell_fields, wall_x, wall_y, RGAS)
    eta = np.array([0.0, 0.0025, 0.005, 0.0075])
    uvel_before, _ = sample_middle_station(sampler, eta)
    uvel_after, temp_after = sample_middle_station(corrected, eta)

    # before: a kink, the first step is far steeper than the rest of the cell
    first_slope_before = uvel_before[1] / eta[1]
    later_slope_before = (uvel_before[3] - uvel_before[1]) / (eta[3] - eta[1])
    assert first_slope_before > 3.0 * later_slope_before

    # after: linear through the origin (constant u / eta), and the exact wall temperature
    slopes_after = uvel_after[1:] / eta[1:]
    np.testing.assert_allclose(slopes_after, slopes_after[0], rtol=1e-10)
    assert temp_after[0] == pytest.approx(TEMP_WALL, rel=1e-12)


def test_wall_point_not_on_a_node_rejected(mesh) -> None:
    """Wall coordinates must be mesh nodes."""

    # shift one wall point off its node
    sampler, cell_fields, wall_x, wall_y = mesh
    wall_x = wall_x.copy()
    wall_x[2] += 1.0e-3

    with pytest.raises(ValueError, match="does not coincide with a mesh node"):
        apply_wall_conditions(sampler, cell_fields, wall_x, wall_y, RGAS)


def test_nodal_input_is_not_flagged() -> None:
    """Nodal input (e.g. CFD++) is used as is and not flagged for the wall correction."""

    # nodal fields supplied directly
    nodal_x, nodal_y, connectivity = build_mesh()
    n_nodes = nodal_x.size
    nodal_fields = {name: np.ones(n_nodes) for name in ("u", "v", "w", "t", "p", "rho")}
    sampler = build_quad_mesh_sampler(
        nodal_x, nodal_y, connectivity, cell_fields={}, existing_nodal_fields=nodal_fields
    )

    # check
    assert sampler.reconstructed_from_cells is False
