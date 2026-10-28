"""Wall conditions for nodal fields reconstructed from cell-centred data.

Cell-centred solvers (e.g. US3D) store values at cell centres. lst-tools turns
them into nodal values by inverse-distance averaging of the adjacent cell centres
(``reconstruct_nodal_fields``). A wall node only has cell centres above it, half a
cell off the wall, so the averaged wall value is roughly the first-cell value:
no-slip and the wall temperature are lost exactly where the boundary layer starts.
Profiles then interpolate between that wrong wall value and the node above, which
puts a kink into every profile inside the first cell.

``apply_wall_conditions`` replaces the wall-node values with the wall state:

- velocity: u = v = w = 0 (no-slip)
- temperature: linear extrapolation to the wall from the first two layers of cell
  centres, each at its own wall distance. Uses only the solver's cell data (not
  the averaged node values, which are biased on stretched meshes) and is exact for
  a linear profile on any stretching, so it recovers an isothermal wall
  temperature; for an adiabatic wall (zero gradient) it gives the first-cell value
- pressure: zero wall-normal gradient (boundary-layer approximation), i.e. the
  mean of the adjacent cell centres
- density: perfect-gas law with the wall pressure and temperature

Nodal input (e.g. CFD++) already has correct wall values and is left unchanged.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import dataclasses
import logging

import numpy as np

from lst_tools.extract._types import QuadMeshSampler

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------
def apply_wall_conditions(
    mesh_sampler: QuadMeshSampler,
    cell_fields: dict[str, np.ndarray],
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    rgas: float,
) -> QuadMeshSampler:
    """Return a sampler whose wall-node values satisfy the wall conditions.

    Args:
        mesh_sampler: Sampler whose nodal fields were reconstructed from cell data.
        cell_fields: Cell-centred fields ``u``, ``v``, ``w``, ``t``, ``p`` (one value
            per cell, in connectivity order).
        wall_x: Wall node x-coordinates in arc order (as returned by the wall
            extraction; they are copies of nodal coordinates).
        wall_y: Wall node y-coordinates in arc order.
        rgas: Specific gas constant [J/(kg K)] for the wall density.

    Returns:
        New sampler with corrected nodal fields; the input sampler is not modified.

    Raises:
        ValueError: If a wall point doesn't coincide with a mesh node.
    """

    # wall nodes and their outward normals
    wall_nodes = _wall_node_indices(mesh_sampler.nodal_x, mesh_sampler.nodal_y, wall_x, wall_y)
    normal_x, normal_y = _wall_normals(wall_x, wall_y)

    # first and second layer of cells above every wall node
    first_layer, second_layer = _near_wall_cells(mesh_sampler.connectivity, wall_nodes)

    # copy the nodal fields so the input sampler stays unchanged
    nodal_fields = {name: values.copy() for name, values in mesh_sampler.nodal_fields.items()}

    # wall state, one wall node at a time
    n_fallback = 0
    for wall_index, node in enumerate(wall_nodes):
        # wall point and its normal
        node_x = mesh_sampler.nodal_x[node]
        node_y = mesh_sampler.nodal_y[node]
        nx = normal_x[wall_index]
        ny = normal_y[wall_index]

        # first-layer cell centres: mean temperature, pressure and wall distance
        cells_1 = first_layer[wall_index]
        temp_1 = float(np.mean(cell_fields["t"][cells_1]))
        pres_1 = float(np.mean(cell_fields["p"][cells_1]))
        dist_1 = _mean_wall_distance(mesh_sampler, cells_1, node_x, node_y, nx, ny)

        # default: first-cell temperature (zero gradient), used when the extrapolation
        # isn't possible (no second layer, degenerate distances) or not physical (T <= 0)
        temp_wall = temp_1
        extrapolated = False

        # second-layer cell centres, then linear extrapolation to the wall through
        # (dist_1, temp_1) and (dist_2, temp_2)
        cells_2 = second_layer[wall_index]
        if cells_2.size > 0:
            temp_2 = float(np.mean(cell_fields["t"][cells_2]))
            dist_2 = _mean_wall_distance(mesh_sampler, cells_2, node_x, node_y, nx, ny)
            if dist_2 > dist_1 > 0.0:
                slope = (temp_2 - temp_1) / (dist_2 - dist_1)
                temp_candidate = temp_1 - slope * dist_1
                if temp_candidate > 0.0:
                    temp_wall = temp_candidate
                    extrapolated = True

        # count wall nodes that kept the first-cell temperature
        if not extrapolated:
            n_fallback += 1

        # write the wall state
        nodal_fields["u"][node] = 0.0
        nodal_fields["v"][node] = 0.0
        nodal_fields["w"][node] = 0.0
        nodal_fields["t"][node] = temp_wall
        nodal_fields["p"][node] = pres_1
        nodal_fields["rho"][node] = pres_1 / (rgas * temp_wall)

    # debug output for devs
    logger.debug(
        "applied wall conditions at %d wall nodes (%d kept the first-cell temperature)",
        wall_nodes.size,
        n_fallback,
    )

    corrected = dataclasses.replace(mesh_sampler, nodal_fields=nodal_fields)
    return corrected


# --------------------------------------------------
# private helpers
# --------------------------------------------------
def _wall_node_indices(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    wall_x: np.ndarray,
    wall_y: np.ndarray,
) -> np.ndarray:
    """Map wall coordinates back to node indices (0-based) by exact coordinate match."""

    # look-up table from node coordinates to node index
    node_lookup: dict[tuple[float, float], int] = {}
    for node_index, (x, y) in enumerate(zip(nodal_x.tolist(), nodal_y.tolist(), strict=True)):
        node_lookup.setdefault((x, y), node_index)

    # find every wall point
    wall_nodes = []
    for x, y in zip(wall_x.tolist(), wall_y.tolist(), strict=True):
        if (x, y) not in node_lookup:
            raise ValueError(f"wall point ({x:.6e}, {y:.6e}) does not coincide with a mesh node")
        wall_nodes.append(node_lookup[(x, y)])

    return np.asarray(wall_nodes, dtype=int)


def _wall_normals(wall_x: np.ndarray, wall_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals of the wall arc at its nodes (orientation is irrelevant here)."""

    # tangent along the arc, then rotate by 90 degrees
    tangent_x = np.gradient(wall_x)
    tangent_y = np.gradient(wall_y)
    length = np.hypot(tangent_x, tangent_y)
    length = np.where(length > 0.0, length, 1.0)
    normal_x = -tangent_y / length
    normal_y = tangent_x / length

    return normal_x, normal_y


def _near_wall_cells(
    connectivity: np.ndarray,
    wall_nodes: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """First and second layer of cells above each wall node.

    The first layer are the cells that contain the wall node. The second layer are
    the cells that share a node with the first layer but don't touch the wall; on a
    boundary-layer mesh that is the row of cells directly above.

    Args:
        connectivity: Quad connectivity, 0-based, shape (n_cells, 4).
        wall_nodes: Wall node indices, 0-based.

    Returns:
        Per wall node: array of first-layer cell indices, array of second-layer cell indices.
    """

    # cells touching the wall, and a lookup from wall node to its position
    touches_wall = np.isin(connectivity, wall_nodes).any(axis=1)
    position_of = {node: index for index, node in enumerate(wall_nodes.tolist())}

    # first layer: cells containing each wall node
    first_layer: list[list[int]] = [[] for _ in range(wall_nodes.size)]
    for cell_index in np.flatnonzero(touches_wall).tolist():
        for node in connectivity[cell_index].tolist():
            if node in position_of:
                first_layer[position_of[node]].append(cell_index)

    # node -> cells lookup, restricted to cells that don't touch the wall
    off_wall_cells_of_node: dict[int, list[int]] = {}
    for cell_index in np.flatnonzero(~touches_wall).tolist():
        for node in connectivity[cell_index].tolist():
            off_wall_cells_of_node.setdefault(node, []).append(cell_index)

    # second layer: off-wall cells sharing a node with the first layer
    second_layer: list[np.ndarray] = []
    for cells in first_layer:
        layer_nodes = set(connectivity[cells].ravel().tolist())
        neighbours: set[int] = set()
        for node in layer_nodes:
            neighbours.update(off_wall_cells_of_node.get(node, []))
        second_layer.append(np.asarray(sorted(neighbours), dtype=int))

    first_layer_arrays = [np.asarray(cells, dtype=int) for cells in first_layer]

    return first_layer_arrays, second_layer


def _mean_wall_distance(
    mesh_sampler: QuadMeshSampler,
    cells: np.ndarray,
    node_x: float,
    node_y: float,
    normal_x: float,
    normal_y: float,
) -> float:
    """Mean distance of the cells' centroids from the wall, along the wall normal."""

    # project each centroid's offset from the wall node onto the normal
    offset_x = mesh_sampler.cell_x[cells] - node_x
    offset_y = mesh_sampler.cell_y[cells] - node_y
    distance = np.abs(offset_x * normal_x + offset_y * normal_y)

    return float(np.mean(distance))
