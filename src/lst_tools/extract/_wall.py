"""Wall boundary extraction from the FE-quad mesh."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from collections import Counter
import logging

import numpy as np


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# constants
# --------------------------------------------------

# wall nodes satisfy no-slip: |u|+|v|(+|w|) ≈ 0.  After IDW reconstruction
# from cell-centred data the values aren't exactly zero, but they are much
# smaller than the freestream speed that appears on the farfield boundary.
# Threshold at this fraction of the peak boundary speed; nodes below the
# threshold are classified as wall.
WALL_VELOCITY_FRACTION = 0.05


# --------------------------------------------------
# boundary identification
# --------------------------------------------------
def build_boundary_edges(connectivity: np.ndarray) -> list[tuple[int, int]]:
    """Return all edges shared by exactly one quadrilateral element.

    Args:
        connectivity: FE quad connectivity table (E × 4, 1-based).

    Returns:
        List of boundary edges as undirected (min_node, max_node) pairs.
    """

    # count undirected edges across all quads
    edge_counter: Counter[tuple[int, int]] = Counter()
    for quad in connectivity:
        quad_nodes = quad.tolist()
        edge_list = [
            (quad_nodes[0], quad_nodes[1]),
            (quad_nodes[1], quad_nodes[2]),
            (quad_nodes[2], quad_nodes[3]),
            (quad_nodes[3], quad_nodes[0]),
        ]

        for node_a, node_b in edge_list:
            # store as undirected (sorted) pair
            edge_key = tuple(sorted((node_a, node_b)))
            edge_counter[edge_key] += 1

    # keep only edges that appear in exactly one element (boundary edges)
    boundary_edges = [edge for edge, count in edge_counter.items() if count == 1]
    return boundary_edges


def order_boundary_loop(boundary_edges: list[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Build a two-neighbor lookup for the closed outer boundary loop.

    Args:
        boundary_edges: List of boundary edges from ``build_boundary_edges``.

    Returns:
        Mapping from each boundary node to its two neighbors.

    Raises:
        ValueError: If any node does not have exactly two neighbors.
    """

    # build an adjacency list for the boundary graph
    adjacency: dict[int, list[int]] = {}
    for node_a, node_b in boundary_edges:
        adjacency.setdefault(node_a, []).append(node_b)
        adjacency.setdefault(node_b, []).append(node_a)

    # validate the expected closed-loop topology (each node has exactly two neighbors)
    for node_id, neighbors in adjacency.items():
        if len(neighbors) != 2:
            raise ValueError(
                f"Boundary node {node_id} has degree {len(neighbors)}; expected a closed loop"
            )

    # build the two-neighbor lookup from the adjacency list
    ordered_neighbors = {
        node_id: (neighbors[0], neighbors[1])
        for node_id, neighbors in adjacency.items()
    }

    return ordered_neighbors


def extract_lower_wall(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
    nodal_fields: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the body-surface wall from the boundary loop.

    Thin wrapper around ``extract_body_wall`` for backward compatibility.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quadrilateral connectivity (1-based).
        nodal_fields: Optional canonical nodal flow fields. When ``u``/``v``/``w``
            are available, the wall can be identified from the lowest-velocity
            boundary segment instead of only from geometry.

    Returns:
        Wall x and y arrays in loop order (arc from trailing edge to trailing edge).
    """
    return extract_body_wall(
        nodal_x,
        nodal_y,
        connectivity,
        nodal_fields=nodal_fields,
    )


def extract_body_wall(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
    nodal_fields: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the body-surface arc from the boundary loop.

    For 2D body cross-sections (like delta wing slices), the physical wall is the
    complete body surface — an open arc near y≈0 running from one trailing-edge
    endpoint through the nose to the other trailing-edge endpoint. The farfield
    boundary lies at large |y|.

    Strategy:
    1. Walk the single closed boundary loop.
    2. Classify each node as body-surface (|y| < tol) vs. farfield (|y| >= tol),
       where tol = 5% of (y_max - y_min).
    3. Extract the longest contiguous sub-chain of body-surface nodes.
    4. Return nodes in loop order (geometric arc order), NOT sorted by x.

    Falls back to envelope-based extraction if no body-surface sub-chain is found.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quadrilateral connectivity (1-based).
        nodal_fields: Optional canonical nodal flow fields used for a low-speed
            boundary-wall classifier when available.

    Returns:
        Wall x and y arrays in loop order (arc from trailing edge through nose
        to trailing edge).
    """

    try:
        wall_x, wall_y = _extract_body_surface_arc(
            nodal_x,
            nodal_y,
            connectivity,
            nodal_fields=nodal_fields,
        )
        if wall_x.size > 10:
            return wall_x, wall_y
        logger.debug("body-surface arc found only %d points, falling back", wall_x.size)
    except ValueError as e:
        logger.debug("body-surface extraction failed: %s, trying envelope", e)

    # fall back to lower-envelope approach
    return _extract_lower_wall_envelope(nodal_x, nodal_y, connectivity)


def _extract_body_surface_arc(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
    nodal_fields: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the body-surface arc as the longest |y|<tol sub-chain.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quadrilateral connectivity (1-based).
        nodal_fields: Optional canonical nodal flow fields used for a low-speed
            boundary-wall classifier when available.

    Returns:
        Wall x and y arrays in loop order (NOT sorted by x).

    Raises:
        ValueError: If boundary loop cannot be built or no body-surface arc found.
    """

    # build the boundary loop
    boundary_edges = build_boundary_edges(connectivity)
    boundary_loop = order_boundary_loop(boundary_edges)

    # walk the full loop to get ordered node list
    ordered_nodes = _walk_boundary_loop(boundary_loop)

    # check for a low-velocity wall segment first when reconstructed wall values exist.
    # u and v are required; w is optional (2-D flows may not carry it).
    if nodal_fields is not None and all(
        field_name in nodal_fields for field_name in ("u", "v")
    ):
        try:
            wall_x, wall_y = _extract_velocity_wall_segment(
                ordered_nodes,
                nodal_x,
                nodal_y,
                nodal_fields,
            )
            boundary_fraction = wall_x.size / max(len(ordered_nodes), 1)
            if boundary_fraction >= 0.8:
                raise ValueError(
                    "Low-speed segment spans most of the boundary loop"
                )
            if wall_x.size > 10:
                return wall_x, wall_y

            logger.debug(
                "velocity-based wall segment found only %d points, falling back to geometry",
                wall_x.size,
            )
        except ValueError as e:
            logger.debug("velocity-based wall classification failed: %s", e)

    # define body-surface tolerance: 5% of y-range
    y_min = nodal_y.min()
    y_max = nodal_y.max()
    tol = 0.05 * (y_max - y_min)

    # mark each node as body-surface vs. farfield (1-based indexing)
    is_body = [abs(nodal_y[node - 1]) < tol for node in ordered_nodes]

    # find contiguous sub-chains of body-surface nodes
    segments = _collect_boundary_segments(ordered_nodes, is_body)

    if not segments:
        raise ValueError("No body-surface boundary segment found")

    # select the longest sub-chain (the body-surface arc)
    longest_segment = max(segments, key=len)
    logger.debug("found %d body-surface segments, longest has %d nodes",
                 len(segments), len(longest_segment))

    # extract coordinates in loop order (NOT sorted by x)
    # this preserves the arc geometry: TE-lower → nose → TE-upper
    wall_nodes = np.array(longest_segment)
    wall_x = nodal_x[wall_nodes - 1]
    wall_y = nodal_y[wall_nodes - 1]

    logger.debug("body-surface arc: %d points, x in [%.4e, %.4e], y in [%.4e, %.4e]",
                 wall_x.size, wall_x.min(), wall_x.max(), wall_y.min(), wall_y.max())

    return wall_x, wall_y


def _walk_boundary_loop(
    boundary_loop: dict[int, tuple[int, int]],
) -> list[int]:
    """Walk a closed boundary loop and return node IDs in loop order."""

    start_node = next(iter(boundary_loop.keys()))
    ordered_nodes = [start_node]
    prev_node = start_node
    curr_node = boundary_loop[start_node][0]

    while curr_node != start_node:
        ordered_nodes.append(curr_node)
        neighbor_a, neighbor_b = boundary_loop[curr_node]
        next_node = neighbor_b if neighbor_a == prev_node else neighbor_a
        prev_node = curr_node
        curr_node = next_node

    return ordered_nodes


def _collect_boundary_segments(
    ordered_nodes: list[int],
    mask: list[bool] | np.ndarray,
) -> list[list[int]]:
    """Collect contiguous selected segments on a circular boundary loop."""

    segments: list[list[int]] = []
    current_segment: list[int] = []

    for index, node_id in enumerate(ordered_nodes):
        if mask[index]:
            current_segment.append(node_id)
        else:
            if current_segment:
                segments.append(current_segment)
                current_segment = []

    if current_segment:
        segments.append(current_segment)

    if len(segments) >= 2 and mask[0] and mask[-1]:
        segments[0] = segments[-1] + segments[0]
        segments.pop()

    return segments


def _extract_velocity_wall_segment(
    ordered_nodes: list[int],
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    nodal_fields: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the longest low-velocity segment from the boundary loop.

    Uses the no-slip condition: wall nodes have u = v = (w) = 0, while
    farfield boundary nodes carry the freestream speed.  After IDW
    reconstruction from cell-centred data the wall values are not exactly
    zero, but they are orders of magnitude smaller than the freestream speed
    that appears on the opposite (farfield) boundary.  The threshold is
    ``WALL_VELOCITY_FRACTION`` times the peak boundary speed.
    """

    # -- build boundary speed from whichever velocity components are available --
    boundary_indices = np.asarray(ordered_nodes, dtype=int) - 1
    velocity_components = ["u", "v"]
    if "w" in nodal_fields:
        velocity_components.append("w")
    boundary_speed = sum(
        np.abs(nodal_fields[comp][boundary_indices]) for comp in velocity_components
    )
    # cast in case `sum` returns a Python int (empty list guard)
    boundary_speed = np.asarray(boundary_speed, dtype=float)

    # -- set threshold relative to peak boundary speed (≈ freestream speed) --
    max_speed = float(boundary_speed.max())
    if max_speed < 1.0e-12:
        raise ValueError(
            "All boundary nodes have near-zero velocity — cannot distinguish wall from farfield"
        )
    tol_vel = WALL_VELOCITY_FRACTION * max_speed

    # debug output for devs
    logger.debug(
        "no-slip wall detection: max_speed=%.4e, tol_vel=%.4e (%.0f%% of max)",
        max_speed,
        tol_vel,
        WALL_VELOCITY_FRACTION * 100,
    )

    # -- classify boundary nodes and collect contiguous low-speed segments --
    is_body = (boundary_speed <= tol_vel).tolist()
    segments = _collect_boundary_segments(ordered_nodes, is_body)
    if not segments:
        raise ValueError("No low-velocity boundary segment found")

    longest_segment = max(segments, key=len)
    wall_nodes = np.asarray(longest_segment, dtype=int)
    wall_x = nodal_x[wall_nodes - 1]
    wall_y = nodal_y[wall_nodes - 1]

    logger.debug(
        "velocity wall arc: %d points, tol_vel=%.4e, x in [%.4e, %.4e], y in [%.4e, %.4e]",
        wall_x.size,
        tol_vel,
        wall_x.min(),
        wall_x.max(),
        wall_y.min(),
        wall_y.max(),
    )

    return wall_x, wall_y



def _extract_lower_wall_envelope(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the lower wall as the lower envelope of mesh cell edges.

    Finds edges on the lower boundary by checking if each quad edge has y
    values that are local minima.
    """

    # convert 1-based to 0-based indexing
    zero_conn = connectivity - 1

    # collect all unique edges with their y-coordinates
    edge_min_y: dict[tuple[int, int], float] = {}
    edge_count: dict[tuple[int, int], int] = Counter()

    for quad in zero_conn:
        edges = [
            (quad[0], quad[1]),
            (quad[1], quad[2]),
            (quad[2], quad[3]),
            (quad[3], quad[0]),
        ]
        for n0, n1 in edges:
            key = tuple(sorted((n0, n1)))
            edge_count[key] += 1
            y_avg = 0.5 * (nodal_y[n0] + nodal_y[n1])
            if key not in edge_min_y or y_avg < edge_min_y[key]:
                edge_min_y[key] = y_avg

    # keep boundary edges (shared by only one cell)
    boundary_edges = [e for e, c in edge_count.items() if c == 1]

    if not boundary_edges:
        raise ValueError("No boundary edges found in mesh")

    # find the overall y range to define "lower" region
    all_y = nodal_y
    y_mid = 0.5 * (all_y.min() + all_y.max())

    # filter to edges in the lower half
    lower_edges = []
    for n0, n1 in boundary_edges:
        y_avg = 0.5 * (nodal_y[n0] + nodal_y[n1])
        if y_avg < y_mid:
            lower_edges.append((n0, n1))

    if not lower_edges:
        raise ValueError("No lower boundary edges found")

    # collect unique nodes from lower edges
    lower_nodes = set()
    for n0, n1 in lower_edges:
        lower_nodes.add(n0)
        lower_nodes.add(n1)

    # sort by x coordinate
    sorted_nodes = sorted(lower_nodes, key=lambda n: nodal_x[n])

    wall_x = nodal_x[sorted_nodes]
    wall_y = nodal_y[sorted_nodes]

    # remove duplicate x values (keep minimum y)
    unique_x = []
    unique_y = []
    prev_x = None
    for x, y in zip(wall_x, wall_y):
        if prev_x is None or abs(x - prev_x) > 1e-12:
            unique_x.append(x)
            unique_y.append(y)
            prev_x = x
        elif y < unique_y[-1]:
            unique_y[-1] = y

    wall_x = np.array(unique_x)
    wall_y = np.array(unique_y)

    # ensure strictly increasing x (downstream direction)
    increasing_mask = np.concatenate([[True], np.diff(wall_x) > 1e-12])
    wall_x = wall_x[increasing_mask]
    wall_y = wall_y[increasing_mask]

    logger.debug("lower wall envelope: %d points, x in [%.4e, %.4e]",
                 wall_x.size, wall_x[0], wall_x[-1])

    return wall_x, wall_y
