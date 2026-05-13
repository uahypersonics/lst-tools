"""Core logic for extracting wall-normal profiles from a Tecplot FE-quad file.

Reads an unstructured FEQUADRILATERAL BLOCK Tecplot ASCII slice, identifies
the lower wall boundary, reconstructs nodal fields from cell-centered CFD data,
and samples wall-normal profiles via barycentric interpolation.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np


# --------------------------------------------------
# constants
# --------------------------------------------------

# gas properties for air
GAMMA = 1.4
GAS_CONSTANT = 287.05
SUTHERLAND_T0 = 273.15
SUTHERLAND_MU0 = 1.716e-5
SUTHERLAND_S = 110.4

# wall-normal grid size
N_ETA = 200

# flow field names expected in the cell-centered dataset
REQUIRED_FIELDS = ("u", "v", "w", "t", "p", "rho")


# --------------------------------------------------
# data container for unstructured dataset
# --------------------------------------------------
@dataclass(slots=True)
class TecplotUnstructuredData:
    """Container for the FE quadrilateral Tecplot dataset."""

    nodal: dict[str, np.ndarray]
    cell: dict[str, np.ndarray]
    connectivity: np.ndarray


# --------------------------------------------------
# data container for sampled profiles
# --------------------------------------------------
@dataclass(slots=True)
class SampledProfiles:
    """Container for sampled wall-normal profile data."""

    station_x: np.ndarray
    station_y: np.ndarray
    station_s: np.ndarray
    eta: np.ndarray
    sample_x: np.ndarray
    sample_y: np.ndarray
    uvel: np.ndarray
    vvel: np.ndarray
    wvel: np.ndarray
    temp: np.ndarray
    pres: np.ndarray
    rho: np.ndarray


# --------------------------------------------------
# data container for interpolation stencil
# --------------------------------------------------
@dataclass(slots=True)
class InterpolationStencil:
    """Interpolation stencil for a point inside a quadrilateral cell."""

    cell_index: int
    node_indices: tuple[int, int, int]
    weights: np.ndarray


# --------------------------------------------------
# data container for quad mesh sampler
# --------------------------------------------------
@dataclass(slots=True)
class QuadMeshSampler:
    """Geometry and lookup data needed for scalable quad sampling."""

    nodal_x: np.ndarray
    nodal_y: np.ndarray
    connectivity: np.ndarray
    cell_x: np.ndarray
    cell_y: np.ndarray
    nodal_fields: dict[str, np.ndarray]
    cell_min_x: np.ndarray
    cell_max_x: np.ndarray
    cell_min_y: np.ndarray
    cell_max_y: np.ndarray
    x_min: float
    y_min: float
    bin_size_x: float
    bin_size_y: float
    n_bin_x: int
    n_bin_y: int
    bin_to_cells: dict[tuple[int, int], list[int]]


# --------------------------------------------------
# Tecplot FE-quad reader
# --------------------------------------------------
def read_fequad_block_tecplot(path: str | Path) -> TecplotUnstructuredData:
    """Read a FEQUADRILATERAL BLOCK Tecplot ASCII file.

    The file must have:
    - Line 0: ``VARIABLES = x, y, z, ...``
    - Line 1: ``ZONE N=<nodes>, E=<elements>, ...``
    - Lines 2-3: ``DATAPACKING=BLOCK`` and ``ZONETYPE=FEQUADRILATERAL``
    - Numeric body: nodal variables (N values each), then cell-centered
      variables (E values each), then connectivity (E rows × 4 columns).

    Args:
        path: Input Tecplot ASCII file path.

    Returns:
        Parsed nodal coordinates, cell-centered fields, and connectivity.

    Raises:
        ValueError: If the zone header cannot be parsed.
    """

    # convert to Path object
    file_path = Path(path)

    # read full file text and split into lines
    text = file_path.read_text()
    lines = text.splitlines()

    # parse variable names from the first header line
    variables_text = lines[0].split("=", 1)[1]
    variable_names = [item.strip() for item in variables_text.split(",")]

    # parse node and element counts from the zone header
    size_match = re.search(r"N=(\d+),\s*E=(\d+)", lines[1])
    if size_match is None:
        raise ValueError("Could not parse N and E from Tecplot zone header")

    n_node = int(size_match.group(1))
    n_elem = int(size_match.group(2))

    # tokenize the numeric body after the 4-line header using a regex
    # that matches both fixed-point and scientific-notation numbers
    body_text = "\n".join(lines[4:])
    token_pattern = r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:[Ee][-+]?\d+)?"
    numeric_tokens = [float(token) for token in re.findall(token_pattern, body_text)]

    # split the token stream into nodal and cell-centered variable blocks
    index = 0
    nodal: dict[str, np.ndarray] = {}
    for name in variable_names[:3]:
        # first 3 variables are nodal coordinates (x, y, z)
        nodal[name] = np.asarray(numeric_tokens[index:index + n_node], dtype=float)
        index += n_node

    cell: dict[str, np.ndarray] = {}
    for name in variable_names[3:]:
        # remaining variables are cell-centered flow fields
        cell[name] = np.asarray(numeric_tokens[index:index + n_elem], dtype=float)
        index += n_elem

    # read the FE quadrilateral connectivity table (1-based node indices)
    connectivity_values = numeric_tokens[index:index + 4 * n_elem]
    connectivity = np.asarray(connectivity_values, dtype=int).reshape(n_elem, 4)

    return TecplotUnstructuredData(
        nodal=nodal,
        cell=cell,
        connectivity=connectivity,
    )


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
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the downstream lower wall from the outer boundary loop.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quadrilateral connectivity (1-based).

    Returns:
        Sorted arrays ``wall_x`` and ``wall_y`` for the lower wall.
    """

    # build the full outer boundary graph from element edges
    boundary_edges = build_boundary_edges(connectivity)
    boundary_loop = order_boundary_loop(boundary_edges)

    # find the nose node: minimum (y, -x) key anchors the leftmost/lowest point
    start_node = min(
        boundary_loop,
        key=lambda node_id: (nodal_y[node_id - 1], nodal_x[node_id - 1]),
    )
    neighbor_a, neighbor_b = boundary_loop[start_node]

    # choose the branch that stays on the lower wall and moves downstream
    start_neighbors = [neighbor_a, neighbor_b]
    next_node = min(
        start_neighbors,
        key=lambda node_id: (nodal_y[node_id - 1], -nodal_x[node_id - 1]),
    )

    # trace the boundary in the chosen direction
    wall_nodes = [start_node, next_node]
    previous_node = start_node
    current_node = next_node

    while True:
        candidate_a, candidate_b = boundary_loop[current_node]
        if candidate_a == previous_node:
            next_candidate = candidate_b
        else:
            next_candidate = candidate_a

        if next_candidate == start_node:
            break

        wall_nodes.append(next_candidate)
        previous_node = current_node
        current_node = next_candidate

    # convert node indices to physical coordinates (Tecplot is 1-based)
    traced_x = nodal_x[np.asarray(wall_nodes) - 1]
    traced_y = nodal_y[np.asarray(wall_nodes) - 1]

    # truncate when the boundary turns back upstream (dx <= 0)
    dx = np.diff(traced_x)
    non_increasing = np.where(dx <= 1.0e-12)[0]
    if non_increasing.size > 0:
        stop_index = int(non_increasing[0])
        wall_x = traced_x[:stop_index + 1]
        wall_y = traced_y[:stop_index + 1]
    else:
        wall_x = traced_x
        wall_y = traced_y

    return wall_x, wall_y


# --------------------------------------------------
# nodal field reconstruction
# --------------------------------------------------
def build_cell_centers(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cell centroids from quad connectivity.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quad connectivity (E × 4, 1-based).

    Returns:
        Cell centroid arrays ``cell_x`` and ``cell_y``.
    """

    # convert 1-based Tecplot connectivity to 0-based indexing
    node_index = connectivity - 1

    # compute centroid coordinates as the mean of the four corner nodes
    cell_x = nodal_x[node_index].mean(axis=1)
    cell_y = nodal_y[node_index].mean(axis=1)

    return cell_x, cell_y


def reconstruct_nodal_fields(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    cell_fields: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Reconstruct nodal fields from cell-centered data via inverse-distance weighting.

    For each node i, collects adjacent cells C(i) and computes the weighted average
    of cell-centered values using weights w_ij = 1 / ||x_i - x_j||.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quad connectivity (E × 4, 0-based).
        cell_x: Cell centroid x-coordinates.
        cell_y: Cell centroid y-coordinates.
        cell_fields: Cell-centered flow variable arrays.

    Returns:
        Dictionary of nodal field arrays (same keys as ``cell_fields``).
    """

    # build node-to-cell adjacency once for all fields
    n_node = nodal_x.size
    node_to_cells: list[list[int]] = [[] for _ in range(n_node)]
    for cell_index, cell_nodes in enumerate(connectivity):
        for node_index in cell_nodes:
            node_to_cells[node_index].append(cell_index)

    # reconstruct each field independently
    nodal_fields: dict[str, np.ndarray] = {}
    for field_name, cell_values in cell_fields.items():
        nodal_values = np.zeros(n_node, dtype=float)

        for node_index, cell_indices_list in enumerate(node_to_cells):
            if not cell_indices_list:
                continue

            # collect adjacent cell centroids
            cell_indices = np.asarray(cell_indices_list, dtype=int)
            dx = cell_x[cell_indices] - nodal_x[node_index]
            dy = cell_y[cell_indices] - nodal_y[node_index]
            distance = np.hypot(dx, dy)

            # inverse-distance weights; regularize with epsilon to avoid div-by-zero
            weights = 1.0 / np.maximum(distance, 1.0e-12)

            weighted_sum = float(np.sum(weights * cell_values[cell_indices]))
            weight_total = float(np.sum(weights))
            nodal_values[node_index] = weighted_sum / weight_total

        nodal_fields[field_name] = nodal_values

    return nodal_fields


# --------------------------------------------------
# quad mesh sampler
# --------------------------------------------------
def clamp_bin_index(index: int, n_bin: int) -> int:
    """Clamp a spatial-bin index to the valid range [0, n_bin - 1].

    Args:
        index: Raw bin index.
        n_bin: Total number of bins along this axis.

    Returns:
        Clamped index.
    """

    return max(0, min(index, n_bin - 1))


def build_quad_mesh_sampler(
    nodal_x: np.ndarray,
    nodal_y: np.ndarray,
    connectivity: np.ndarray,
    cell_fields: dict[str, np.ndarray],
) -> QuadMeshSampler:
    """Build a scalable quad mesh sampler with a uniform spatial bin index.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quad connectivity (E × 4, 1-based).
        cell_fields: Cell-centered flow variable arrays.

    Returns:
        Populated ``QuadMeshSampler`` ready for point queries.
    """

    # convert 1-based Tecplot connectivity to 0-based indexing
    zero_based_connectivity = connectivity - 1

    # compute cell centroids and reconstruct nodal values for interpolation
    cell_x, cell_y = build_cell_centers(nodal_x, nodal_y, connectivity)

    # only reconstruct fields required for profile extraction
    selected_fields = {field_name: cell_fields[field_name] for field_name in REQUIRED_FIELDS}
    nodal_fields = reconstruct_nodal_fields(
        nodal_x,
        nodal_y,
        zero_based_connectivity,
        cell_x,
        cell_y,
        selected_fields,
    )

    # precompute cell bounding boxes for fast point-in-cell rejection
    quad_x = nodal_x[zero_based_connectivity]
    quad_y = nodal_y[zero_based_connectivity]
    cell_min_x = quad_x.min(axis=1)
    cell_max_x = quad_x.max(axis=1)
    cell_min_y = quad_y.min(axis=1)
    cell_max_y = quad_y.max(axis=1)

    x_min = float(cell_min_x.min())
    x_max = float(cell_max_x.max())
    y_min = float(cell_min_y.min())
    y_max = float(cell_max_y.max())

    # size the spatial bin grid from the mesh size and aspect ratio
    n_cell = zero_based_connectivity.shape[0]
    extent_x = max(x_max - x_min, 1.0e-12)
    extent_y = max(y_max - y_min, 1.0e-12)
    aspect_ratio = extent_x / extent_y
    n_bin_x = max(1, int(np.ceil(np.sqrt(n_cell * aspect_ratio))))
    n_bin_y = max(1, int(np.ceil(np.sqrt(n_cell / aspect_ratio))))
    bin_size_x = extent_x / n_bin_x
    bin_size_y = extent_y / n_bin_y

    # assign each quad to every bin that its bounding box overlaps
    bin_to_cells: dict[tuple[int, int], list[int]] = {}
    for cell_index in range(n_cell):
        ix_start = clamp_bin_index(
            int(np.floor((cell_min_x[cell_index] - x_min) / bin_size_x)), n_bin_x
        )
        ix_end = clamp_bin_index(
            int(np.floor((cell_max_x[cell_index] - x_min) / bin_size_x)), n_bin_x
        )
        iy_start = clamp_bin_index(
            int(np.floor((cell_min_y[cell_index] - y_min) / bin_size_y)), n_bin_y
        )
        iy_end = clamp_bin_index(
            int(np.floor((cell_max_y[cell_index] - y_min) / bin_size_y)), n_bin_y
        )

        for ix in range(ix_start, ix_end + 1):
            for iy in range(iy_start, iy_end + 1):
                bin_to_cells.setdefault((ix, iy), []).append(cell_index)

    return QuadMeshSampler(
        nodal_x=nodal_x,
        nodal_y=nodal_y,
        connectivity=zero_based_connectivity,
        cell_x=cell_x,
        cell_y=cell_y,
        nodal_fields=nodal_fields,
        cell_min_x=cell_min_x,
        cell_max_x=cell_max_x,
        cell_min_y=cell_min_y,
        cell_max_y=cell_max_y,
        x_min=x_min,
        y_min=y_min,
        bin_size_x=bin_size_x,
        bin_size_y=bin_size_y,
        n_bin_x=n_bin_x,
        n_bin_y=n_bin_y,
        bin_to_cells=bin_to_cells,
    )


# --------------------------------------------------
# barycentric interpolation
# --------------------------------------------------
def compute_triangle_barycentric_weights(
    point_x: float,
    point_y: float,
    tri_x: np.ndarray,
    tri_y: np.ndarray,
) -> np.ndarray | None:
    """Return barycentric weights if a point lies inside a triangle.

    Args:
        point_x: Query point x-coordinate.
        point_y: Query point y-coordinate.
        tri_x: Triangle vertex x-coordinates (length 3).
        tri_y: Triangle vertex y-coordinates (length 3).

    Returns:
        Barycentric weights array of shape (3,), or ``None`` if the point
        lies outside the triangle.
    """

    # compute the scalar denominator D
    denominator = (
        (tri_y[1] - tri_y[2]) * (tri_x[0] - tri_x[2])
        + (tri_x[2] - tri_x[1]) * (tri_y[0] - tri_y[2])
    )
    if abs(denominator) <= 1.0e-14:
        # degenerate triangle
        return None

    # compute lambda_0 and lambda_1; lambda_2 = 1 - lambda_0 - lambda_1
    weight_a = (
        (tri_y[1] - tri_y[2]) * (point_x - tri_x[2])
        + (tri_x[2] - tri_x[1]) * (point_y - tri_y[2])
    ) / denominator
    weight_b = (
        (tri_y[2] - tri_y[0]) * (point_x - tri_x[2])
        + (tri_x[0] - tri_x[2]) * (point_y - tri_y[2])
    ) / denominator
    weight_c = 1.0 - weight_a - weight_b

    weights = np.asarray([weight_a, weight_b, weight_c], dtype=float)

    # reject the point if any weight is negative (outside the triangle)
    if np.min(weights) < -1.0e-10:
        return None

    return weights


def build_interpolation_stencil(
    mesh_sampler: QuadMeshSampler,
    cell_index: int,
    point_x: float,
    point_y: float,
) -> InterpolationStencil | None:
    """Build an interpolation stencil if a point lies inside a quad cell.

    Each quad is split into two triangles (0,1,2) and (0,2,3).

    Args:
        mesh_sampler: Populated quad mesh sampler.
        cell_index: Cell index to test.
        point_x: Query x-coordinate.
        point_y: Query y-coordinate.

    Returns:
        ``InterpolationStencil`` with node indices and barycentric weights,
        or ``None`` if the point lies outside this cell.
    """

    # quick bounding-box rejection before the more expensive barycentric test
    if point_x < mesh_sampler.cell_min_x[cell_index] - 1.0e-12:
        return None
    if point_x > mesh_sampler.cell_max_x[cell_index] + 1.0e-12:
        return None
    if point_y < mesh_sampler.cell_min_y[cell_index] - 1.0e-12:
        return None
    if point_y > mesh_sampler.cell_max_y[cell_index] + 1.0e-12:
        return None

    cell_nodes = mesh_sampler.connectivity[cell_index]

    # split quad into two triangles and test each
    triangle_sets = ((0, 1, 2), (0, 2, 3))
    for triangle_local in triangle_sets:
        node_indices = tuple(int(cell_nodes[local_index]) for local_index in triangle_local)
        tri_x = mesh_sampler.nodal_x[np.asarray(node_indices, dtype=int)]
        tri_y = mesh_sampler.nodal_y[np.asarray(node_indices, dtype=int)]
        weights = compute_triangle_barycentric_weights(point_x, point_y, tri_x, tri_y)

        if weights is not None:
            return InterpolationStencil(
                cell_index=cell_index,
                node_indices=node_indices,
                weights=weights,
            )

    return None


def locate_interpolation_stencil(
    mesh_sampler: QuadMeshSampler,
    point_x: float,
    point_y: float,
    previous_cell_index: int | None,
) -> InterpolationStencil | None:
    """Locate the interpolation stencil for a sample point.

    Searches in priority order:
    1. Previous cell (warm-start for successive ray points).
    2. Current spatial bin and expanding radius up to 2.
    3. Global fallback over all cells.

    Args:
        mesh_sampler: Populated quad mesh sampler.
        point_x: Query x-coordinate.
        point_y: Query y-coordinate.
        previous_cell_index: Cell index from the previous sample point, or
            ``None`` on first call.

    Returns:
        ``InterpolationStencil`` if the point is inside the mesh, else ``None``.
    """

    # warm-start: try the previous cell first since ray points are close together
    if previous_cell_index is not None:
        stencil = build_interpolation_stencil(mesh_sampler, previous_cell_index, point_x, point_y)
        if stencil is not None:
            return stencil

    # determine the primary spatial bin for this point
    bin_x = clamp_bin_index(
        int(np.floor((point_x - mesh_sampler.x_min) / mesh_sampler.bin_size_x)),
        mesh_sampler.n_bin_x,
    )
    bin_y = clamp_bin_index(
        int(np.floor((point_y - mesh_sampler.y_min) / mesh_sampler.bin_size_y)),
        mesh_sampler.n_bin_y,
    )

    checked_cells: set[int] = set()

    # search expanding rings of bins around the primary bin
    for search_radius in range(3):
        ix_min = max(0, bin_x - search_radius)
        ix_max = min(mesh_sampler.n_bin_x - 1, bin_x + search_radius)
        iy_min = max(0, bin_y - search_radius)
        iy_max = min(mesh_sampler.n_bin_y - 1, bin_y + search_radius)

        for ix in range(ix_min, ix_max + 1):
            for iy in range(iy_min, iy_max + 1):
                for cell_index in mesh_sampler.bin_to_cells.get((ix, iy), []):
                    if cell_index in checked_cells:
                        continue

                    checked_cells.add(cell_index)
                    stencil = build_interpolation_stencil(mesh_sampler, cell_index, point_x, point_y)
                    if stencil is not None:
                        return stencil

    # global fallback only if the spatial index failed (should be rare)
    for cell_index in range(mesh_sampler.connectivity.shape[0]):
        if cell_index in checked_cells:
            continue

        stencil = build_interpolation_stencil(mesh_sampler, cell_index, point_x, point_y)
        if stencil is not None:
            return stencil

    return None


def sample_fields_from_stencil(
    mesh_sampler: QuadMeshSampler,
    stencil: InterpolationStencil,
) -> dict[str, float]:
    """Interpolate all required flow variables from a located stencil.

    Args:
        mesh_sampler: Populated quad mesh sampler.
        stencil: Located interpolation stencil.

    Returns:
        Dictionary of interpolated field values at the query point.
    """

    # interpolate each required field using the precomputed barycentric weights
    sampled_values: dict[str, float] = {}
    node_indices = np.asarray(stencil.node_indices, dtype=int)

    for field_name in REQUIRED_FIELDS:
        nodal_values = mesh_sampler.nodal_fields[field_name][node_indices]
        sampled_values[field_name] = float(np.dot(stencil.weights, nodal_values))

    return sampled_values


# --------------------------------------------------
# arc-length and station normals
# --------------------------------------------------
def compute_wall_arc_length(wall_x: np.ndarray, wall_y: np.ndarray) -> np.ndarray:
    """Compute cumulative arc length along the extracted wall polyline.

    Args:
        wall_x: Wall x-coordinates.
        wall_y: Wall y-coordinates.

    Returns:
        Arc-length array s starting at 0.
    """

    # compute segment lengths and accumulate
    dx = np.diff(wall_x)
    dy = np.diff(wall_y)
    ds = np.hypot(dx, dy)
    s = np.concatenate(([0.0], np.cumsum(ds)))

    return s


def build_station_normals(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    station_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build station locations and outward unit normals from the wall polyline.

    The tangent t_hat = (dx/ds, dy/ds) is computed by second-order centered
    finite differences.  The outward normal is n_hat = (-dy/ds, dx/ds),
    sign-corrected so n_y > 0.

    Args:
        wall_x: Wall x-coordinates.
        wall_y: Wall y-coordinates.
        station_x: Streamwise x-coordinates for the extraction stations.

    Returns:
        Tuple of (station_y, station_s, normal_x, normal_y, wall_s).
    """

    # compute the wall arc-length coordinate
    wall_s = compute_wall_arc_length(wall_x, wall_y)

    # interpolate wall height and arc length at each station
    station_y = np.interp(station_x, wall_x, wall_y)
    station_s = np.interp(station_x, wall_x, wall_s)

    # differentiate the wall polyline with respect to arc length
    edge_order = 2 if wall_x.size >= 3 else 1
    tangent_x = np.gradient(wall_x, wall_s, edge_order=edge_order)
    tangent_y = np.gradient(wall_y, wall_s, edge_order=edge_order)

    # interpolate tangent components to station locations
    station_tangent_x = np.interp(station_s, wall_s, tangent_x)
    station_tangent_y = np.interp(station_s, wall_s, tangent_y)

    # normalize the tangent vectors
    tangent_norm = np.hypot(station_tangent_x, station_tangent_y)
    station_tangent_x = station_tangent_x / tangent_norm
    station_tangent_y = station_tangent_y / tangent_norm

    # rotate tangent 90 degrees counterclockwise to get the outward normal
    normal_x = -station_tangent_y
    normal_y = station_tangent_x

    # sign-correct to ensure the normal points away from the wall (n_y > 0)
    flip_mask = normal_y < 0.0
    normal_x[flip_mask] *= -1.0
    normal_y[flip_mask] *= -1.0

    return station_y, station_s, normal_x, normal_y, wall_s


def compute_eta_max(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    wall_x: np.ndarray,
    wall_y: np.ndarray,
) -> float:
    """Estimate the wall-normal profile extent as the 95th percentile cell height.

    Args:
        cell_x: Cell centroid x-coordinates.
        cell_y: Cell centroid y-coordinates.
        wall_x: Wall x-coordinates.
        wall_y: Wall y-coordinates.

    Returns:
        eta_max in physical length units.
    """

    # interpolate wall height to centroid x locations and compute cell elevation
    wall_y_at_cells = np.interp(cell_x, wall_x, wall_y)
    wall_distance = cell_y - wall_y_at_cells

    # use the 95th percentile to avoid far-field corner cells skewing the range
    eta_max = float(np.quantile(wall_distance, 0.95))
    return eta_max


# --------------------------------------------------
# profile sampling
# --------------------------------------------------
def sample_profiles(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    mesh_sampler: QuadMeshSampler,
    station_x: np.ndarray,
    n_eta: int = N_ETA,
) -> SampledProfiles:
    """Sample wall-normal profiles using barycentric interpolation on the quad mesh.

    Args:
        wall_x: Lower-wall x-coordinates.
        wall_y: Lower-wall y-coordinates.
        mesh_sampler: Populated quad mesh sampler.
        station_x: Streamwise x-coordinates of extraction stations.
        n_eta: Number of wall-normal sample points per profile.

    Returns:
        Sampled profile arrays for all requested stations.

    Raises:
        ValueError: If ``station_x`` is not strictly increasing or lies outside
            the wall x-range.
    """

    # validate station locations
    n_stations = station_x.size
    if np.any(np.diff(station_x) <= 0.0):
        raise ValueError("station_x must be strictly increasing")

    if station_x[0] < wall_x[0] or station_x[-1] > wall_x[-1]:
        raise ValueError(
            "station_x must lie within the extracted wall x-range "
            f"[{wall_x[0]:.6e}, {wall_x[-1]:.6e}]"
        )

    # build station locations and wall-normal unit vectors
    station_y, station_s, station_normal_x, station_normal_y, _ = build_station_normals(
        wall_x,
        wall_y,
        station_x,
    )

    # estimate the wall-normal profile extent from cell elevations
    eta_max = compute_eta_max(mesh_sampler.cell_x, mesh_sampler.cell_y, wall_x, wall_y)
    eta = np.linspace(0.0, eta_max, n_eta)

    # initialize output arrays: shape (n_stations, n_eta)
    sample_x_all = np.zeros((n_stations, n_eta), dtype=float)
    sample_y_all = np.zeros((n_stations, n_eta), dtype=float)
    uvel = np.zeros((n_stations, n_eta), dtype=float)
    vvel = np.zeros((n_stations, n_eta), dtype=float)
    wvel = np.zeros((n_stations, n_eta), dtype=float)
    temp = np.zeros((n_stations, n_eta), dtype=float)
    pres = np.zeros((n_stations, n_eta), dtype=float)
    rho = np.zeros((n_stations, n_eta), dtype=float)

    # sample each station along an approximate wall-normal ray
    for station_index in range(n_stations):
        x0 = station_x[station_index]
        y0 = station_y[station_index]
        normal_x = station_normal_x[station_index]
        normal_y = station_normal_y[station_index]

        # build the ray: x(eta) = x_wall + eta * n_hat
        sample_x = x0 + eta * normal_x
        sample_y = y0 + eta * normal_y

        # allocate per-station profile arrays
        u_profile = np.zeros(n_eta, dtype=float)
        v_profile = np.zeros(n_eta, dtype=float)
        w_profile = np.zeros(n_eta, dtype=float)
        t_profile = np.zeros(n_eta, dtype=float)
        p_profile = np.zeros(n_eta, dtype=float)
        rho_profile = np.zeros(n_eta, dtype=float)

        previous_cell_index: int | None = None
        previous_values: dict[str, float] | None = None

        for eta_index in range(n_eta):
            point_x = float(sample_x[eta_index])
            point_y = float(sample_y[eta_index])

            stencil = locate_interpolation_stencil(
                mesh_sampler,
                point_x,
                point_y,
                previous_cell_index,
            )

            if stencil is None:
                # extrapolate using the last successfully sampled point
                if previous_values is None:
                    raise ValueError(
                        f"Could not locate sample point at x={point_x:.6e}, y={point_y:.6e}"
                    )
                sampled_values = previous_values.copy()
            else:
                sampled_values = sample_fields_from_stencil(mesh_sampler, stencil)
                previous_cell_index = stencil.cell_index
                previous_values = sampled_values.copy()

            u_profile[eta_index] = sampled_values["u"]
            v_profile[eta_index] = sampled_values["v"]
            w_profile[eta_index] = sampled_values["w"]
            t_profile[eta_index] = sampled_values["t"]
            p_profile[eta_index] = sampled_values["p"]
            rho_profile[eta_index] = sampled_values["rho"]

        # enforce no-slip wall boundary condition at eta = 0
        u_profile[0] = 0.0
        v_profile[0] = 0.0
        w_profile[0] = 0.0

        # use the first interior sample for the wall thermodynamic state
        if n_eta > 1:
            t_profile[0] = t_profile[1]
            p_profile[0] = p_profile[1]
            rho_profile[0] = p_profile[0] / (GAS_CONSTANT * t_profile[0])

        # store station results into the output arrays
        sample_x_all[station_index, :] = sample_x
        sample_y_all[station_index, :] = sample_y
        uvel[station_index, :] = u_profile
        vvel[station_index, :] = v_profile
        wvel[station_index, :] = w_profile
        temp[station_index, :] = t_profile
        pres[station_index, :] = p_profile
        rho[station_index, :] = rho_profile

    return SampledProfiles(
        station_x=station_x,
        station_y=station_y,
        station_s=station_s,
        eta=eta,
        sample_x=sample_x_all,
        sample_y=sample_y_all,
        uvel=uvel,
        vvel=vvel,
        wvel=wvel,
        temp=temp,
        pres=pres,
        rho=rho,
    )


# --------------------------------------------------
# freestream attributes
# --------------------------------------------------
def compute_freestream_attrs(
    profiles: SampledProfiles,
    mach: float,
    t_inf: float,
) -> dict[str, float]:
    """Compute HDF5 root attributes from freestream conditions via Sutherland's law.

    Args:
        profiles: Sampled wall-normal profile data (used for edge-state pressure).
        mach: Freestream Mach number.
        t_inf: Freestream static temperature in Kelvin.

    Returns:
        Attribute dictionary with exact names expected by the lst_next_gen
        ``Hdf5Loader``.
    """

    # compute freestream pressure from the mean edge-state value
    p_inf = float(np.mean(profiles.pres[:, -1]))

    # compute freestream density from the ideal gas law
    rho_inf = p_inf / (GAS_CONSTANT * t_inf)

    # compute freestream viscosity from Sutherland's law
    mu_inf = (
        SUTHERLAND_MU0
        * (t_inf / SUTHERLAND_T0) ** 1.5
        * (SUTHERLAND_T0 + SUTHERLAND_S)
        / (t_inf + SUTHERLAND_S)
    )

    # build the attribute dictionary with exact names matched by Hdf5Loader substrings
    attrs = {
        "mach number": mach,
        "heat capacity ratio": GAMMA,
        "prandtl number": 0.71,
        "gas constant": GAS_CONSTANT,
        "static temperature": t_inf,
        "static density": rho_inf,
        "freestream viscosity": mu_inf,
    }

    return attrs


# --------------------------------------------------
# HDF5 writer
# --------------------------------------------------
def write_profiles_hdf5(
    path: str | Path,
    profiles: SampledProfiles,
    attrs: dict[str, float],
) -> Path:
    """Write the extracted profiles to HDF5 in the lst_next_gen schema.

    All datasets have shape (N_ETA, n_stations) — rows are wall-normal points,
    columns are streamwise stations — matching the layout expected by
    ``lst_next_gen``'s ``Hdf5Loader``.

    Args:
        path: Output HDF5 file path.
        profiles: Sampled wall-normal profile data.
        attrs: Freestream attribute dictionary for root-level metadata.

    Returns:
        Resolved output path.
    """

    # convert to Path object
    out_path = Path(path)

    # write HDF5 file with the lst_next_gen schema
    with h5py.File(out_path, "w") as hdf5_file:
        # write all root-level freestream attributes as scalar float64
        for attr_name, attr_value in attrs.items():
            hdf5_file.attrs[attr_name] = np.float64(attr_value)

        # write all datasets as 2D arrays with shape (N_ETA, n_stations)
        # the .T transposes from (n_stations, n_eta) to (n_eta, n_stations)
        hdf5_file.create_dataset("x",    data=profiles.sample_x.T, dtype=np.float64)
        hdf5_file.create_dataset("y",    data=profiles.sample_y.T, dtype=np.float64)
        hdf5_file.create_dataset("uvel", data=profiles.uvel.T,     dtype=np.float64)
        hdf5_file.create_dataset("vvel", data=profiles.vvel.T,     dtype=np.float64)
        hdf5_file.create_dataset("wvel", data=profiles.wvel.T,     dtype=np.float64)
        hdf5_file.create_dataset("temp", data=profiles.temp.T,     dtype=np.float64)
        hdf5_file.create_dataset("pres", data=profiles.pres.T,     dtype=np.float64)
        hdf5_file.create_dataset("dens", data=profiles.rho.T,      dtype=np.float64)

    return out_path


# --------------------------------------------------
# Tecplot writers
# --------------------------------------------------
def write_profiles_tecplot(
    path: str | Path,
    profiles: SampledProfiles,
) -> Path:
    """Write the extracted wall-normal profiles as a multi-zone Tecplot file.

    Args:
        path: Output Tecplot ASCII file path.
        profiles: Sampled wall-normal profile data.

    Returns:
        Resolved output path.
    """

    # convert to Path object
    out_path = Path(path)

    # write one ordered 1-D zone per station
    with out_path.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "extracted_profiles"\n')
        stream.write('VARIABLES = "x" "y" "eta" "u" "v" "w" "T" "p" "rho"\n')

        for station_index in range(profiles.station_x.size):
            station_x = profiles.station_x[station_index]
            station_y = profiles.station_y[station_index]
            zone_name = (
                f"extracted_profile_{station_index:03d} "
                f"x={station_x:.6e} "
                f"y={station_y:.6e}"
            )
            stream.write(
                f'ZONE T="{zone_name}", I={profiles.eta.size}, DATAPACKING=POINT\n'
            )

            for eta_index in range(profiles.eta.size):
                stream.write(
                    f"{profiles.sample_x[station_index, eta_index]:.8e} "
                    f"{profiles.sample_y[station_index, eta_index]:.8e} "
                    f"{profiles.eta[eta_index]:.8e} "
                    f"{profiles.uvel[station_index, eta_index]:.8e} "
                    f"{profiles.vvel[station_index, eta_index]:.8e} "
                    f"{profiles.wvel[station_index, eta_index]:.8e} "
                    f"{profiles.temp[station_index, eta_index]:.8e} "
                    f"{profiles.pres[station_index, eta_index]:.8e} "
                    f"{profiles.rho[station_index, eta_index]:.8e}\n"
                )

    return out_path


def write_wall_profile_tecplot(
    path: str | Path,
    wall_x: np.ndarray,
    wall_y: np.ndarray,
) -> Path:
    """Write the extracted wall curve as a 1-D Tecplot line zone.

    Args:
        path: Output Tecplot ASCII file path.
        wall_x: Wall x-coordinates.
        wall_y: Wall y-coordinates.

    Returns:
        Resolved output path.
    """

    # convert to Path object
    out_path = Path(path)

    # write a simple ordered line zone
    with out_path.open("w", encoding="utf-8") as stream:
        stream.write('TITLE = "wall_profile"\n')
        stream.write('VARIABLES = "x_wall" "y_wall"\n')
        stream.write(f'ZONE T="wall", I={wall_x.size}, DATAPACKING=POINT\n')

        for x_value, y_value in zip(wall_x, wall_y):
            stream.write(f"{x_value:.8e} {y_value:.8e}\n")

    return out_path
