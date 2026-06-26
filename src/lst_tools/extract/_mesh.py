"""Quad mesh sampler construction and barycentric interpolation."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging

import numpy as np

from ._types import InterpolationStencil, QuadMeshSampler


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# constants
# --------------------------------------------------

# flow field names expected in the dataset (internal canonical names)
REQUIRED_FIELDS = ("u", "v", "w", "t", "p", "rho")


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
    existing_nodal_fields: dict[str, np.ndarray] | None = None,
) -> QuadMeshSampler:
    """Build a scalable quad mesh sampler with a uniform spatial bin index.

    Args:
        nodal_x: Node x-coordinates.
        nodal_y: Node y-coordinates.
        connectivity: FE quad connectivity (E × 4, 1-based).
        cell_fields: Cell-centered flow variable arrays.
        existing_nodal_fields: Optional pre-existing nodal fields (skip reconstruction).

    Returns:
        Populated ``QuadMeshSampler`` ready for point queries.
    """

    # convert 1-based Tecplot connectivity to 0-based indexing
    zero_based_connectivity = connectivity - 1

    # compute cell centroids
    cell_x, cell_y = build_cell_centers(nodal_x, nodal_y, connectivity)

    # use existing nodal fields if available, otherwise reconstruct from cell data
    if existing_nodal_fields is not None and all(
        f in existing_nodal_fields for f in REQUIRED_FIELDS
    ):
        nodal_fields = {f: existing_nodal_fields[f] for f in REQUIRED_FIELDS}
        logger.debug("using existing nodal fields (no reconstruction needed)")
    else:
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
# compute barycentric weights
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


# --------------------------------------------------
# compute bilinear map inverse for quads
# --------------------------------------------------
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
