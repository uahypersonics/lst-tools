"""Profile geometry, wall-normal sampling, and freestream attributes."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging

import numpy as np

from ._mesh import locate_interpolation_stencil, sample_fields_from_stencil
from ._types import QuadMeshSampler, SampledProfiles

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)

# --------------------------------------------------
# set sensible defaults for profile extraction
# --------------------------------------------------
#
# number of points along the wall-normal direction
default_n_eta = 400
#
# default point distribution
default_eta_distribution = "tanh"
#
# default stretching strength for the tanh distribution
default_eta_stretch = 2.0


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


# --------------------------------------------------
# split the wall arc into lower and upper branches
# --------------------------------------------------
def build_wall_branches(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the body arc into lower and upper x-monotone branches.

    For a full two-sided body contour the arc is split at the nose (minimum x
    point).  For a one-sided slice (nose at the start or end of the arc) the
    entire arc is returned as whichever branch matches the mean y sign, and the
    other branch is returned as an empty array.

    Args:
        wall_x: Wall x-coordinates in arc order.
        wall_y: Wall y-coordinates in arc order.

    Returns:
        Tuple of ``(lower_x, lower_y, upper_x, upper_y)`` with each branch sorted
        by increasing x.

    Raises:
        ValueError: If the wall arc does not contain enough points to build both
            branches.
    """

    # validate inputs
    if wall_x.size < 3:
        raise ValueError("wall arc must contain at least 3 points")

    # find the nose point, which separates the lower and upper branches
    nose_index = int(np.argmin(wall_x))

    # one-sided arc: nose is at an end of the arc, not in the interior.
    # This happens when the mesh covers only the upper or lower surface.
    # Return the whole arc as the appropriate branch and leave the other empty.
    if nose_index == 0 or nose_index == wall_x.size - 1:
        order = np.argsort(wall_x)
        sorted_x = wall_x[order]
        sorted_y = wall_y[order]
        empty_x: np.ndarray = np.empty(0, dtype=wall_x.dtype)
        empty_y: np.ndarray = np.empty(0, dtype=wall_y.dtype)
        # classify by mean y: positive y is the upper surface
        if np.mean(wall_y) >= 0.0:
            return empty_x, empty_y, sorted_x, sorted_y
        else:
            return sorted_x, sorted_y, empty_x, empty_y

    # build the lower branch from trailing edge to nose
    lower_x = wall_x[: nose_index + 1][::-1]
    lower_y = wall_y[: nose_index + 1][::-1]

    # build the upper branch from nose to trailing edge
    upper_x = wall_x[nose_index:]
    upper_y = wall_y[nose_index:]

    # sort each branch by x so interpolation/search is well-defined
    lower_order = np.argsort(lower_x)
    upper_order = np.argsort(upper_x)

    lower_x = lower_x[lower_order]
    lower_y = lower_y[lower_order]
    upper_x = upper_x[upper_order]
    upper_y = upper_y[upper_order]

    return lower_x, lower_y, upper_x, upper_y


# --------------------------------------------------
# pick the wall branch that matches the requested surface side
# --------------------------------------------------
def pick_wall_branch(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    target_y: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the wall branch that matches the requested surface side.

    For a one-sided mesh (only upper or only lower surface present) the
    available branch is returned automatically, with a warning if it differs
    from what was requested.

    Args:
        wall_x: Wall x-coordinates in arc order.
        wall_y: Wall y-coordinates in arc order.
        target_y: Optional preferred surface side. Positive selects the upper
            branch, negative selects the lower branch, and ``None`` selects the
            only available branch on a one-sided wall.

    Returns:
        Selected wall branch as ``(branch_x, branch_y)`` sorted by increasing x.
    """

    # build the two physical branches of the body arc
    lower_x, lower_y, upper_x, upper_y = build_wall_branches(wall_x, wall_y)

    lower_available = lower_x.size >= 2
    upper_available = upper_x.size >= 2

    # auto-select only when the wall has one unambiguous physical branch
    if target_y is None:
        if lower_available and upper_available:
            raise ValueError(
                "surface selection required for two-sided wall; "
                "set surface to 'lower' or 'upper'"
            )
        if lower_available:
            selected_x, selected_y = lower_x, lower_y
        elif upper_available:
            selected_x, selected_y = upper_x, upper_y
        else:
            raise ValueError("wall does not contain a valid surface branch")
    else:
        if target_y <= 0.0:
            selected_x, selected_y = lower_x, lower_y
            requested = "lower"
            fallback_x, fallback_y = upper_x, upper_y
            fallback = "upper"
        else:
            selected_x, selected_y = upper_x, upper_y
            requested = "upper"
            fallback_x, fallback_y = lower_x, lower_y
            fallback = "lower"

        # auto-fallback for one-sided meshes: if the requested branch is empty or
        # degenerate, use the available branch and warn the user
        if selected_x.size < 2 and fallback_x.size >= 2:
            logger.warning(
                "requested '%s' surface has only %d point(s) — "
                "this appears to be a one-sided %s-surface mesh. "
                "automatically using the '%s' surface. "
                "pass --surface %s to suppress this warning.",
                requested,
                selected_x.size,
                fallback,
                fallback,
                fallback,
            )
            selected_x, selected_y = fallback_x, fallback_y

    return selected_x, selected_y


# --------------------------------------------------
# resolve requested extraction stations against the selected wall branch
# --------------------------------------------------
def resolve_profile_stations(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    requested_x: np.ndarray | None,
    target_y: float | None,
    snap_to_wall: bool = False,
    all_wall_points: bool = False,
) -> np.ndarray:
    """Resolve profile x-coordinates for the requested station-selection mode.

    Args:
        wall_x: Wall x-coordinates in arc order.
        wall_y: Wall y-coordinates in arc order.
        requested_x: Explicit or range-generated x-coordinates. May be ``None``
            only when ``all_wall_points`` is enabled.
        target_y: Preferred surface side passed to :func:`pick_wall_branch`.
        snap_to_wall: Snap each requested x-coordinate to its nearest wall node.
        all_wall_points: Extract one profile at every node on the selected wall.

    Returns:
        Strictly increasing profile x-coordinates.

    Raises:
        ValueError: If no stations are available for the selected mode.
    """

    # select the same physical wall branch used by profile sampling
    branch_x, _ = pick_wall_branch(wall_x, wall_y, target_y)

    # use every defined wall point when requested
    if all_wall_points:
        stations = np.unique(branch_x)
    elif requested_x is None or requested_x.size == 0:
        raise ValueError("extraction station list cannot be empty")
    elif snap_to_wall:
        # map each requested location to its nearest defined wall point
        nearest_indices = np.abs(branch_x[:, np.newaxis] - requested_x).argmin(axis=0)
        stations = np.unique(branch_x[nearest_indices])
    else:
        # retain continuous x locations for wall-edge interpolation
        stations = np.unique(np.asarray(requested_x, dtype=float))

    if stations.size == 0:
        raise ValueError("selected wall does not contain any profile stations")

    return stations


# --------------------------------------------------
# routine to generate eta coordinates for extraction
# --------------------------------------------------
def build_eta_coordinates(
    eta_max: float,
    n_eta: int,
    distribution: str = default_eta_distribution,
    eta_stretch: float = default_eta_stretch,
    eta_wall_spacing: float | None = None,
) -> np.ndarray:
    """Build the wall-normal sampling coordinates.

    Args:
        eta_max: Maximum wall-normal distance.
        n_eta: Number of wall-normal sample points.
        distribution: Point distribution name. ``uniform`` uses equally spaced
            points, ``cosine`` clusters points near the wall, ``tanh`` applies
            near-wall clustering controlled by ``eta_stretch``, and
            ``geometric`` grows intervals from ``eta_wall_spacing``.
        eta_stretch: Stretching strength for ``tanh`` distribution.
        eta_wall_spacing: First off-wall interval for ``geometric`` distribution.

    Returns:
        Wall-normal coordinate array from 0 to ``eta_max``.

    Raises:
        ValueError: If the distribution name is unsupported.
    """

    # validate inputs
    if n_eta < 2:
        raise ValueError("n_eta must be at least 2")
    if eta_max <= 0.0:
        raise ValueError("eta_max must be positive")
    if eta_stretch <= 0.0:
        raise ValueError("eta_stretch must be positive")

    # build the normalized wall-normal coordinate
    xi = np.linspace(0.0, 1.0, n_eta)

    # build the requested point distribution
    if distribution == "uniform":
        # uniform distribution from 0 to eta_max
        eta = eta_max * xi
    elif distribution == "cosine":
        # cosine distribution: clusters points near the wall with a half-cosine from 0 to pi/2 mapping to 0 to eta_max
        eta = eta_max * (1.0 - np.cos(0.5 * np.pi * xi))
    elif distribution == "tanh":
        # tanh distribution: clusters points near the wall with a hyperbolic tangent
        eta = eta_max * (1.0 - np.tanh(eta_stretch * (1.0 - xi)) / np.tanh(eta_stretch))
    elif distribution == "geometric":
        # geometric distribution: first interval is eta_wall_spacing, subsequent intervals grow geometrically
        if eta_wall_spacing is None or eta_wall_spacing <= 0.0:
            raise ValueError(
                "eta_wall_spacing must be positive for geometric distribution"
            )

        # require intervals to grow away from the wall rather than contract
        n_intervals = n_eta - 1
        uniform_spacing = eta_max / n_intervals
        if eta_wall_spacing > uniform_spacing:
            raise ValueError(
                "eta_wall_spacing must not exceed eta_max / (n_eta - 1) "
                "for geometric distribution"
            )

        # use uniform spacing for the limiting geometric ratio r = 1
        if np.isclose(eta_wall_spacing, uniform_spacing):
            eta = eta_max * xi
        else:
            # solve eta_max = deta_wall * sum(r**j) for the ratio r > 1
            powers = np.arange(n_intervals, dtype=float)

            def geometric_extent(ratio: float) -> float:
                intervals = eta_wall_spacing * np.power(ratio, powers)
                extent = float(np.sum(intervals))
                return extent

            ratio_lower = 1.0
            ratio_upper = 2.0
            while geometric_extent(ratio_upper) < eta_max:
                ratio_upper *= 2.0

            for _ in range(100):
                ratio_midpoint = 0.5 * (ratio_lower + ratio_upper)
                if geometric_extent(ratio_midpoint) < eta_max:
                    ratio_lower = ratio_midpoint
                else:
                    ratio_upper = ratio_midpoint

            stretch_ratio = 0.5 * (ratio_lower + ratio_upper)
            intervals = eta_wall_spacing * np.power(stretch_ratio, powers)
            eta = np.concatenate(([0.0], np.cumsum(intervals)))
            eta[-1] = eta_max
    else:
        raise ValueError(
            "eta distribution must be 'uniform', 'cosine', 'tanh', or 'geometric'"
        )

    return eta


# --------------------------------------------------
# compute wall normals at each station
# --------------------------------------------------
def build_station_normals(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    station_x: np.ndarray,
    target_y: float | None = None,
    body_centroid: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build station locations and unit normals pointing outward from the body.

    The wall may be an arc in loop order (not sorted by x). For each station_x,
    finds the closest wall point by arc-length search, computes the local tangent
    and normal, and sign-corrects the normal to point away from the body centroid.

    Args:
        wall_x: Wall x-coordinates (may be in loop order, not sorted).
        wall_y: Wall y-coordinates.
        station_x: Streamwise x-coordinates for the extraction stations.
        target_y: Optional preferred wall branch. Positive chooses the upper
            surface and negative chooses the lower surface.
        body_centroid: (x, y) centroid of the body surface. Normals point away
            from this point. If None, uses the mean of wall coordinates.

    Returns:
        Tuple of (station_y, station_s, normal_x, normal_y, wall_s).
    """

    # pick the physical wall branch to sample from
    branch_x, branch_y = pick_wall_branch(wall_x, wall_y, target_y)

    # compute the wall arc-length coordinate
    wall_s = compute_wall_arc_length(branch_x, branch_y)

    # compute body centroid if not provided
    if body_centroid is None:
        body_centroid = (float(np.mean(wall_x)), float(np.mean(wall_y)))
    centroid_x, centroid_y = body_centroid

    # for each station_x, intersect the piecewise-linear wall branch
    station_y = np.zeros(station_x.size)
    station_s = np.zeros(station_x.size)
    station_edge_indices = np.zeros(station_x.size, dtype=int)

    for i, sx in enumerate(station_x):
        # find the wall edge whose x interval contains this station
        edge_index = int(np.searchsorted(branch_x, sx, side="right") - 1)
        edge_index = max(0, min(edge_index, branch_x.size - 2))
        station_edge_indices[i] = edge_index

        # linearly interpolate the exact point and arc length on the FE wall edge
        edge_fraction = (sx - branch_x[edge_index]) / (
            branch_x[edge_index + 1] - branch_x[edge_index]
        )
        station_y[i] = branch_y[edge_index] + edge_fraction * (
            branch_y[edge_index + 1] - branch_y[edge_index]
        )
        station_s[i] = wall_s[edge_index] + edge_fraction * (
            wall_s[edge_index + 1] - wall_s[edge_index]
        )

    # use the tangent of each intersected FE wall edge
    station_tangent_x = (
        branch_x[station_edge_indices + 1] - branch_x[station_edge_indices]
    )
    station_tangent_y = (
        branch_y[station_edge_indices + 1] - branch_y[station_edge_indices]
    )

    # normalize the tangent vectors
    tangent_norm = np.hypot(station_tangent_x, station_tangent_y)
    station_tangent_x = station_tangent_x / tangent_norm
    station_tangent_y = station_tangent_y / tangent_norm

    # rotate tangent 90 degrees counterclockwise to get a candidate normal
    normal_x = -station_tangent_y
    normal_y = station_tangent_x

    # sign-correct to point away from body centroid (into the flow domain)
    for i in range(station_x.size):
        # vector from body centroid to station point
        to_station_x = station_x[i] - centroid_x
        to_station_y = station_y[i] - centroid_y

        # check if normal points in same general direction as centroid-to-station
        dot = normal_x[i] * to_station_x + normal_y[i] * to_station_y
        if dot < 0:
            # normal points toward centroid, flip it
            normal_x[i] *= -1.0
            normal_y[i] *= -1.0

    # enforce the requested surface side for one-sided wall branches.
    # Top-only and bottom-only meshes can place the centroid on the same side of
    # the wall, which makes the centroid dot-product test ambiguous.
    branch_y_min = float(np.min(branch_y))
    branch_y_max = float(np.max(branch_y))
    is_one_sided_branch = branch_y_min >= -1.0e-6 or branch_y_max <= 1.0e-6

    # determine the outward direction from the branch geometry itself.
    # pick_wall_branch may auto-select the opposite surface on a one-sided mesh
    # (e.g. a top-only mesh when the default 'lower' side was requested), so the
    # outward normal must follow where the mesh data actually lies, not the
    # requested target_y label.
    # check which side the one-sided branch sits on and set the outward sign
    if branch_y_min >= -1.0e-6:
        # all wall points lie at y >= 0: this is an upper surface, normals go +y
        effective_target_y = 1.0
    elif branch_y_max <= 1.0e-6:
        # all wall points lie at y <= 0: this is a lower surface, normals go -y
        effective_target_y = -1.0
    else:
        # two-sided branch: keep the caller's requested side preference
        effective_target_y = target_y

    if effective_target_y is not None and is_one_sided_branch:
        for i in range(station_x.size):
            # flip any normal that points into the body instead of the flow
            if effective_target_y * normal_y[i] < 0.0:
                normal_x[i] *= -1.0
                normal_y[i] *= -1.0

    logger.debug("station normals: centroid=(%.4e, %.4e)", centroid_x, centroid_y)

    return station_y, station_s, normal_x, normal_y, wall_s


# --------------------------------------------------
# compute the maximum wall-normal extent of the profile grid
# --------------------------------------------------
def compute_eta_max(
    cell_x: np.ndarray,
    cell_y: np.ndarray,
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    target_y: float | None = None,
) -> float:
    """Estimate the wall-normal profile extent as the 95th percentile cell distance.

    Args:
        cell_x: Cell centroid x-coordinates.
        cell_y: Cell centroid y-coordinates.
        wall_x: Wall x-coordinates.
        wall_y: Wall y-coordinates.
        target_y: Optional preferred wall branch. Positive chooses the upper
            surface and negative chooses the lower surface.

    Returns:
        eta_max in physical length units.
    """

    # pick the physical wall branch before interpolating in x
    branch_x, branch_y = pick_wall_branch(wall_x, wall_y, target_y)

    # check which cells lie on the selected side of the body
    if target_y is not None and target_y > 0.0:
        side_mask = cell_y >= 0.0
    else:
        side_mask = cell_y <= 0.0

    # fall back to the full mesh if the side filter would remove everything
    if not np.any(side_mask):
        side_mask = np.ones(cell_y.shape, dtype=bool)

    # interpolate wall height to centroid x locations and compute cell distance
    wall_y_at_cells = np.interp(cell_x[side_mask], branch_x, branch_y)
    wall_distance = np.abs(cell_y[side_mask] - wall_y_at_cells)

    # use the 95th percentile to avoid far-field corner cells skewing the range
    eta_max = float(np.quantile(wall_distance, 0.95))
    return eta_max


# --------------------------------------------------
# profile sampling
# --------------------------------------------------
def _sample_one_station(
    x0: float,
    y0: float,
    normal_x: float,
    normal_y: float,
    eta: np.ndarray,
    mesh_sampler: QuadMeshSampler,
    rgas: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample all wall-normal points for a single station along its ray.

    Marches outward from the wall point ``(x0, y0)`` along the unit normal
    ``(normal_x, normal_y)`` and interpolates the mesh fields at each ``eta``
    coordinate. The near-wall gap (where the ray starts inside the solid body
    and finds no cell) is filled with the first valid interior sample, the
    no-slip condition is enforced at the wall, and the wall thermodynamic state
    is taken from the first interior point.

    Args:
        x0: Wall-point x-coordinate for this station.
        y0: Wall-point y-coordinate for this station.
        normal_x: x-component of the outward wall-normal unit vector.
        normal_y: y-component of the outward wall-normal unit vector.
        eta: Wall-normal sampling coordinates (length ``n_eta``).
        mesh_sampler: Populated quad mesh sampler.
        rgas: Specific gas constant (J/(kg·K)). Used for ideal-gas wall density.

    Returns:
        Tuple of ``(u_profile, v_profile, w_profile, t_profile, p_profile,
        rho_profile)``, each an array of length ``n_eta``.

    Raises:
        ValueError: If no sample point along the ray lands inside the mesh.
    """

    # number of wall-normal sample points
    n_eta = eta.size

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
    first_valid_values: dict[str, float] | None = None
    first_valid_idx: int = 0

    # find the first eta index where cells exist (skip wall boundary gap)
    for eta_index in range(n_eta):
        point_x = float(sample_x[eta_index])
        point_y = float(sample_y[eta_index])
        stencil = locate_interpolation_stencil(mesh_sampler, point_x, point_y, None)
        if stencil is not None:
            first_valid_idx = eta_index
            first_valid_values = sample_fields_from_stencil(mesh_sampler, stencil)
            previous_cell_index = stencil.cell_index
            break
    else:
        raise ValueError(f"Could not locate any sample point for station x={x0:.6e}")

    # sample from first_valid_idx onward
    for eta_index in range(first_valid_idx, n_eta):
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
            sampled_values = first_valid_values.copy()
        else:
            sampled_values = sample_fields_from_stencil(mesh_sampler, stencil)
            previous_cell_index = stencil.cell_index
            first_valid_values = sampled_values.copy()

        u_profile[eta_index] = sampled_values["u"]
        v_profile[eta_index] = sampled_values["v"]
        w_profile[eta_index] = sampled_values["w"]
        t_profile[eta_index] = sampled_values["t"]
        p_profile[eta_index] = sampled_values["p"]
        rho_profile[eta_index] = sampled_values["rho"]

    # fill in the wall boundary gap with the first valid values
    for eta_index in range(first_valid_idx):
        u_profile[eta_index] = first_valid_values["u"]
        v_profile[eta_index] = first_valid_values["v"]
        w_profile[eta_index] = first_valid_values["w"]
        t_profile[eta_index] = first_valid_values["t"]
        p_profile[eta_index] = first_valid_values["p"]
        rho_profile[eta_index] = first_valid_values["rho"]

    # enforce no-slip wall boundary condition at eta = 0
    u_profile[0] = 0.0
    v_profile[0] = 0.0
    w_profile[0] = 0.0

    # return the six sampled field profiles for this station
    return (
        u_profile,
        v_profile,
        w_profile,
        t_profile,
        p_profile,
        rho_profile,
    )


# --------------------------------------------------
# sample profiles along multiple stations
# --------------------------------------------------
def sample_profiles(
    wall_x: np.ndarray,
    wall_y: np.ndarray,
    mesh_sampler: QuadMeshSampler,
    station_x: np.ndarray,
    target_y: float | None = None,
    n_eta: int = default_n_eta,
    eta_max: float | None = None,
    eta_distribution: str = default_eta_distribution,
    eta_stretch: float = default_eta_stretch,
    eta_wall_spacing: float | None = None,
    rgas: float = 287.15,
) -> SampledProfiles:
    """Sample wall-normal profiles using barycentric interpolation on the quad mesh.

    Args:
        wall_x: Lower-wall x-coordinates.
        wall_y: Lower-wall y-coordinates.
        mesh_sampler: Populated quad mesh sampler.
        station_x: Streamwise x-coordinates of extraction stations.
        target_y: Optional preferred wall branch. Positive chooses the upper
            surface and negative chooses the lower surface.
        n_eta: Number of wall-normal sample points per profile.
        eta_max: Optional absolute wall-normal profile extent. When ``None``,
            an automatic estimate is computed from mesh geometry.
        eta_distribution: Point distribution along the wall-normal coordinate.
        eta_stretch: Stretching strength for ``tanh`` eta distribution.
        eta_wall_spacing: First off-wall interval for ``geometric`` distribution.
        rgas: Specific gas constant (J/(kg·K)). Used for ideal-gas wall density.

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

    # use min/max of wall_x for range check (wall may be in loop order, not sorted)
    wall_x_min, wall_x_max = float(wall_x.min()), float(wall_x.max())
    if station_x[0] < wall_x_min or station_x[-1] > wall_x_max:
        raise ValueError(
            "station_x must lie within the extracted wall x-range "
            f"[{wall_x_min:.6e}, {wall_x_max:.6e}]"
        )

    # compute body centroid for normal direction (normals point away from body)
    body_centroid = (float(np.mean(wall_x)), float(np.mean(wall_y)))

    # build station locations and wall-normal unit vectors
    station_y, station_s, station_normal_x, station_normal_y, _ = build_station_normals(
        wall_x,
        wall_y,
        station_x,
        target_y=target_y,
        body_centroid=body_centroid,
    )

    # resolve wall-normal profile extent (explicit override or automatic estimate)
    if eta_max is None:
        eta_max_value = compute_eta_max(
            mesh_sampler.cell_x,
            mesh_sampler.cell_y,
            wall_x,
            wall_y,
            target_y=target_y,
        )
    else:
        eta_max_value = float(eta_max)

    eta = build_eta_coordinates(
        eta_max_value,
        n_eta,
        distribution=eta_distribution,
        eta_stretch=eta_stretch,
        eta_wall_spacing=eta_wall_spacing,
    )

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

        # sample all wall-normal points for this station along its ray
        (
            u_profile,
            v_profile,
            w_profile,
            t_profile,
            p_profile,
            rho_profile,
        ) = _sample_one_station(
            x0,
            y0,
            normal_x,
            normal_y,
            eta,
            mesh_sampler,
            rgas,
        )

        # rebuild the ray coordinates for the output arrays
        sample_x = x0 + eta * normal_x
        sample_y = y0 + eta * normal_y

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
