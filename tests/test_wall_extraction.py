"""Tests for body-wall extraction on full symmetric meshes and surface resolution."""

from __future__ import annotations

import numpy as np

from lst_tools.cli.cmd_extract import _resolve_surface
from lst_tools.extract._profile import pick_wall_branch
from lst_tools.extract._wall import extract_body_wall


# --------------------------------------------------
# synthetic mesh helpers
# --------------------------------------------------
def _build_full_cone_mesh(
    n_i: int = 21, n_j: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a synthetic FE-quad mesh wrapping a full (two-sided) cone slice.

    The wall (``j = 0``) runs from the lower trailing edge ``(1, -0.2)`` through
    the nose ``(0, 0)`` to the upper trailing edge ``(1, 0.2)``.  The farfield
    (``j = n_j - 1``) is an outward box kept inside ``0 < x < 1`` so the nose is
    the unique global minimum-x boundary node and the two trailing edges are the
    global maximum-x nodes — the layout the geometric fallback expects.

    Returns:
        ``(nodal_x, nodal_y, connectivity)`` with 1-based quad connectivity.
    """

    nodal_x = np.empty(n_i * n_j)
    nodal_y = np.empty(n_i * n_j)

    def node_id(i: int, j: int) -> int:  # 1-based
        return i * n_j + j + 1

    for i in range(n_i):
        s = i / (n_i - 1)
        u = 2.0 * s - 1.0
        wall_x, wall_y = abs(u), 0.2 * u
        far_x, far_y = 0.15 + 0.7 * u * u, 2.0 * u
        for j in range(n_j):
            t = j / (n_j - 1)
            k = node_id(i, j) - 1
            nodal_x[k] = (1.0 - t) * wall_x + t * far_x
            nodal_y[k] = (1.0 - t) * wall_y + t * far_y

    quads = []
    for i in range(n_i - 1):
        for j in range(n_j - 1):
            quads.append(
                [
                    node_id(i, j),
                    node_id(i + 1, j),
                    node_id(i + 1, j + 1),
                    node_id(i, j + 1),
                ]
            )
    connectivity = np.asarray(quads, dtype=int)

    return nodal_x, nodal_y, connectivity


# --------------------------------------------------
# tests
# --------------------------------------------------
def test_extract_body_wall_full_mesh() -> None:
    """Full symmetric mesh yields a two-sided wall arc split cleanly by surface."""

    # build
    nodal_x, nodal_y, connectivity = _build_full_cone_mesh()

    # execute (no nodal_fields -> exercises the geometric fallback)
    wall_x, wall_y = extract_body_wall(nodal_x, nodal_y, connectivity)

    # validate: the arc spans both surfaces and both trailing edges + nose
    assert wall_x.size >= 11
    assert wall_y.max() > 0.0
    assert wall_y.min() < 0.0
    assert np.isclose(wall_x.min(), 0.0)
    assert np.isclose(wall_x.max(), 1.0)

    # branch selection returns two distinct, sign-pure surfaces
    tol = 1.0e-9
    upper_x, upper_y = pick_wall_branch(wall_x, wall_y, target_y=1.0)
    lower_x, lower_y = pick_wall_branch(wall_x, wall_y, target_y=-1.0)

    assert upper_y.size > 1
    assert lower_y.size > 1

    upper_pos = bool(np.all(upper_y >= -tol)) and bool(np.all(lower_y <= tol))
    upper_neg = bool(np.all(upper_y <= tol)) and bool(np.all(lower_y >= -tol))
    assert upper_pos or upper_neg
    assert not np.array_equal(np.sort(upper_y), np.sort(lower_y))


def test_extract_body_wall_half_mesh() -> None:
    """Half-mesh (one-sided) falls back to |y|<tol and returns only wall nodes.

    The |y|<tol geometric fallback is designed for flat-plate / near-horizontal
    wall topologies where wall nodes have small |y| and the farfield is at large
    |y|.  This test verifies that farfield nodes do NOT leak into the result.
    """

    # build a flat-plate half-mesh: wall j=0 at y=0 (flat plate, exactly on
    # x-axis); farfield j=n_j-1 at large negative y.  No velocity data is
    # provided so the geometric fallback must fire.
    n_i = 21
    n_j = 5

    nodal_x = np.empty(n_i * n_j)
    nodal_y = np.empty(n_i * n_j)

    def nid(i: int, j: int) -> int:
        return i * n_j + j + 1

    for i in range(n_i):
        s = i / (n_i - 1)
        wx, wy = s, 0.0             # wall: flat plate at y=0
        fx, fy = s, -0.5            # farfield well below wall
        for j in range(n_j):
            t = j / (n_j - 1)
            k = nid(i, j) - 1
            nodal_x[k] = (1 - t) * wx + t * fx
            nodal_y[k] = (1 - t) * wy + t * fy

    quads = []
    for i in range(n_i - 1):
        for j in range(n_j - 1):
            quads.append([nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)])
    connectivity = np.asarray(quads, dtype=int)

    # no nodal_fields → exercises the geometric fallback
    wall_x, wall_y = extract_body_wall(nodal_x, nodal_y, connectivity)

    # farfield is at y=-0.5; tol = 5% of 0.5 = 0.025.  All wall nodes have
    # y=0 which is well within tol; no farfield node should appear.
    y_range = nodal_y.max() - nodal_y.min()
    tol = 0.05 * y_range
    assert wall_x.size > 0, "no wall nodes returned"
    assert np.all(np.abs(wall_y) < tol), (
        f"farfield nodes leaked into wall result: "
        f"|y|_max={np.abs(wall_y).max():.4e}, tol={tol:.4e}"
    )
    # wall must include both endpoints (nose at x=0 and TE at x=1)
    assert np.isclose(wall_x.min(), 0.0, atol=1e-9)
    assert np.isclose(wall_x.max(), 1.0, atol=1e-9)


def test_surface_sentinel_resolution() -> None:
    """Three-way surface resolution: explicit CLI flag always wins over config."""

    assert _resolve_surface(None, None) == "lower"
    assert _resolve_surface(None, "upper") == "upper"
    assert _resolve_surface("lower", "upper") == "lower"
    assert _resolve_surface("upper", "lower") == "upper"
