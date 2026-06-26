"""Backward-compatibility facade for the FE-quad extraction package.

The implementation was split into focused modules (``_types``, ``_reader``,
``_wall``, ``_mesh``, ``_profile``, ``_writers``).  This module re-exports the
public names so existing imports from ``lst_tools.extract._fequad`` keep
working unchanged.
"""

# --------------------------------------------------
# re-export public names for backward compatibility
# --------------------------------------------------
from ._types import (
    InterpolationStencil,
    QuadMeshSampler,
    SampledProfiles,
    TecplotUnstructuredData,
)
from ._reader import read_fequad_block_tecplot
from ._wall import (
    build_boundary_edges,
    extract_body_wall,
    extract_lower_wall,
    order_boundary_loop,
)
from ._mesh import (
    build_quad_mesh_sampler,
    locate_interpolation_stencil,
    sample_fields_from_stencil,
)
from ._profile import (
    DEFAULT_ETA_DISTRIBUTION,
    N_ETA,
    build_eta_coordinates,
    build_station_normals,
    build_wall_branches,
    compute_eta_max,
    compute_freestream_attrs,
    pick_wall_branch,
    sample_profiles,
)
from ._writers import (
    write_profiles_hdf5,
    write_profiles_tecplot,
    write_wall_profile_tecplot,
)


__all__ = [
    "InterpolationStencil",
    "QuadMeshSampler",
    "SampledProfiles",
    "TecplotUnstructuredData",
    "read_fequad_block_tecplot",
    "build_boundary_edges",
    "extract_body_wall",
    "extract_lower_wall",
    "order_boundary_loop",
    "build_quad_mesh_sampler",
    "locate_interpolation_stencil",
    "sample_fields_from_stencil",
    "DEFAULT_ETA_DISTRIBUTION",
    "N_ETA",
    "build_eta_coordinates",
    "build_station_normals",
    "build_wall_branches",
    "compute_eta_max",
    "compute_freestream_attrs",
    "pick_wall_branch",
    "sample_profiles",
    "write_profiles_hdf5",
    "write_profiles_tecplot",
    "write_wall_profile_tecplot",
]
