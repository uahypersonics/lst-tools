"""Wall-normal profile extraction from unstructured FE-quad Tecplot meshes."""

from ._fequad import (
    SampledProfiles,
    compute_freestream_attrs,
    extract_lower_wall,
    build_quad_mesh_sampler,
    sample_profiles,
    read_fequad_block_tecplot,
    write_profiles_hdf5,
    write_profiles_tecplot,
    write_wall_profile_tecplot,
)
from ._normalize import detect_dimensional, normalize_profiles

__all__ = [
    "SampledProfiles",
    "compute_freestream_attrs",
    "extract_lower_wall",
    "build_quad_mesh_sampler",
    "sample_profiles",
    "read_fequad_block_tecplot",
    "write_profiles_hdf5",
    "write_profiles_tecplot",
    "write_wall_profile_tecplot",
]
