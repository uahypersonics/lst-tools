"""Wall-normal profile extraction from unstructured FE-quad Tecplot meshes."""

from ._fequad import (
    SampledProfiles,
    build_quad_mesh_sampler,
    extract_lower_wall,
    read_fequad_block_tecplot,
    resolve_profile_stations,
    sample_profiles,
    write_profiles_hdf5,
    write_profiles_tecplot,
    write_wall_profile_tecplot,
)
from ._normalize import detect_dimensional, normalize_profiles

__all__ = [
    "SampledProfiles",
    "extract_lower_wall",
    "build_quad_mesh_sampler",
    "sample_profiles",
    "read_fequad_block_tecplot",
    "resolve_profile_stations",
    "write_profiles_hdf5",
    "write_profiles_tecplot",
    "write_wall_profile_tecplot",
    "detect_dimensional",
    "normalize_profiles",
]
