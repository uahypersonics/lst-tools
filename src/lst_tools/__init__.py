"""LST Tools - Linear Stability Theory analysis tools."""

import logging
from importlib import import_module

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Define API modules and their exports
_API_MODULES = {
    "data_io": [
        "read_flow_conditions",
        "FortranBinaryWriter",
        "FortranBinaryReader",
        "LastracReader",
        "LastracWriter",
        "read_tecplot_ascii",
    ],
    "config": [
        "read_config",
        "write_config",
        "find_config",
        "check_consistency",
        "format_report",
    ],
    "core": [
        "Grid",
        "Flow",
    ],
    "convert": [
        "convert_meanflow",
        "generate_lst_input_deck",
    ],
    "geometry": [
        "curvature",
        "curvilinear_coordinate",
        "surface_angle",
        "list_geometry_kinds",
        "radius",
        "GeometryKind",
    ],
    "setup": [
        "parsing_setup",
        "tracking_setup",
        "spectra_setup",
    ],
    "process": [
        "tracking_process",
        "spectra_process",
    ],
    "hpc": [
        "ResolvedJob",
        "script_build",
        "hpc_configure",
    ],
}

# Import all API components
__all__ = []

for module_name, exports in _API_MODULES.items():
    # Import the submodule
    module = import_module(f".{module_name}", package="lst_tools")

    # Add exports to namespace
    for name in exports:
        globals()[name] = getattr(module, name)

    # Add to __all__
    __all__.extend(exports)

# Version handling — reads from pyproject.toml via the installed metadata

def _get_version():
    """Get the package version from installed metadata."""
    try:
        from importlib.metadata import version as pkg_version

        return pkg_version("lst-tools")
    except Exception:
        return "0.1.0"


__version__ = _get_version()

# Clean up namespace
del _API_MODULES, _get_version, import_module
