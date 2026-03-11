"""Configuration management public API."""

from .check_consistency import check_consistency, format_report
from .find_config import find_config
from .geometry import GeometryPreset, GEOMETRY_TEMPLATES
from .merge import merge_dicts, merge_flow_defaults
from .read_config import read_config
from .schema import Config
from .write_config import write_config

__all__ = [
    # Core configuration operations
    "Config",
    "GeometryPreset",
    "GEOMETRY_TEMPLATES",
    "merge_dicts",
    "merge_flow_defaults",
    "read_config",
    "write_config",
    "find_config",

    # Consistency checking
    "check_consistency",
    "format_report",
]
