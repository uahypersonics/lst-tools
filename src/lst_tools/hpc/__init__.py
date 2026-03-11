"""HPC job-script generation public API."""

from ._detect import DetectedEnv, detect
from ._profiles import ClusterProfile, Scheduler
from ._resolve import ResolvedJob, resolve
from .configure import hpc_configure
from .scripts import script_build

__all__ = [
    # Configuration
    "ResolvedJob",
    "hpc_configure",
    # Detection
    "DetectedEnv",
    "detect",
    # Profiles
    "ClusterProfile",
    "Scheduler",
    # Resolution
    "resolve",
    # Script generation
    "script_build",
]
