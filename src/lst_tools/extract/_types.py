"""Dataclasses shared across the FE-quad extraction pipeline."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
