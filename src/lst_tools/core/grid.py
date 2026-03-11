"""Immutable structured-grid container."""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, Mapping
import numpy as np


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# data class for grid
# --------------------------------------------------
@dataclass(frozen=True)
class Grid:
    """Two- or three-dimensional structured grid."""

    # data class attributes

    # x, y, z: numpy arrays of grid coordinates
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None

    # grid attributes: dictionary of additional attributes
    # e.g., {'attribute_name': value}
    attrs: Mapping[str, float | int | str] = None
    cfg: dict[str, Any] | None = None

    # data class methods
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the grid arrays."""
        return self.x.shape
