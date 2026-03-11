"""Immutable flow-field container bound to a Grid."""

# --------------------------------------------------
# load necessary libraries
# --------------------------------------------------
from dataclasses import dataclass
import logging
from typing import Mapping
import numpy as np
from .grid import Grid


# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# flow data structure (immutable)
# --------------------------------------------------
@dataclass(frozen=True)
class Flow:
    """Flow fields defined on a Grid."""

    # data class attributes
    grid: Grid
    fields: Mapping[str, np.ndarray]
    attrs: Mapping[str, float | int | str]

    # data class methods
    def field(self, name: str) -> np.ndarray:
        """Return the named field array, checking shape against the grid.

        Args:
            name (str): Key into the ``fields`` mapping.

        Returns:
            np.ndarray: The field array.

        Raises:
            KeyError: If *name* is not in ``fields``.
            ValueError: If the field shape does not match the grid shape.
        """
        a = self.fields[name]
        if a.shape != self.grid.shape:
            raise ValueError(
                f"field '{name}' shape {a.shape} != grid shape {self.grid.shape}"
            )
        return a
