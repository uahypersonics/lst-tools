"""Tests for wall-normal eta coordinate construction."""

from __future__ import annotations

import numpy as np
import pytest

from lst_tools.extract._profile import build_eta_coordinates


class TestBuildEtaCoordinates:
    """Validate eta distribution behavior."""

    def test_uniform_distribution(self) -> None:
        """Uniform distribution should be linearly spaced."""
        eta = build_eta_coordinates(eta_max=2.0, n_eta=5, distribution="uniform")

        expected = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        assert np.allclose(eta, expected)

    def test_cosine_distribution_monotonic_and_bounded(self) -> None:
        """Cosine distribution should be monotone with endpoint inclusion."""
        eta = build_eta_coordinates(eta_max=3.0, n_eta=9, distribution="cosine")

        assert eta[0] == pytest.approx(0.0)
        assert eta[-1] == pytest.approx(3.0)
        assert np.all(np.diff(eta) > 0.0)

    def test_tanh_distribution_monotonic_and_bounded(self) -> None:
        """Tanh distribution should be monotone with endpoint inclusion."""
        eta = build_eta_coordinates(
            eta_max=4.0,
            n_eta=21,
            distribution="tanh",
            eta_stretch=3.0,
        )

        assert eta[0] == pytest.approx(0.0)
        assert eta[-1] == pytest.approx(4.0)
        assert np.all(np.diff(eta) > 0.0)

    def test_tanh_more_aggressive_near_wall_than_uniform(self) -> None:
        """For a given n_eta, tanh should place the 2nd point closer to the wall than uniform."""
        eta_uniform = build_eta_coordinates(
            eta_max=1.0,
            n_eta=201,
            distribution="uniform",
        )
        eta_tanh = build_eta_coordinates(
            eta_max=1.0,
            n_eta=201,
            distribution="tanh",
            eta_stretch=3.0,
        )

        # compare first off-wall sample location
        assert eta_tanh[1] < eta_uniform[1]

    def test_invalid_eta_max_raises(self) -> None:
        """Non-positive eta_max should fail."""
        with pytest.raises(ValueError, match="eta_max must be positive"):
            build_eta_coordinates(eta_max=0.0, n_eta=5, distribution="uniform")

    def test_invalid_eta_stretch_raises(self) -> None:
        """Non-positive eta_stretch should fail."""
        with pytest.raises(ValueError, match="eta_stretch must be positive"):
            build_eta_coordinates(
                eta_max=1.0,
                n_eta=5,
                distribution="tanh",
                eta_stretch=0.0,
            )
