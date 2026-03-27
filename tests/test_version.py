"""Tests for generated package version module."""

from __future__ import annotations

import importlib.util

import lst_tools
import pytest


def test_public_package_version_exists() -> None:
    """Top-level package should always expose __version__."""
    assert isinstance(lst_tools.__version__, str)
    assert len(lst_tools.__version__) > 0


def test_version_exports_are_present() -> None:
    """Private generated _version exports are consistent when module exists."""
    spec = importlib.util.find_spec("lst_tools._version")
    if spec is None:
        pytest.skip("lst_tools._version is not available in this build")

    import lst_tools._version as version_mod

    assert isinstance(version_mod.__version__, str)
    assert version_mod.__version__ == version_mod.version
    assert version_mod.__version_tuple__ == version_mod.version_tuple
    assert version_mod.__commit_id__ == version_mod.commit_id


def test_version_all_symbols_exist() -> None:
    """Every symbol listed in _version.__all__ should resolve when available."""
    spec = importlib.util.find_spec("lst_tools._version")
    if spec is None:
        pytest.skip("lst_tools._version is not available in this build")

    import lst_tools._version as version_mod

    for name in version_mod.__all__:
        assert hasattr(version_mod, name)
