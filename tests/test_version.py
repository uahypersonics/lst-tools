"""Tests for generated package version module."""

from __future__ import annotations

import lst_tools._version as version_mod


def test_version_exports_are_present() -> None:
    """Public version exports should exist and be internally consistent."""
    assert isinstance(version_mod.__version__, str)
    assert version_mod.__version__ == version_mod.version
    assert version_mod.__version_tuple__ == version_mod.version_tuple
    assert version_mod.__commit_id__ == version_mod.commit_id


def test_version_all_symbols_exist() -> None:
    """Every symbol listed in __all__ should resolve on the module."""
    for name in version_mod.__all__:
        assert hasattr(version_mod, name)
