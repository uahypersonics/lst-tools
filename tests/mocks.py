import pathlib
import warnings
import pytest

MOCK_FLOW_CONDITIONS_DAT = pathlib.Path.cwd() / "tests" / "mock_flow_conditions.dat"

MOCK_CONFIG_CFG = pathlib.Path.cwd() / "tests" / "mock_config.cfg"

MOCK_BASE_FLOW_OGIVE_HDF5 = pathlib.Path.cwd() / "tests" / "mock_base_flow_ogive.hdf5"

_REQUIRED_FILES = (MOCK_BASE_FLOW_OGIVE_HDF5, MOCK_CONFIG_CFG, MOCK_FLOW_CONDITIONS_DAT)

if "_MOCKS_CHECKED" not in globals():
    global _MOCKS_CHECKED
    _MOCKS_CHECKED = True
    if any([not file.exists() for file in _REQUIRED_FILES]):
        _separator = "\n\t"
        warnings.warn(
            f"""

    Missing test dependency files. Some tests will be skipped.\n\tMissing required file(s):{"".join([f"{_separator}- {str(file)}" for file in _REQUIRED_FILES if not file.exists()])}
    """,
            RuntimeWarning,
        )


def skip_if_missing(*deps: pathlib.Path):
    _separator = "\n\t- "
    if any([not file.exists() for file in deps]):
        pytest.skip(
            f"""Missing required file(s): {_separator}{_separator.join([str(dep) for dep in deps])}"""
        )
