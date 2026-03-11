# Installation

## Requirements

- Python 3.10 or later

## From PyPI

```bash
pip install lst-tools
```

## From Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/uahypersonics/lst-tools.git
cd lst-tools
pip install -e .
```

## Development Installation

For development, install with the `dev` extras:

```bash
pip install -e ".[dev]"
```

This includes:

- [pytest](https://docs.pytest.org/) and [pytest-cov](https://pytest-cov.readthedocs.io/) for testing
- [ruff](https://docs.astral.sh/ruff/) for linting

## Verify Installation

```python
import lst_tools
print(lst_tools.__version__)
```

Or from the command line:

```bash
lst-tools --version
```
