# Installation

## Requirements

- Python **3.10** or later
- The external **`lst.x`** solver on your `PATH`
  (set its location in `lst.cfg` via `lst_exe`)
- Optional: visualization dependency for contour plotting

## From PyPI

```bash
pip install lst-tools
```

## From Source

```bash
git clone https://github.com/uahypersonics/lst-tools.git
cd lst-tools
pip install -e .
```

## Optional Extras

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

Includes:

- [pytest](https://docs.pytest.org/) and [pytest-cov](https://pytest-cov.readthedocs.io/) for testing
- [ruff](https://docs.astral.sh/ruff/) for linting

For building the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Verify Installation

```bash
lst-tools --version
```

Or in Python:

```python
import lst_tools
print(lst_tools.__version__)
```
