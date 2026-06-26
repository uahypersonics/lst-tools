# Installation

## From PyPI

```bash
pip install lst-tools
```

To upgrade an existing installation:

```bash
pip install --upgrade lst-tools
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
