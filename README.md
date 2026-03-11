# lst-tools

Pre- and post-processing toolkit for Linear Stability Theory (LST) analyses of high-speed boundary layers.

[![Test](https://github.com/uahypersonics/lst-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/uahypersonics/lst-tools/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/uahypersonics/lst-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/uahypersonics/lst-tools)
[![PyPI](https://img.shields.io/pypi/v/lst-tools)](https://pypi.org/project/lst-tools/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://uahypersonics.github.io/lst-tools/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Install

```bash
pip install lst-tools
```

## Quick Start

```bash
# create a default config for a cone geometry
lst-tools init --geometry cone

# prepare an HDF5 meanflow file before using lst-tools
lst-tools lastrac

# set up and run LST
lst-tools setup parsing --auto-fill

# set up tracking and spectra
lst-tools setup tracking --auto-fill
lst-tools setup spectra

# post-process results
lst-tools process tracking
lst-tools process spectra
```

Run `lst-tools --help` for a full list of commands.

## Documentation

Full documentation: https://uahypersonics.github.io/lst-tools

## Testing

```bash
pytest tests/ --cov --cov-report=term-missing -q
```

## Code Style

| Convention | Reference |
|---|---|
| [PEP 8](https://peps.python.org/pep-0008/) | Python standard style guide |
| [PEP 257](https://peps.python.org/pep-0257/) | Python standard docstring conventions |
| [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) | Google Python style guide |
| [Ruff](https://docs.astral.sh/ruff/) | Automated linting and formatting |

## Versioning & Releasing

This project uses [Semantic Versioning](https://semver.org/) (`vMAJOR.MINOR.PATCH`).

To publish a new version to [PyPI](https://pypi.org/project/lst-tools/):

1. Update the version in `pyproject.toml`
2. Commit and push to `main`
3. Tag and push:
   ```bash
   git tag -a vMAJOR.MINOR.PATCH -m "Release vMAJOR.MINOR.PATCH"
   git push origin vMAJOR.MINOR.PATCH
   ```

The GitHub Actions workflow will automatically build and publish to PyPI via Trusted Publishing.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
