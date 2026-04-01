# lst-tools

Pre- and post-processing toolkit for Linear Stability Theory (LST) analyses of high-speed boundary layers.

[![Test](https://github.com/uahypersonics/lst-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/uahypersonics/lst-tools/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/uahypersonics/lst-tools/branch/main/graph/badge.svg)](https://codecov.io/gh/uahypersonics/lst-tools)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey)](https://zenodo.org/)
[![PyPI](https://img.shields.io/pypi/v/lst-tools)](https://pypi.org/project/lst-tools/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://uahypersonics.github.io/lst-tools/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg)](https://www.python.org/downloads/)
[![Cite](https://img.shields.io/badge/Cite-this%20repository-blue)](https://github.com/uahypersonics/lst-tools?tab=citations)
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

# visualize parsing/tracking outputs (via cfd-viz wrappers)
lst-tools visualize parsing
lst-tools visualize tracking
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

To publish a new version to [PyPI](https://pypi.org/project/lst-tools/) and GitHub Releases:

1. Commit and push to `main`
2. Create and push an annotated tag:
   ```bash
   git tag -a vMAJOR.MINOR.PATCH -m "Release vMAJOR.MINOR.PATCH"
   git push origin vMAJOR.MINOR.PATCH
   ```

The GitHub Actions workflow will automatically test, build, publish to PyPI via Trusted Publishing, and create a GitHub Release from the tag.

## Citation & Zenodo

This repository includes `CITATION.cff` and `.zenodo.json` metadata to support software citation.

To enable DOI minting for each release:

1. Log into Zenodo and authorize GitHub access.
2. In Zenodo GitHub settings, enable archiving for `uahypersonics/lst-tools`.
3. Trigger a new tagged release (`vMAJOR.MINOR.PATCH`).
4. Zenodo will archive that release and mint a DOI.

Recommended citation sources:

1. `CITATION.cff` in this repository for software metadata.
2. The Zenodo DOI landing page for release-specific archival citations.

## License

BSD-3-Clause. See [LICENSE](LICENSE) for details.
