# lst-tools

`lst-tools` is a Python toolkit for pre- and post-processing of Linear Stability Theory (LST) analyses of high-speed boundary layers.

## Features

- **CLI**: Streamlined workflow from the terminal
- **API**: Python interface for scripting and automation
- **HPC**: Job script generation for cluster environments
- **Post-processing**: Ridge-line maxima extraction and 3-D tracking volume assembly

## Quick Start

[Download PDF Documentation](https://uahypersonics.github.io/lst-tools/pdf/lst-tools-documentation.pdf){ .md-button .md-button--primary }

## PDF Downloads

- Latest PDF (stable URL):
	https://uahypersonics.github.io/lst-tools/pdf/lst-tools-documentation.pdf
- Versioned PDF (tag builds):
	https://uahypersonics.github.io/lst-tools/pdf/lst-tools-vX.Y.Z.pdf

For example, tag `v0.1.3` publishes:
`https://uahypersonics.github.io/lst-tools/pdf/lst-tools-v0.1.3.pdf`

Install the package (see [Installation](installation.md) for detailed instructions):

```bash
pip install lst-tools
```

```bash
# initialize a config file
lst-tools init

# convert base flow to LASTRAC format
lst-tools lastrac

# set up and run parsing sweep
lst-tools setup parsing --auto-fill

# set up and run tracking
lst-tools setup tracking --auto-fill

# post-process tracking results (maxima + volume)
lst-tools process tracking --interpolate

# optional: set up and process spectra
lst-tools setup spectra
lst-tools process spectra

# optional: clean generated artifacts
lst-tools clean tracking --force
```

`lst-tools` can also be used as a Python library. See the [API Reference](api/index.md) for details.

## License

BSD-3-Clause. See [LICENSE](https://github.com/uahypersonics/lst-tools/blob/main/LICENSE) for details.
