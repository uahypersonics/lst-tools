# lst-tools

`lst-tools` is a Python toolkit for reading, processing, and visualizing data from Linear Stability Theory (LST) analyses.

## Features

- **CLI**: Streamlined workflow from the terminal
- **API**: Python interface for scripting and automation
- **I/O**: Read/write Fortran binary, Tecplot ASCII, and LASTRAC formats
- **Geometry**: Curvature, surface angle, and curvilinear coordinate utilities
- **HPC**: Job script generation for cluster environments

## Quick Start

Install the package (see [Installation](installation.md) for detailed instructions):

```bash
pip install -e .
```

Use the `lst_tools` package (see [User Guide](user-guide/index.md) for detailed examples):

```python
import cfd_io
import lst_tools

# read a base flow via cfd-io
grid_raw, flow_raw, attrs = cfd_io.read_file("baseflow.hdf5")

# compute geometry
kappa = lst_tools.curvature(grid)
```

## CLI

```bash
# initialize a config file
lst-tools init

# convert base flow to LASTRAC format
lst-tools lastrac

# set up tracking calculations
lst-tools tracking

# post-process results
lst-tools tracking-process
```

## License

BSD-3-Clause. See [LICENSE](https://github.com/uahypersonics/lst-tools/blob/main/LICENSE) for details.
