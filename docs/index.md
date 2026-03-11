# lst-tools

`lst-tools` is a Python toolkit for pre- and post-processing of Linear Stability Theory (LST) analyses of high-speed boundary layers.

## Features

- **CLI**: Streamlined workflow from the terminal
- **API**: Python interface for scripting and automation
- **HPC**: Job script generation for cluster environments

## Quick Start

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

# post-process results
lst-tools process tracking
```

`lst-tools` can also be used as a Python library. See the [API Reference](api/index.md) for details.

## License

BSD-3-Clause. See [LICENSE](https://github.com/uahypersonics/lst-tools/blob/main/LICENSE) for details.
