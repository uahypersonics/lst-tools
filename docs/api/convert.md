# Convert

Format conversion utilities for base flow data.

## Functions

### `convert_meanflow`

```python
from lst_tools import convert_meanflow

convert_meanflow(grid, flow, "meanflow.bin", cfg=config)
```

Convert an HDF5 base flow to the Fortran binary format used by the LST solver.

::: lst_tools.convert.convert_meanflow

### `generate_lst_input_deck`

```python
from lst_tools import generate_lst_input_deck

generate_lst_input_deck(cfg=config, out_path="lst_input.dat")
```

Generate an input deck for the LST solver.

::: lst_tools.convert.generate_lst_input_deck

convert_meanflow(grid, flow, "meanflow.bin", cfg=config)
```

Use this path when base-flow data is already available in Python. For the
exact data requirements and helper objects, use the
[Convert](../api/convert.md) and [Grid & Flow](../api/core.md) reference
pages.

## Typical API Automation Pattern

Use one Python driver to edit config values and prepare several runs.

```python
from lst_tools import read_config, parsing_setup, write_config

config = read_config("lst.cfg")

for beta_max in (0.02, 0.04, 0.06):
	config.lst.params.beta_e = beta_max
	out_name = f"lst_input_beta_{beta_max:.2f}.dat"
	parsing_setup(cfg=config, out_name=out_name, auto_fill=True)

write_config("lst.cfg", overwrite=True, cfg_data=config.to_dict())
```

This pattern is usually the main reason to switch from CLI to API.

## Where the API Reference Fits

Use this page for practical workflow patterns. Use the
[API Reference](../api/index.md) when you need exact signatures, class
definitions, module organization, or lower-level utility details.
