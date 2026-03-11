# Convert

Format conversion utilities for base flow data.

## Functions

### `convert_meanflow`

```python
from lst_tools import convert_meanflow

convert_meanflow(grid, flow, config)
```

Convert an HDF5 base flow to the Fortran binary format used by the LST solver.

::: lst_tools.convert.convert_meanflow

### `generate_lst_input_deck`

```python
from lst_tools import generate_lst_input_deck

generate_lst_input_deck(config)
```

Generate an input deck for the LST solver.

::: lst_tools.convert.generate_lst_input_deck
