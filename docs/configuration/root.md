# Root

Edit these keys first.

| Key | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | `"base_flow.hdf5"` | Path to the HDF5 base flow file |
| `lst_exe` | `str` | `"lst.x"` | Path to the LST solver executable |

`input_file` controls which HDF5 base flow is converted during meanflow
preparation. `lst_exe` points to the external solver that runs after setup.
