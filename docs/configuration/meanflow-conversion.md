# Meanflow Conversion

Use these values to control the HDF5 to `meanflow.bin` conversion step.

| Key | Type | Default | Description |
|---|---|---|---|
| `i_s` | `int` | `0` | Start index |
| `i_e` | `int` | - | End index |
| `d_i` | `int` | `1` | Index stride |
| `set_v_zero` | `bool` | `true` | Zero out wall-normal velocity |

Adjust these fields when the converted meanflow should use a specific streamwise
slice range or stride.
