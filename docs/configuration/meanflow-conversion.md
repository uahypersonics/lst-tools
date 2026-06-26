# Meanflow Conversion

Set parameters to control the HDF5 to `meanflow.bin` conversion step.

| Key | Type | Default | Description |
|---|---|---|---|
| `i_s` | `int` | `0` | Start index |
| `i_e` | `int` | - | End index |
| `d_i` | `int` | `1` | Index stride |
| `set_v_zero` | `bool` | `true` | Zero out wall-normal velocity |

Adjust these fields when the converted meanflow should use a specific streamwise
slice range or stride.

## Example

```toml
[meanflow_conversion]
i_s        = 0
i_e        = 200
d_i        = 1
set_v_zero = true
```
