# Meanflow Conversion

Set parameters to control the HDF5 to `meanflow.bin` conversion step.

| Key | Type | Default | Description |
|---|---|---|---|
| `i_s` | `int` | `0` | Start index |
| `i_e` | `int` | - | End index |
| `d_i` | `int` | `1` | Index stride |
| `set_v_zero` | `bool` | `true` | Zero out wall-normal velocity |
| `nondimensionalize` | `bool` | `false` | Scale dimensional flow fields by freestream references |

Adjust these fields when the converted meanflow should use a specific streamwise
slice range or stride.

When dimensional input is detected but this option is `false`, `lst-tools lastrac`
warns that dimensional values will be written. For the legacy `lst.x` workflow,
set it to `true` when nondimensional profiles are required. Velocity is scaled by
`uvel_inf`, temperature by `temp_inf`, and pressure by `dens_inf * uvel_inf²`;
ensure those freestream values match the HDF5 data. The source HDF5 is not changed.

## Example

```toml
[meanflow_conversion]
i_s        = 0
i_e        = 200
d_i        = 1
set_v_zero = true
nondimensionalize = true
```
