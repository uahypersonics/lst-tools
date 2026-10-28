# Meanflow Conversion

Set parameters to control the HDF5 to `meanflow.bin` conversion step.

| Key | Type | Default | Description |
|---|---|---|---|
| `i_s` | `int` | `0` | Start index |
| `i_e` | `int` | - | End index |
| `d_i` | `int` | `1` | Index stride |
| `set_v_zero` | `bool` | `true` | Zero out wall-normal velocity |
| `nondimensionalize` | `bool` | `true` | Divide the profiles by the reference state |
| `reference` | `str` | `"freestream"` | Reference state: `"freestream"` or `"custom"` |
| `uvel_ref` | `float` | - | Custom reference velocity [m/s] (`reference = "custom"` only) |
| `temp_ref` | `float` | - | Custom reference temperature [K] (`reference = "custom"` only) |
| `dens_ref` | `float` | - | Custom reference density [kg/m³] (`reference = "custom"` only) |
| `re1_ref` | `float` | - | Custom unit Reynolds number [1/m] (`reference = "custom"` only) |

Adjust `i_s`, `i_e` and `d_i` when the converted meanflow should use a specific
streamwise slice range or stride.

## Reference state

The conversion is the only step that nondimensionalizes: `lst-tools extract` always
writes dimensional profiles. One reference state is used for everything:

- the profiles are divided by it: velocity by `uvel`, temperature by `temp`, and
  pressure by `dens * uvel²`
- the same values go into every station header of `meanflow.bin`
  (`stat_uvel`, `stat_temp`, `stat_dens`, `re1`), and the reference pressure into the
  file header

So the scale LASTRAC reads from each station header is always the scale the
profiles actually have.

| `reference` | Values used |
|---|---|
| `"freestream"` | `[flow_conditions]` `uvel_inf`, `temp_inf`, `dens_inf`, `re1`, and `pres_inf` for the file header |
| `"custom"` | `uvel_ref`, `temp_ref`, `dens_ref`, `re1_ref` above; the file-header pressure is `dens_ref * rgas * temp_ref` |

For a custom reference, `re1_ref` must be the unit Reynolds number of that state,
`dens_ref * uvel_ref / mu(temp_ref)`, or the stability analysis runs at the wrong
Reynolds number. All reference values must be positive.

When the input is already nondimensional, set `nondimensionalize = false`; the
conversion refuses to scale nondimensional input a second time. The reference
state is still written into the headers, so it must describe the scale the input
was nondimensionalized by. The source HDF5 is not changed.

## Example

```toml
[meanflow_conversion]
i_s        = 0
i_e        = 200
d_i        = 1
set_v_zero = true
nondimensionalize = true
reference = "freestream"
```

With a custom reference state:

```toml
[meanflow_conversion]
i_s        = 0
i_e        = 200
d_i        = 1
set_v_zero = true
nondimensionalize = true
reference = "custom"
uvel_ref = 800.0
temp_ref = 120.0
dens_ref = 0.03
re1_ref  = 5.0e6
```
