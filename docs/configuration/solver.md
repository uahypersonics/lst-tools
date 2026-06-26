# Solver

Set solver mode, run parameters, and file paths.

## `[lst.solver]`

Set solver formulation choices.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | `int` | `1` | Solver type |
| `is_simplified` | `bool` | `false` | Use simplified equations |
| `alpha_i_threshold` | `float` | `-100.0` | Threshold used in alpha_i filtering |
| `generalized` | `int` | `0` | Generalized eigenproblem switch |
| `spatial_temporal` | `int` | `1` | Spatial (1) or temporal (0) analysis |
| `energy_formulation` | `int` | `1` | Energy equation formulation |

## `[lst.options]`

Set geometry-specific or curvature-related solver switches.

| Key | Type | Default | Description |
|---|---|---|---|
| `geometry_switch` | `int` | - | Geometry mode switch |
| `longitudinal_curvature` | `int` | `0` | Enable longitudinal curvature terms |

## `[lst.params]`

Set LST parameters

| Key | Type | Default | Description |
|---|---|---|---|
| `ny` | `int` | `150` | Number of wall-normal grid points |
| `yl_in` | `float` | `0.0` | Initial wall-normal location |
| `tol_lst` | `float` | `1e-5` | Convergence tolerance |
| `max_iter` | `int` | `15` | Maximum iterations |
| `x_s` | `float` | - | Start streamwise location |
| `x_e` | `float` | - | End streamwise location |
| `i_step` | `int` | - | Streamwise index stride |
| `tracking_dir` | `int` | `1` | Tracking direction flag |
| `f_min` | `float` | - | Minimum frequency [Hz] |
| `f_max` | `float` | - | Maximum frequency [Hz] |
| `d_f` | `float` | - | Frequency step [Hz] |
| `f_init` | `float` | `0.0` | Tracking initialization frequency [Hz] |
| `beta_s` | `float` | - | Start spanwise wavenumber |
| `beta_e` | `float` | - | End spanwise wavenumber |
| `d_beta` | `float` | - | Spanwise wavenumber step |
| `beta_init` | `float` | `0.0` | Tracking initialization spanwise wavenumber |
| `alpha_0` | `complex` | `(0, 0)` | Initial complex alpha guess |

## `[lst.io]`

Set file names (recommended: keep defaults)

| Key | Type | Default | Description |
|---|---|---|---|
| `baseflow_input` | `str` | `"meanflow.bin"` | Converted base flow file |
| `solution_output` | `str` | `"growth_rate.dat"` | LST output file |

## Example

```toml
[lst.solver]
type               = 1
spatial_temporal   = 1
energy_formulation = 1

[lst.params]
ny     = 150
x_s    = 0.10
x_e    = 0.60
i_step = 1
f_min  = 50000
f_max  = 400000
d_f    = 10000

[lst.io]
baseflow_input  = "meanflow.bin"
solution_output = "growth_rate.dat"
```
