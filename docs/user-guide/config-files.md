# Config Files

`lst_tools` uses a TOML-style configuration file (`lst.cfg`) to control the LST workflow.

## Creating a Config File

Generate a default configuration:

```bash
lst-tools init
```

Geometry presets are available:

```bash
lst-tools init --geometry cone
```

## Configuration Sections

### Top-Level

| Key | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | `"base_flow.hdf5"` | Path to the HDF5 base flow file |
| `lst_exe` | `str` | `"lst.x"` | Path to the LST solver executable |

### `[flow_conditions]`

Flow conditions for the LST analysis:

| Key | Type | Default | Description |
|---|---|---|---|
| `mach` | `float` | — | Freestream Mach number |
| `re1` | `float` | — | Unit Reynolds number [1/m] |
| `gamma` | `float` | `1.4` | Ratio of specific heats |
| `cp` | `float` | `1005.025` | Specific heat at constant pressure [J/(kg K)] |
| `cv` | `float` | `717.875` | Specific heat at constant volume [J/(kg K)] |
| `rgas` | `float` | `287.15` | Specific gas constant [J/(kg K)] |
| `pres_0` | `float` | — | Stagnation pressure [Pa] |
| `temp_0` | `float` | — | Stagnation temperature [K] |
| `pres_inf` | `float` | — | Freestream pressure [Pa] |
| `temp_inf` | `float` | — | Freestream temperature [K] |
| `dens_inf` | `float` | — | Freestream density [kg/m^3] |
| `uvel_inf` | `float` | — | Freestream velocity [m/s] |
| `visc_law` | `int` | `0` | Viscosity law index |

### `[geometry]`

Geometry description:

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | — | Geometry kind (e.g., cone, flat plate) |
| `theta_deg` | `float` | — | Half-angle [deg] |
| `r_nose` | `float` | — | Nose radius [m] |
| `l_ref` | `float` | `1` | Reference length [m] |
| `is_body_fitted` | `bool` | `false` | Whether the grid is body-fitted |

### `[meanflow_conversion]`

Controls for base flow conversion:

| Key | Type | Default | Description |
|---|---|---|---|
| `i_s` | `int` | `0` | Start index |
| `i_e` | `int` | — | End index |
| `d_i` | `int` | `1` | Index stride |
| `set_v_zero` | `bool` | `true` | Zero out wall-normal velocity |

### `[lst.solver]`

LST solver settings:

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | `int` | `1` | Solver type |
| `is_simplified` | `bool` | `false` | Use simplified equations |
| `alpha_i_threshold` | `float` | `-100.0` | Threshold used in alpha_i filtering |
| `generalized` | `int` | `0` | Generalized eigenproblem switch |
| `spatial_temporal` | `int` | `1` | Spatial (1) or temporal (0) analysis |
| `energy_formulation` | `int` | `1` | Energy equation formulation |

### `[lst.options]`

Additional solver options:

| Key | Type | Default | Description |
|---|---|---|---|
| `geometry_switch` | `int` | — | Geometry mode switch |
| `longitudinal_curvature` | `int` | `0` | Enable longitudinal curvature terms |

### `[lst.params]`

LST calculation parameters:

| Key | Type | Default | Description |
|---|---|---|---|
| `ny` | `int` | `150` | Number of wall-normal grid points |
| `yl_in` | `float` | `0.0` | Initial wall-normal location |
| `tol_lst` | `float` | `1e-5` | Convergence tolerance |
| `max_iter` | `int` | `15` | Maximum iterations |
| `x_s` | `float` | — | Start streamwise location |
| `x_e` | `float` | — | End streamwise location |
| `i_step` | `int` | — | Streamwise index stride |
| `tracking_dir` | `int` | `1` | Tracking direction flag |
| `f_min` | `float` | — | Minimum frequency [Hz] |
| `f_max` | `float` | — | Maximum frequency [Hz] |
| `d_f` | `float` | — | Frequency step [Hz] |
| `f_init` | `float` | `0.0` | Tracking initialization frequency [Hz] |
| `beta_s` | `float` | — | Start spanwise wavenumber |
| `beta_e` | `float` | — | End spanwise wavenumber |
| `d_beta` | `float` | — | Spanwise wavenumber step |
| `beta_init` | `float` | `0.0` | Tracking initialization spanwise wavenumber |
| `alpha_0` | `complex` | `(0, 0)` | Initial complex alpha guess |

### `[lst.io]`

LST input/output file paths:

| Key | Type | Default | Description |
|---|---|---|---|
| `baseflow_input` | `str` | `"meanflow.bin"` | Converted base flow file |
| `solution_output` | `str` | `"growth_rate.dat"` | LST output file |

### `[hpc]`

HPC job settings:

| Key | Type | Default | Description |
|---|---|---|---|
| `account` | `str` | — | Account/allocation name |
| `nodes` | `int` | — | Number of nodes |
| `time` | `str` | — | Wall time |
| `partition` | `str` | — | Partition/queue name |
| `extra_env` | `table` | — | Extra environment variables injected into run scripts |

### `[processing.tracking]`

Tracking post-processing controls:

| Key | Type | Default | Description |
|---|---|---|---|
| `interpolate` | `bool` | `false` | Enable parabolic peak interpolation |
| `gate_tol` | `float` | `0.10` | Ridge-tracking gating tolerance |
| `min_valid` | `int` | `40` | Minimum valid points for ridge acceptance |
| `peak_order` | `int` | `1` | Local peak search order |

### `[processing.spectra]`

Spectra post-processing controls:

| Key | Type | Default | Description |
|---|---|---|---|
| `alpr_min` | `float` | — | Lower bound for `re(alpha)` gating |
| `alpr_max` | `float` | — | Upper bound for `re(alpha)` gating |
| `alpi_min` | `float` | — | Lower bound for `-im(alpha)` gating |
| `alpi_max` | `float` | — | Upper bound for `-im(alpha)` gating |
| `branch_gate` | `float` | `0.25` | Maximum normalized jump allowed during branch matching |
| `branch_min_points` | `int` | `2` | Minimum points required to keep a tracked branch |
| `isolation_k` | `int` | `3` | Neighbor count used in the isolation score |
| `isolation_threshold` | `float` | — | Minimum isolation score required for classification |
| `classify_min_points` | `int` | `3` | Minimum isolated points required to keep a classified branch |

### `[processing.parsing]`

Reserved for parsing-specific post-processing controls.

## Programmatic Access

```python
from lst_tools.config import read_config, write_config

# read as typed dataclass
config = read_config("lst.cfg")

# modify values
config.lst.params.ny = 200

# write to TOML
write_config("lst_modified.cfg", overwrite=True, cfg_data=config.to_dict())
```
