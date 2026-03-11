# Config Files

`lst_tools` uses a TOML-style configuration file (`lst.cfg`) to control the LST workflow.

## Creating a Config File

Generate a default configuration:

```bash
lst-tools init
```

Or interactively:

```bash
lst-tools init -i
```

## Configuration Sections

### Top-Level

| Key | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | `"base_flow.hdf5"` | Path to the HDF5 base flow file |
| `debug` | `bool` | `false` | Enable debug output |
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
| `type` | `str` | — | Solver type |
| `is_simplified` | `bool` | `false` | Use simplified equations |
| `spatial_temporal` | `int` | `1` | Spatial (1) or temporal (0) analysis |
| `energy_formulation` | `int` | `1` | Energy equation formulation |

### `[lst.params]`

LST calculation parameters:

| Key | Type | Default | Description |
|---|---|---|---|
| `ny` | `int` | `150` | Number of wall-normal grid points |
| `tol_lst` | `float` | `1e-5` | Convergence tolerance |
| `max_iter` | `int` | `15` | Maximum iterations |
| `x_s` | `float` | — | Start streamwise location |
| `x_e` | `float` | — | End streamwise location |
| `f_min` | `float` | — | Minimum frequency [Hz] |
| `f_max` | `float` | — | Maximum frequency [Hz] |
| `d_f` | `float` | — | Frequency step [Hz] |

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

## Programmatic Access

```python
from lst_tools import read_config, validate_config, write_config

# read
config = read_config("lst.cfg")

# validate
validate_config(config)

# modify and write back
config["lst"]["params"]["ny"] = 200
write_config(config, "lst_modified.cfg")
```
