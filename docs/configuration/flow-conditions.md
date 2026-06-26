# Flow Conditions

Set flow conditions values from case metadata or `flow_conditions.dat`.

| Key | Type | Default | Description |
|---|---|---|---|
| `mach` | `float` | - | Freestream Mach number |
| `re1` | `float` | - | Unit Reynolds number [1/m] |
| `gamma` | `float` | `1.4` | Ratio of specific heats |
| `cp` | `float` | `1005.025` | Specific heat at constant pressure [J/(kg K)] |
| `cv` | `float` | `717.875` | Specific heat at constant volume [J/(kg K)] |
| `rgas` | `float` | `287.15` | Specific gas constant [J/(kg K)] |
| `pres_0` | `float` | - | Stagnation pressure [Pa] |
| `temp_0` | `float` | - | Stagnation temperature [K] |
| `pres_inf` | `float` | - | Freestream pressure [Pa] |
| `temp_inf` | `float` | - | Freestream temperature [K] |
| `dens_inf` | `float` | - | Freestream density [kg/m^3] |
| `uvel_inf` | `float` | - | Freestream velocity [m/s] |
| `visc_law` | `int` | `0` | Viscosity law index |

## Example

```toml
[flow_conditions]
mach     = 6.0
re1      = 1.2e7
gamma    = 1.4
temp_inf = 50.0
visc_law = 0
```
