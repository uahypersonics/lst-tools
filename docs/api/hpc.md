# HPC

HPC job script generation for cluster environments.

## Functions

### `hpc_configure`

```python
from lst_tools import hpc_configure

hpc_cfg = hpc_configure(config)
```

Build an HPC configuration object from the project config.

::: lst_tools.hpc.hpc_configure

### `script_build`

```python
from lst_tools import script_build

script_build(hpc_cfg)
```

Generate job submission scripts from the HPC configuration.

::: lst_tools.hpc.script_build

### `HPCcfg`

```python
from lst_tools import HPCcfg
```

HPC configuration dataclass.

::: lst_tools.hpc.HPCcfg
