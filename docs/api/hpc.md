# HPC

HPC job script generation for cluster environments.

## Functions

### `hpc_configure`

```python
from lst_tools import hpc_configure

hpc_cfg = hpc_configure(config, set_defaults=True)
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

### `ResolvedJob`

```python
from lst_tools import ResolvedJob
```

Resolved scheduler-specific job settings used by script generation.

::: lst_tools.hpc.ResolvedJob

### `detect`

```python
from lst_tools.hpc import detect

env = detect()
```

Detect scheduler/environment information (for example SLURM or PBS context).

::: lst_tools.hpc.detect
