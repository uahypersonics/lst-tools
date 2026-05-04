# HPC

Edit this section only for cluster runs or scheduler-script generation.

| Key | Type | Default | Description |
|---|---|---|---|
| `account` | `str` | - | Account/allocation name |
| `nodes` | `int` | - | Number of nodes |
| `time` | `str` | - | Wall time |
| `partition` | `str` | - | Partition/queue name |
| `extra_env` | `table` | - | Extra environment variables injected into run scripts |

The setup commands write scheduler scripts automatically when the scheduler
is detected. Use this section to control the script contents.
