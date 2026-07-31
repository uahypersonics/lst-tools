# `init`

Create a default configuration file.

## Options

| Option | Meaning |
|---|---|
| `--out`, `-o` | Write the generated config to a specific path |
| `--force`, `-f` | Overwrite an existing config file |
| `--merge` | Merge an existing config at `--out` into the generated scaffold |
| `--geometry`, `-g` | Pre-populate the config for a geometry preset |
| `--flow`, `-F` | Read defaults from `flow_conditions.dat` |

```bash
# create default lst.cfg
lst-tools init

# pre-populate geometry defaults
lst-tools init --geometry cone

# specify output path
lst-tools init --out myconfig.cfg

# merge with flow_conditions.dat
lst-tools init --flow flow_conditions.dat

# merge an existing config into the generated scaffold
lst-tools init --out myconfig.cfg --merge

# overwrite existing config
lst-tools init --force
```