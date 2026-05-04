# `setup parsing`

Generate input decks for the LST parsing sweep.

Config values in `lst.cfg` provide the default parsing inputs. CLI options
override those defaults for the current invocation.

## Options

| Option | Meaning |
|---|---|
| `--cfg`, `-c` | Load a specific config file |
| `--out`, `-o` | Choose the output directory |
| `--name`, `-n` | Choose the output filename |
| `--auto-fill`, `-a` | Fill unset sweep values from `meanflow.bin` |
| `--force`, `-f` | Overwrite existing config values when used with `--auto-fill` |

`--auto-fill` can persist derived values such as sweep bounds and strides
back into `lst.cfg` when a config path is available.

```bash
lst-tools setup parsing

# auto-fill parameters (x_s, x_e, i_step, f_min, f_max, d_f, etc.) from the meanflow
lst-tools setup parsing --auto-fill

# auto-fill and overwrite existing values
lst-tools setup parsing --auto-fill --force

# write input deck into another directory/name
lst-tools setup parsing --out runs --name lst_input.dat
```