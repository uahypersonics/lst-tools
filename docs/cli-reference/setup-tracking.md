# `setup tracking`

Set up tracking calculations, including directory structure and input decks.

Config values in `lst.cfg` provide the default tracking inputs. CLI options
override those defaults for the current invocation.

## Options

| Option | Meaning |
|---|---|
| `--cfg`, `-c` | Load a specific config file |
| `--auto-fill`, `-a` | Fill unset tracking sweep values with defaults |
| `--force`, `-f` | Overwrite existing config values when used with `--auto-fill` |
| `--finit` | Override the initialization frequency in Hz |
| `--debug`, `-d` | Global option that writes tracking diagnostics to `./debug/` |

`--auto-fill` can write updated sweep values back into `lst.cfg`. `--finit`
overrides the initialization frequency for the current run without changing
the config file.

```bash
lst-tools setup tracking

# auto-fill parameters from the meanflow
lst-tools setup tracking --auto-fill

# auto-fill and overwrite existing values
lst-tools setup tracking --auto-fill --force

# set a fixed initialization frequency
lst-tools setup tracking --finit 120000.0
```