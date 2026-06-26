# CLI Reference

Detailed overview of the Command Line Interface (CLI). See
[Workflow](../workflow/index.md) and [CLI Usage](../user-guide/cli-usage.md)
for the overall sequence.

## General Usage

```bash
lst-tools [options] <subcommand> [<args>]
```

Global options:

```bash
lst-tools --help
lst-tools -V
lst-tools --version
lst-tools --verbose
lst-tools --debug
```

Some options also provide shorthand forms. For example, `-V` and
`--version` both print the installed version.

Show top-level help:

```bash
lst-tools --help
```

Show version:

```bash
lst-tools -V
lst-tools --version
```

Get help for any subcommand:

```bash
lst-tools <subcommand> --help
```

Global options such as `--verbose`/`-v` and `--debug`/`-d` apply to the
command groups and subcommands below.

## Option Precedence

When a command supports both `lst.cfg` values and explicit CLI options, use
this rule:

1. Load defaults from `lst.cfg`.
2. Apply explicit CLI options for the current invocation.
3. Keep the CLI override local unless the command explicitly writes values
	 back to the config.

Practical implications:

- `--cfg` changes which config file is loaded; it does not override one field.
- Output-path options such as `--out` and `--name` affect the current run only.
- `setup parsing --auto-fill` and `setup tracking --auto-fill` can write
	derived values back into `lst.cfg` when a config path is available.
- Processing commands use config values as defaults, but explicit CLI flags
	such as `--interpolate` take precedence for that run.
- Selector flags such as `--maxima`, `--volume`, `--animate`, or `--branches`
	choose which actions run; they do not rewrite config values.

## Subcommands

### `lst-tools init`

| Command | Purpose |
|---|---|
| [init](init.md) | Create or seed `lst.cfg` |

### `lst-tools lastrac`

| Command | Purpose |
|---|---|
| [lastrac](lastrac.md) | Convert HDF5 base flow to `meanflow.bin` |

### `lst-tools extract`

| Command | Purpose |
|---|---|
| [extract](extract.md) | Extract wall-normal profiles from a CFD mesh |

### `lst-tools setup`

| Subcommand | Purpose |
|---|---|
| [parsing](setup-parsing.md) | Write the broad parsing input deck |
| [tracking](setup-tracking.md) | Build tracking case directories and decks |
| [spectra](setup-spectra.md) | Write fixed-station spectra decks |

### `lst-tools process`

| Subcommand | Purpose |
|---|---|
| [tracking](process-tracking.md) | Post-process tracking results |
| [spectra](process-spectra.md) | Post-process spectra results |

### `lst-tools visualize`

| Subcommand | Purpose |
|---|---|
| [parsing](visualize-parsing.md) | Render quick preview PNGs from parsing output |
| [tracking](visualize-tracking.md) | Render quick preview PNGs from tracking output |

### `lst-tools clean`

| Subcommand | Purpose |
|---|---|
| [parsing](clean-parsing.md) | Remove parsing artifacts |
| [tracking](clean-tracking.md) | Remove tracking artifacts |
| [spectra](clean-spectra.md) | Remove spectra artifacts |

### Utility Commands

| Command | Purpose |
|---|---|
| [hpc](hpc.md) | Generate or regenerate a run script |
| [info](info.md) | Inspect `meanflow.bin` metadata |