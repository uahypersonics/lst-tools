# `process tracking`

Post-process tracking calculation results.

`cfg.processing.tracking` provides the default processing settings. Explicit
CLI flags override those defaults for the current invocation.

## Options

| Option | Meaning |
|---|---|
| `--cfg`, `-c` | Load a specific config file |
| `--dir` | Process selected `kc_*` directories only; repeat the option to select more than one |
| `--maxima` | Run only ridge-line maxima extraction |
| `--volume` | Run only 3-D volume assembly |
| `--interpolate`, `--no-interpolate` | Enable or disable sub-grid peak interpolation explicitly |
| `--plain-output` | Use plain text progress output instead of rich progress bars |

When `--dir` is provided, volume assembly is disabled automatically because
volume output requires the full set of tracking slices.

Maxima outputs are grouped in the directory where you run the command. Each
detected mode gets its own folder, such as `alpi_max_mode_001/` or
`nfac_max_mode_001/`, and each processed `kc_*` case writes its extracted ridge
file into the matching mode folder.

```bash
lst-tools process tracking

# only maxima extraction
lst-tools process tracking --maxima

# only volume assembly
lst-tools process tracking --volume

# process selected kc directories (repeat --dir)
lst-tools process tracking --dir kc_10pt00 --dir kc_20pt00

# enable/disable sub-grid interpolation explicitly
lst-tools process tracking --interpolate
lst-tools process tracking --no-interpolate
```