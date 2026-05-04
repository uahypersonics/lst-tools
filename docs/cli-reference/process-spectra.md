# `process spectra`

Post-process spectra calculation results.

`cfg.processing.spectra` provides gating and classification defaults.
Selector flags on the CLI choose which outputs run for the current
invocation.

## Options

| Option | Meaning |
|---|---|
| `--cfg`, `-c` | Load a specific config file |
| `--animate` | Run only raw spectra animation-file output |
| `--branches` | Run only branch-tracking output |
| `--classify` | Run only isolation-score-based branch classification output |

If no selector flags are provided, the command runs animation and branch
tracking by default. Classification is also included automatically when
`processing.spectra.isolation_threshold` is set in the config.

```bash
lst-tools process spectra
```