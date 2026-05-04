# Postprocessing and Cleanup

This phase turns completed solver output into processed results, quick-look
figures, or a cleaned working directory.

| Stage | What it does | Typical output |
|---|---|---|
| Visualization | Renders quick preview PNGs from parsing or tracking output | contour PNGs |
| Processing | Post-processes tracking or spectra outputs | processed tracking or spectra results |
| Cleaning | Removes generated files and case artifacts | reduced working directory clutter |

Visualization usually follows parsing or tracking. Processing usually follows
tracking or spectra.

## Spectra Branch

Use spectra subcommand to obtain the full eigenvalue spectrum at a fixed location for a specific frequency and wavenumber.

```bash
lst-tools setup spectra
lst-tools process spectra
```

See [CLI Usage](../user-guide/cli-usage.md) or
[API Usage](../user-guide/api-usage.md) for runnable steps.