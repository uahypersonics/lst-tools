# Setup Runs

This phase prepares the run artifacts that the external solver consumes.
Parsing and spectra can begin after meanflow preparation. Tracking begins
after parsing because it depends on parsing output.

| Stage | What it does | Main artifact |
|---|---|---|
| Setup Parsing | Defines the broad instability sweep | parsing input deck |
| Setup Spectra | Defines the fixed-station spectra branch | spectra input decks |
| Setup Tracking | Builds per-mode tracking cases from parsing output | `kc_*` case directories |

Scheduler scripts belong to this phase when an HPC scheduler is configured.
The optional HPC helper generates or regenerates run scripts for the current
directory.

## Tracking and Seed Tables

Seed-table generation belongs to the tracking stage. With seed-table
generation enabled in `lst.cfg`, `setup tracking` writes `seed_alpha.dat`
into each `kc_*` case directory.

```toml
[seed_table]
enabled = true
```

`lst-tools setup tracking` reads the parsing solution, harvests ridge-based
seed points, and writes those seeds as part of the tracking setup process.

See [CLI Usage](../user-guide/cli-usage.md) or
[API Usage](../user-guide/api-usage.md) for runnable sequences.