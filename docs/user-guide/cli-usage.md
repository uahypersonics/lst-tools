# CLI Usage

This page shows the practical command-line workflow. Use
[Workflow](../workflow/index.md) for the phase layout and
[CLI Reference](../cli-reference/index.md) for full command lookup.

The CLI is self-discoverable:

```bash
lst-tools --help
lst-tools -V
lst-tools setup parsing --help
```

## Initialization and Meanflow Preparation

Create the starting config file:

```bash
lst-tools init --geometry cone
```

Add `--flow flow_conditions.dat` when flow conditions are already available.

Prepare `meanflow.bin` for `lst.x`:

```bash
lst-tools lastrac
```

`lst-tools --debug lastrac` writes conversion diagnostics and a Tecplot debug
snapshot to `./debug/`.

Inspect `meanflow.bin` when needed:

```bash
lst-tools info meanflow.bin
```

## Setup Runs

Prepare the broad sweep input deck:

```bash
lst-tools setup parsing --auto-fill
```

`--auto-fill` fills sweep bounds or index ranges when they are unset in
`lst.cfg`.

Prepare tracking cases from the parsing solution:

```bash
lst-tools setup tracking --auto-fill
```

Prepare spectra cases when the fixed-station spectra branch is needed:

```bash
lst-tools setup spectra
```

`lst-tools` prepares input decks and scripts; run `lst.x` separately.

When an HPC scheduler is configured, the setup phase also writes scheduler
scripts for generated decks and case directories. `lst-tools hpc` regenerates
the run script for the current directory when needed.

## Postprocessing and Cleanup

After the tracking solves finish, process the results:

```bash
lst-tools process tracking --interpolate
lst-tools visualize tracking
```

Repeat `--dir` to process selected `kc_*` directories only.

Render parsing preview PNGs after parsing when needed:

```bash
lst-tools visualize parsing
```

`visualize` renders quick contour PNGs on the remote machine. `--out` writes
the PNGs to a specific directory.

Post-process spectra output when needed:

```bash
lst-tools process spectra
```

Clean generated artifacts when they are no longer needed:

```bash
lst-tools clean parsing --force
lst-tools clean tracking --force
lst-tools clean spectra --force
```