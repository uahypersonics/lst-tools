# CLI

`lst-tools` provides the `lst-tools` command-line interface for the full LST workflow.

## General Usage

```bash
lst-tools [options] <subcommand> [<args>]
```

Show version:

```bash
lst-tools --version
```

Get help for any subcommand:

```bash
lst-tools <subcommand> --help
```

## Subcommands

### `init`

Generate a default configuration file:

```bash
# create default lst.cfg
lst-tools init

# interactive mode (asks questions to pre-fill values)
lst-tools init -i

# specify output path
lst-tools init --out myconfig.cfg

# merge with flow_conditions.dat
lst-tools init --flow flow_conditions.dat
```

### `lastrac`

Convert the HDF5 base flow to LASTRAC format:

```bash
lst-tools lastrac
```

### `setup parsing`

Generate input decks for the LST parsing (initial frequency/wavenumber sweep):

```bash
lst-tools setup parsing

# auto-fill parameters (x_s, x_e, i_step, f_min, f_max, d_f, etc.) from the meanflow
lst-tools setup parsing --auto-fill

# auto-fill and overwrite existing values
lst-tools setup parsing --auto-fill --force
```

### `setup tracking`

Set up tracking calculations (directory structure and input decks):

```bash
lst-tools setup tracking

# auto-fill parameters from the meanflow
lst-tools setup tracking --auto-fill

# auto-fill and overwrite existing values
lst-tools setup tracking --auto-fill --force
```

### `setup spectra`

Generate input decks for spectral analysis at multiple streamwise locations:

```bash
lst-tools setup spectra
```

### `process tracking`

Post-process tracking calculation results:

```bash
lst-tools process tracking
```

### `process spectra`

Post-process spectra calculation results:

```bash
lst-tools process spectra
```

### `hpc`

Generate run scripts for HPC systems:

```bash
lst-tools hpc
```

## Typical Workflow

```bash
# 1. initialize config
lst-tools init -i

# 2. convert base flow to LASTRAC format
lst-tools lastrac

# 3. run parsing sweep
lst-tools setup parsing --auto-fill

# 4. set up and run tracking
lst-tools setup tracking --auto-fill

# 5. post-process tracking results
lst-tools process tracking

# 6. set up and run spectra
lst-tools setup spectra

# 7. post-process spectra results
lst-tools process spectra
```
