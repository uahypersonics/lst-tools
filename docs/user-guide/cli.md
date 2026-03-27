# CLI

`lst-tools` provides the `lst-tools` command-line interface for the full LST workflow.

## General Usage

```bash
lst-tools [options] <subcommand> [<args>]
```

Global options:

```bash
lst-tools --version
lst-tools --verbose
lst-tools --debug
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

# pre-populate geometry defaults
lst-tools init --geometry cone

# specify output path
lst-tools init --out myconfig.cfg

# merge with flow_conditions.dat
lst-tools init --flow flow_conditions.dat

# overwrite existing config
lst-tools init --force
```

### `lastrac`

Convert the HDF5 base flow to LASTRAC format:

```bash
lst-tools lastrac

# explicit config path
lst-tools lastrac --cfg myconfig.cfg
```

### `setup parsing`

Generate input decks for the LST parsing (initial frequency/wavenumber sweep):

```bash
lst-tools setup parsing

# auto-fill parameters (x_s, x_e, i_step, f_min, f_max, d_f, etc.) from the meanflow
lst-tools setup parsing --auto-fill

# auto-fill and overwrite existing values
lst-tools setup parsing --auto-fill --force

# write input deck into another directory/name
lst-tools setup parsing --out runs --name lst_input.dat
```

### `setup tracking`

Set up tracking calculations (directory structure and input decks):

```bash
lst-tools setup tracking

# auto-fill parameters from the meanflow
lst-tools setup tracking --auto-fill

# auto-fill and overwrite existing values
lst-tools setup tracking --auto-fill --force

# set a fixed initialization frequency
lst-tools setup tracking --finit 120000.0
```

### `setup spectra`

Generate input decks for spectral analysis at multiple streamwise locations:

```bash
lst-tools setup spectra

# explicit config path
lst-tools setup spectra --cfg myconfig.cfg
```

### `process tracking`

Post-process tracking calculation results (ridge maxima + optional 3-D volume):

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

### `process spectra`

Post-process spectra calculation results:

```bash
lst-tools process spectra
```

### `hpc`

Generate run scripts for HPC systems:

```bash
lst-tools hpc

# explicit config path
lst-tools hpc --cfg myconfig.cfg
```

### `info`

Inspect a LASTRAC meanflow binary:

```bash
lst-tools info meanflow.bin
```

### `clean parsing`

Remove parsing-generated artifacts from one directory:

```bash
lst-tools clean parsing --dir .
lst-tools clean parsing --dir . --force
```

### `clean tracking`

Remove solver artifacts from tracking case directories:

```bash
# clean all kc_* directories in current directory
lst-tools clean tracking --force

# clean selected directories only
lst-tools clean tracking --dir kc_10pt00 --dir kc_20pt00 --force
```

### `clean spectra`

Remove spectra setup outputs:

```bash
lst-tools clean spectra --dir . --force
```

## Typical Workflow

```bash
# 1. initialize config
lst-tools init --geometry cone

# 2. convert base flow to LASTRAC format
lst-tools lastrac

# 3. run parsing sweep
lst-tools setup parsing --auto-fill

# 4. set up and run tracking
lst-tools setup tracking --auto-fill

# 5. post-process tracking results
lst-tools process tracking --interpolate

# 6. set up and run spectra
lst-tools setup spectra

# 7. post-process spectra results
lst-tools process spectra
```
