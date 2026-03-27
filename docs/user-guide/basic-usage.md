# Basic Usage

This page walks through a practical end-to-end workflow.

## 1. Initialize Configuration

Create a baseline config file:

```bash
lst-tools init
```

Common variants:

```bash
# pre-populate geometry fields
lst-tools init --geometry cone

# seed from flow_conditions.dat if available
lst-tools init --flow flow_conditions.dat
```

## 2. Convert Meanflow for LASTRAC

Convert the HDF5 base flow in `input_file` to `meanflow.bin`:

```bash
lst-tools lastrac
```

If you want extra diagnostics:

```bash
lst-tools --debug lastrac
```

This also writes a Tecplot debug snapshot to `./debug/`.

## 3. Set Up Parsing

Generate the parsing input deck:

```bash
lst-tools setup parsing
```

If key sweep values are unset, auto-fill from `meanflow.bin`:

```bash
lst-tools setup parsing --auto-fill
```

## 4. Run the External LST Solver

`lst-tools` prepares input decks and scripts; solver execution is external.

- For a single parsing deck, run your local `lst.x` process as needed.
- For cluster usage, generate a scheduler script with:

```bash
lst-tools hpc
```

## 5. Set Up Tracking and Process Results

Set up tracking cases:

```bash
lst-tools setup tracking --auto-fill
```

After running tracking cases, process results:

```bash
# maxima + volume
lst-tools process tracking

# enable sub-grid peak interpolation
lst-tools process tracking --interpolate
```

Target selected case directories only:

```bash
lst-tools process tracking --dir kc_10pt00 --dir kc_20pt00
```

## 6. Set Up and Process Spectra

Set up spectra cases:

```bash
lst-tools setup spectra
```

After running spectra jobs, process output:

```bash
lst-tools process spectra
```

## 7. Clean Generated Artifacts

```bash
lst-tools clean parsing --force
lst-tools clean tracking --force
lst-tools clean spectra --force
```

## 8. Inspect Meanflow Quickly

```bash
lst-tools info meanflow.bin
```

This prints summary metadata, coordinate range, and reference quantities.

