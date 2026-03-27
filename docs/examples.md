# Examples

Representative command-line examples for common workflows.

## End-to-End Tracking Workflow

```bash
# 1) initialize config
lst-tools init --geometry cone

# 2) convert HDF5 meanflow to meanflow.bin
lst-tools lastrac

# 3) set up parsing and run solver externally
lst-tools setup parsing --auto-fill

# 4) set up tracking and run solver externally in kc_* folders
lst-tools setup tracking --auto-fill

# 5) process tracking outputs
lst-tools process tracking --interpolate
```

## Spectra Workflow

```bash
lst-tools setup spectra
lst-tools process spectra
```

## HPC Script Generation

```bash
lst-tools hpc
```

## Utilities

```bash
# inspect meanflow metadata
lst-tools info meanflow.bin

# clean generated artifacts
lst-tools clean parsing --force
lst-tools clean tracking --force
lst-tools clean spectra --force
```
