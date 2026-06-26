# Configuration

Use this section to navigate `lst.cfg` by topic.

Start with [Root](root.md), [Flow Conditions](flow-conditions.md),
[Geometry](geometry.md), and [Meanflow Conversion](meanflow-conversion.md).
Then adjust [Solver](solver.md), [Processing](processing.md),
[Seed Table](seed-table.md), and [HPC](hpc.md) only when the workflow needs
them.

## Create a Starting Config

```bash
lst-tools init
lst-tools init --geometry cone
```

## Common Editing Order

1. Set the root keys: `input_file` and `lst_exe`.
2. Fill `flow_conditions` and `geometry`.
3. Adjust `meanflow_conversion` for the base-flow slice you want.
   Run `lst-tools extract` after this step when you need wall-normal
   profiles from a CFD mesh file.
4. Set sweep or tracking values in `lst` solver and parameter sections.
5. Add processing, seed-table, or HPC settings only when needed.

## Sections

- [Root](root.md): top-level file paths and solver executable
- [Flow Conditions](flow-conditions.md): freestream and reference inputs
- [Geometry](geometry.md): geometry type and reference dimensions
- [Meanflow Conversion](meanflow-conversion.md): HDF5 to `meanflow.bin` controls
- [Extract](extract.md): wall-normal profile extraction controls
- [Solver](solver.md): solver mode, options, run parameters, and I/O paths
- [Processing](processing.md): tracking and spectra post-processing controls
- [Seed Table](seed-table.md): tracking seed generation controls
- [HPC](hpc.md): scheduler-script settings
