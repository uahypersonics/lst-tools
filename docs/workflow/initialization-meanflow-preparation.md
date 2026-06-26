# Initialization and Meanflow Preparation

This phase prepares the case definition and the solver-ready meanflow data.
Complete both steps before creating parsing or spectra runs.

| Stage | What it does | Main output |
|---|---|---|
| Initialization | Writes the starting configuration for the case | `lst.cfg` |
| Extract *(optional)* | Extracts wall-normal profiles from a CFD mesh when the base flow is in Tecplot FE-quad format | `extracted_baseflow.hdf5` |
| Meanflow Preparation | Converts the base flow into the solver-ready format used by `lst.x` | `meanflow.bin` |

See [CLI Usage](../user-guide/cli-usage.md) or
[API Usage](../user-guide/api-usage.md) for the practical steps.