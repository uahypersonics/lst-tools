# `extract`

Extract wall-normal profiles from a Tecplot FE-quad mesh at user-specified
streamwise stations.

Output path, station locations, and wall-normal resolution are read from
`[extract]` in `lst.cfg`. Freestream metadata (Mach, T_inf) is written to the
HDF5 only when `[flow_conditions]` provides both `mach` and `temp_inf`.

## Options

| Option | Meaning |
|---|---|
| `INPUT` | Tecplot FE-quad BLOCK ASCII input file (positional argument) |
| `--cfg`, `-c` | Explicit `lst.cfg` path (auto-discovered when omitted) |
| `--station` | Streamwise x-coordinate for one profile station (repeatable, overrides `lst.cfg`) |
| `--surface` | Surface side to extract from: `lower` or `upper` (auto-detected when omitted) |

`--surface` is auto-detected on one-sided meshes and only needs to be set when
both surfaces are present.

Output goes to `extracted_baseflow.hdf5` next to the input file by default.
Set `hdf5_out`, `profiles_out`, `wall_out`, `n_eta`, and `eta_distribution` in
`[extract]` when non-default values are needed.

```bash
# minimal — no config required
lst-tools extract mesh.dat --station 0.10 --station 0.20 --station 0.30

# reads stations and output path from lst.cfg
lst-tools extract mesh.dat

# force upper surface on a two-sided mesh
lst-tools extract mesh.dat --station 0.10 --surface upper
```
