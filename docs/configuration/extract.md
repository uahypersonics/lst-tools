# Extract

Set extraction outputs, station locations, and wall-normal resolution for `lst-tools extract`.

| Key | Type | Default | Description |
|---|---|---|---|
| `hdf5_out` | `str` | `extracted_baseflow.hdf5` | Output HDF5 baseflow file path |
| `profiles_out` | `str` | — | Output Tecplot profiles file path (omit to skip) |
| `wall_out` | `str` | — | Output Tecplot wall curve file path (omit to skip) |
| `surface` | `str` | auto | Surface side: `lower` or `upper` |
| `n_eta` | `int` | `200` | Wall-normal points per profile |
| `eta_max` | `float` | auto | Wall-normal extent in mesh length units |
| `eta_distribution` | `str` | `cosine` | Wall-normal point distribution: `uniform`, `cosine`, or `tanh` |
| `eta_stretch` | `float` | `3.0` | Stretching strength for the `tanh` distribution |
| `stations` | `list[float]` | — | Streamwise x-coordinates for profile stations |

```toml
[extract]
hdf5_out    = "extracted_baseflow.hdf5"
profiles_out = "profiles.dat"
wall_out    = "wall.dat"
surface     = "upper"
n_eta       = 200
eta_max     = 0.012
eta_distribution = "cosine"
eta_stretch = 3.0
stations    = [0.10, 0.20, 0.30, 0.40, 0.50]
```
