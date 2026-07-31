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
| `eta_distribution` | `str` | `tanh` | Wall-normal point distribution: `uniform`, `cosine`, `tanh`, or `geometric` |
| `eta_stretch` | `float` | `2.0` | Stretching strength for the `tanh` distribution |
| `eta_wall_spacing` | `float` | — | First off-wall interval for the `geometric` distribution |
| `stations` | `list[float]` | — | Streamwise x-coordinates for profile stations |

```toml
[extract]
hdf5_out    = "extracted_baseflow.hdf5"
profiles_out = "profiles.dat"
wall_out    = "wall.dat"
surface     = "upper"
n_eta       = 200
eta_max     = 0.012
eta_distribution = "tanh"
eta_stretch = 2.0
eta_wall_spacing = ""
stations    = [0.10, 0.20, 0.30, 0.40, 0.50]
```

For geometric spacing, set `eta_wall_spacing` to the desired first interval.
It must not exceed `eta_max / (n_eta - 1)` so intervals grow away from the wall.
