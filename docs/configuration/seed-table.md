# Seed Table

Set values when `setup tracking` is used to write `seed_alpha.dat` into
each `kc_*` directory.

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Enable seed-table generation during tracking setup |
| `source_file` | `str \| null` | `null` | Optional source contour file; default auto-detects from tracking inputs |
| `n_seeds` | `int` | `12` | Number of seeds written per detected ridge |
| `min_growth` | `float` | `0.0` | Minimum growth level required before a station contributes a seed |
| `gate_tol` | `float` | `0.05` | Ridge-matching tolerance passed to the tracker |
| `min_valid` | `int` | `5` | Minimum stations a ridge must span before acceptance |
| `smooth_passes` | `int` | `0` | Optional smoothing passes before ridge detection |
| `gate_by_keep_mask` | `bool` | `true` | Reject off-ridge candidate seeds with the keep-mask filter |
| `x_range` | `list[float]` | `[]` | Optional streamwise clipping window for seed selection |
| `f_range` | `list[float]` | `[]` | Optional frequency clipping window for seed selection |
| `threshold` | `float` | `0.15` | Solver-side override radius in normalized `(x, f)` space |
| `output_file` | `str` | `"seed_alpha.dat"` | Output filename written into each tracking case |

## Example

```toml
[seed_table]
enabled    = true
n_seeds    = 12
min_growth = 0.0
gate_tol   = 0.05
```
