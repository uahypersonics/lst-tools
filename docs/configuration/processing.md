# Processing

Set post-processing controls.

## `[processing.tracking]`

Set values to control ridge extraction and volume assembly for
tracking results.

| Key | Type | Default | Description |
|---|---|---|---|
| `interpolate` | `bool` | `false` | Enable parabolic peak interpolation |
| `gate_tol` | `float` | `0.10` | Ridge-tracking gating tolerance |
| `min_valid` | `int` | `40` | Minimum valid points for ridge acceptance |
| `peak_order` | `int` | `1` | Local peak search order |

## `[processing.spectra]`

Set values to gate, classify, and retain spectra branches.

| Key | Type | Default | Description |
|---|---|---|---|
| `alpr_min` | `float` | - | Lower bound for `re(alpha)` gating |
| `alpr_max` | `float` | - | Upper bound for `re(alpha)` gating |
| `alpi_min` | `float` | - | Lower bound for `-im(alpha)` gating |
| `alpi_max` | `float` | - | Upper bound for `-im(alpha)` gating |
| `branch_gate` | `float` | `0.25` | Maximum normalized jump allowed during branch matching |
| `branch_min_points` | `int` | `2` | Minimum points required to keep a tracked branch |
| `isolation_k` | `int` | `3` | Neighbor count used in the isolation score |
| `isolation_threshold` | `float` | - | Minimum isolation score required for classification |
| `classify_min_points` | `int` | `3` | Minimum isolated points required to keep a classified branch |

## `[processing.parsing]`

(not yet implemented)

## Example

```toml
[processing.tracking]
interpolate = true
gate_tol    = 0.10
min_valid   = 40
peak_order  = 1

[processing.spectra]
alpr_min = 0.0
alpr_max = 2.0
alpi_min = 0.0
alpi_max = 0.5
```
