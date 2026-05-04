# Geometry

Set the geometry model and reference dimensions used by the workflow.

| Key | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | - | Geometry kind (e.g., cone, flat plate) |
| `theta_deg` | `float` | - | Half-angle [deg] |
| `r_nose` | `float` | - | Nose radius [m] |
| `l_ref` | `float` | `1` | Reference length [m] |
| `is_body_fitted` | `bool` | `false` | Whether the grid is body-fitted |

This section controls geometry-specific behavior during meanflow conversion
and input-deck generation.
