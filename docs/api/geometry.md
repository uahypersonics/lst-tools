# Geometry

Surface geometry computations from grid data.

## Functions

### `curvature`

```python
from lst_tools import curvature

kappa = curvature(grid)
```

Compute the surface curvature distribution.

::: lst_tools.geometry.curvature

### `curvilinear_coordinate`

```python
from lst_tools import curvilinear_coordinate

s = curvilinear_coordinate(grid)
```

Compute the curvilinear coordinate along the surface.

::: lst_tools.geometry.curvilinear_coordinate

### `surface_angle`

```python
from lst_tools import surface_angle

theta = surface_angle(grid)
```

Compute the surface angle distribution.

::: lst_tools.geometry.surface_angle

### `radius`

```python
from lst_tools import radius

r = radius(grid)
```

Compute the local body radius.

::: lst_tools.geometry.radius

## Geometry Kinds

### `GeometryKind`

```python
from lst_tools import GeometryKind, list_geometry_kinds

kinds = list_geometry_kinds()
```

Enumeration of supported geometry types.

::: lst_tools.geometry.GeometryKind

::: lst_tools.geometry.list_geometry_kinds
