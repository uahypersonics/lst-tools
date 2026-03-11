# Basic Usage

## Reading Base Flow Data

`lst_tools` uses [cfd-io](https://github.com/uahypersonics/cfd-io) to read base flow files:

```python
import cfd_io

grid_raw, flow_raw, attrs = cfd_io.read_file("baseflow.hdf5")
```

## Geometry Computations

Compute surface geometry quantities from the grid:

```python
# surface curvature
kappa = lst_tools.curvature(grid)

# curvilinear coordinate along the surface
s = lst_tools.curvilinear_coordinate(grid)

# surface angle
theta = lst_tools.surface_angle(grid)

# nose radius
r = lst_tools.radius(grid)
```

## Converting Base Flow

Convert an HDF5 base flow to the Fortran binary format used by the LST solver:

```python
lst_tools.convert_meanflow(grid, flow, config)
```

Or from the CLI:

```bash
lst-tools lastrac
```

## Reading Flow Conditions

```python
fc = lst_tools.read_flow_conditions("flow_conditions.dat")
print(fc)
```

## Configuration

Read and validate a configuration file:

```python
config = lst_tools.read_config("lst.cfg")
lst_tools.validate_config(config)
```

## Setting Up LST Calculations

### Parsing (Initial Sweep)

```python
lst_tools.parsing_setup(config)
```

### Tracking

```python
lst_tools.tracking_setup(config)
```

### Spectra

```python
lst_tools.spectra_setup(config)
```

## Post-Processing Results

### Tracking Results

```python
lst_tools.tracking_process(config)
```

### Spectra Results

```python
lst_tools.spectra_process(config)
```
