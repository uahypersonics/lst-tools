# Examples

Practical examples of using `lst_tools` for common tasks.

## Reading Data

### Flow Conditions

```python
import lst_tools

fc = lst_tools.read_flow_conditions("flow_conditions.dat")
print(f"Mach = {fc['mach']}")
print(f"Re/m = {fc['re1']}")
```

### Tecplot ASCII

```python
import lst_tools

data = lst_tools.read_tecplot_ascii("profile.dat")
```

## Geometry

### Surface Curvature

```python
import cfd_io
import lst_tools

grid_raw, flow_raw, attrs = cfd_io.read_file("baseflow.hdf5")

# compute curvature along the surface
kappa = lst_tools.curvature(grid)

# curvilinear coordinate
s = lst_tools.curvilinear_coordinate(grid)

# surface angle distribution
theta = lst_tools.surface_angle(grid)
```

## LST Workflow

### Full Workflow (CLI)

```bash
# step 1: initialize config from flow conditions
lst-tools init --flow flow_conditions.dat -i

# step 2: convert base flow to LST format
lst-tools convert --input baseflow.hdf5

# step 3: parsing sweep (frequency/wavenumber)
lst-tools parsing

# step 4: tracking (spatial marching)
lst-tools tracking

# step 5: post-process tracking
lst-tools tracking-process
```

### Full Workflow (Python)

```python
import cfd_io
import lst_tools

# read config
config = lst_tools.read_config("lst.cfg")

# read base flow via cfd-io
grid_raw, flow_raw, attrs = cfd_io.read_file(config["input_file"])

# convert to LST format
lst_tools.convert_meanflow(grid, flow, config)

# set up and run tracking
lst_tools.tracking_setup(config)

# post-process
lst_tools.tracking_process(config)
```

## HPC Job Scripts

```python
import lst_tools

# configure HPC settings
hpc_cfg = lst_tools.hpc_configure(config)

# generate job scripts
lst_tools.script_build(hpc_cfg)
```

Or from the CLI:

```bash
lst-tools hpc
```
