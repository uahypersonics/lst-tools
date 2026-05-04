# Data I/O

File readers and writers for the formats used by the LST workflow.

## Fortran Binary (re-exported from `cfd-io`)

`lst_tools` re-exports the Fortran binary reader and writer used for solver
binary files from [`cfd-io`](https://pypi.org/project/cfd-io/):

```python
from lst_tools import FortranBinaryReader, FortranBinaryWriter

reader = FortranBinaryReader("meanflow.bin")
writer = FortranBinaryWriter("output.bin")
```

See the `cfd-io` documentation for the full API.

## Meanflow Binary

### `LastracReader`

```python
from lst_tools.data_io import LastracReader
```

::: lst_tools.data_io.LastracReader

### `LastracWriter`

```python
from lst_tools.data_io import LastracWriter
```

::: lst_tools.data_io.LastracWriter

## Flow Conditions

### `read_flow_conditions`

```python
from lst_tools import read_flow_conditions

fc = read_flow_conditions("flow_conditions.dat")
```

::: lst_tools.data_io.read_flow_conditions

## Tecplot ASCII

### `read_tecplot_ascii`

```python
from lst_tools import read_tecplot_ascii

data = read_tecplot_ascii("profile.dat")
```

::: lst_tools.data_io.read_tecplot_ascii
