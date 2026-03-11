# Data I/O

File readers and writers for various scientific data formats.

## Fortran Binary

### `FortranBinaryReader`

```python
from lst_tools import FortranBinaryReader

reader = FortranBinaryReader("meanflow.bin")
```

::: lst_tools.data_io.FortranBinaryReader

### `FortranBinaryWriter`

```python
from lst_tools import FortranBinaryWriter

writer = FortranBinaryWriter("output.bin")
```

::: lst_tools.data_io.FortranBinaryWriter

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
