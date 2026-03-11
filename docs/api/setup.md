# Setup

LST calculation setup functions for parsing, tracking, and spectra.

## Functions

### `parsing_setup`

```python
from lst_tools import parsing_setup

parsing_setup(config)
```

Set up the parsing step (initial frequency/wavenumber sweep).

::: lst_tools.setup.parsing_setup

### `tracking_setup`

```python
from lst_tools import tracking_setup

tracking_setup(config)
```

Set up the tracking step (spatial marching along disturbance trajectories).

::: lst_tools.setup.tracking_setup

### `spectra_setup`

```python
from lst_tools import spectra_setup

spectra_setup(config)
```

Set up spectra calculations at multiple streamwise locations.

::: lst_tools.setup.spectra_setup
