# API Usage

This page shows practical Python workflows built on `lst_tools`. Use
[Workflow](../workflow/index.md) for the phase layout and
[API Reference](../api/index.md) for full signatures and module-level details.

Use the API when Python control flow is part of the workflow.

- modify `lst.cfg` programmatically before setup
- loop over multiple cases or parameter values
- integrate `lst-tools` into a larger preprocessing or post-processing script
- keep setup and processing inside one Python driver

Use the CLI when one-off commands are enough and shell usage is simpler.

## Initialization and Meanflow Preparation

Most Python workflows start by loading the config, changing a few values,
and writing the updated file back to disk.

```python
from lst_tools import read_config, write_config

config = read_config("lst.cfg")

config.lst.params.ny = 200
config.lst.params.f_min = 20_000.0
config.lst.params.f_max = 120_000.0

write_config("lst.cfg", overwrite=True, cfg_data=config.to_dict())
```

Keep [Configuration](../configuration/index.md) open while editing values.

Meanflow conversion is a lower-level API step than setup or processing.

`convert_meanflow` expects `Grid` and `Flow` objects plus a loaded config:

```python
from lst_tools import convert_meanflow

convert_meanflow(grid, flow, "meanflow.bin", cfg=config)
```

Use this path when base-flow data is already available in Python. For the
exact data requirements and helper objects, use the
[Convert](../api/convert.md) and [Grid & Flow](../api/core.md) reference
pages.

## Setup Runs

The setup functions accept a loaded config and write the same artifacts as
the CLI commands.

```python
from lst_tools import read_config, parsing_setup, tracking_setup, spectra_setup

config = read_config("lst.cfg")

parsing_deck = parsing_setup(cfg=config, auto_fill=True)
tracking_root = tracking_setup(cfg=config, auto_fill=True)
spectra_decks = spectra_setup(cfg=config)
```

Use these functions when a Python script is preparing runs for several cases
or updating the config between steps.

Use one Python driver to edit config values and prepare several runs:

```python
from lst_tools import read_config, parsing_setup, write_config

config = read_config("lst.cfg")

for beta_max in (0.02, 0.04, 0.06):
    config.lst.params.beta_e = beta_max
    out_name = f"lst_input_beta_{beta_max:.2f}.dat"
    parsing_setup(cfg=config, out_name=out_name, auto_fill=True)

write_config("lst.cfg", overwrite=True, cfg_data=config.to_dict())
```

This kind of loop is usually the main reason to switch from CLI to API.

## Postprocessing and Cleanup

The processing functions are useful when results need to be filtered,
batched, or chained into another analysis step.

```python
from pathlib import Path

from lst_tools import read_config, spectra_process, tracking_process

config = read_config("lst.cfg")

tracking_process(
    cfg=config,
    work_dir=Path("."),
    interpolate=True,
)

spectra_process(cfg=config)
```

Pass selected `kc_*` directories through `kc_dirs` when tracking processing
should target only part of a run set.

Quick preview visualization is currently exposed through the CLI wrappers.
Use `lst-tools visualize parsing` or `lst-tools visualize tracking` when PNG
output is needed.