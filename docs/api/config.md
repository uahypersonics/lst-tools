# Config

Configuration management for `lst_tools`.

## Functions

### `read_config`

```python
from lst_tools import read_config

config = read_config("lst.cfg")
```

Read a TOML configuration file and return a typed `Config` dataclass.

::: lst_tools.config.read_config

### `write_config`

```python
from lst_tools import write_config

write_config("lst.cfg", overwrite=True, cfg_data=config.to_dict())
```

Write configuration data to a TOML file.

::: lst_tools.config.write_config

### `Config`

```python
from lst_tools.config import Config

cfg = Config.from_toml("lst.cfg")
```

Typed configuration schema and validation.

::: lst_tools.config.Config

### `find_config`

```python
from lst_tools import find_config

path = find_config(".")
```

Search for a configuration file in the current directory and parent directories.

::: lst_tools.config.find_config

### `check_consistency`

```python
from lst_tools import check_consistency, format_report

report = check_consistency(config)
print(format_report(report))
```

Check configuration for internal consistency and return a diagnostic report.

::: lst_tools.config.check_consistency

::: lst_tools.config.format_report
