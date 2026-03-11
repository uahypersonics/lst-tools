# Config

Configuration management for `lst_tools`.

## Functions

### `read_config`

```python
from lst_tools import read_config

config = read_config("lst.cfg")
```

Read a TOML configuration file and return a dictionary.

::: lst_tools.config.read_config

### `write_config`

```python
from lst_tools import write_config

write_config(config, "lst.cfg")
```

Write a configuration dictionary to a TOML file.

::: lst_tools.config.write_config

### `validate_config`

```python
from lst_tools import validate_config

validate_config(config)
```

Validate a configuration dictionary against the expected schema.

::: lst_tools.config.validate_config

### `find_config`

```python
from lst_tools import find_config

path = find_config()
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
