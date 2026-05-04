# `clean parsing`

Remove parsing-generated artifacts from one directory.

## Options

| Option | Meaning |
|---|---|
| `--dir`, `-d` | Select the directory to clean |
| `--name`, `-n` | Set the input-deck filename to remove |
| `--force`, `-f` | Skip the confirmation prompt |

```bash
lst-tools clean parsing --dir .
lst-tools clean parsing --dir . --force
```