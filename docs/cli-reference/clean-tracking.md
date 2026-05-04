# `clean tracking`

Remove solver artifacts from tracking case directories.

## Options

| Option | Meaning |
|---|---|
| `--dir`, `-d` | Select one or more case directories to clean; omit it to clean all `kc_*` directories |
| `--force`, `-f` | Skip the confirmation prompt |

```bash
# clean all kc_* directories in current directory
lst-tools clean tracking --force

# clean selected directories only
lst-tools clean tracking --dir kc_10pt00 --dir kc_20pt00 --force
```