# `lastrac`

Convert the HDF5 base flow to `meanflow.bin` for `lst.x`.

## Options

| Option | Meaning |
|---|---|
| `--cfg`, `-c` | Load a specific config file instead of auto-discovery |
| `--debug`, `-d` | Global option that writes conversion diagnostics to `./debug/` |

```bash
lst-tools lastrac

# explicit config path
lst-tools lastrac --cfg myconfig.cfg
```