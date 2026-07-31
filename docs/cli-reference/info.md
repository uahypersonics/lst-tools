# `info`

Inspect a meanflow binary.

## Argument

| Argument | Meaning |
|---|---|
| `fpath` | Path to the `meanflow.bin` file to inspect |

## Options

| Option | Meaning |
|---|---|
| `--profiles-out`, `-o` | Optional Tecplot ASCII export of all meanflow profiles |

```bash
lst-tools info meanflow.bin

# write a Tecplot file with all station profiles
lst-tools info meanflow.bin --profiles-out profiles.dat
```