# `info`

Inspect a meanflow binary.

The file header is printed once. Each station then reports its own streamwise
location, eta bounds, point count, geometry values, and reference quantities.

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