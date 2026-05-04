# `visualize parsing`

Render parsing contours with the built-in visualization defaults.

This wrapper is intentionally minimal. In the common case, run it with no
arguments or only set the input and output paths.

It uses built-in defaults for variable mapping and contour styling. When you
need deeper plotting control, use a dedicated plotting script instead of this
wrapper.

## Minimal Usage

```bash
# use default parsing file and defaults
lst-tools visualize parsing

# explicit input/output paths
lst-tools visualize parsing \
	--input growth_rate_with_nfact_amps.dat \
	--out alpi_contours_parsing
```

## Common Options

| Option | Meaning |
|---|---|
| `--input`, `-i` | Select the parsing Tecplot input file |
| `--out`, `-o` | Select the output directory |

The command renders the standard LST alpha contour PNGs for every k-slice into
the selected output directory.