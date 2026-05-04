# `visualize tracking`

Render tracking volume contours with the built-in visualization defaults.

This wrapper is intentionally minimal. In the common case, run it with no
arguments or only set the input and output paths.

It uses built-in defaults for variable mapping and contour styling. When you
need deeper plotting control, use a dedicated plotting script instead of this
wrapper.

## Minimal Usage

```bash
# use default tracking volume file and defaults
lst-tools visualize tracking

# explicit input/output paths
lst-tools visualize tracking \
	--input lst_vol.dat \
	--out viz_tracking
```

## Common Options

| Option | Meaning |
|---|---|
| `--input`, `-i` | Select the tracking volume input file |
| `--out`, `-o` | Select the output directory |

The command renders the standard LST alpha contour PNGs for every k-slice into
the selected output directory.

Tracking fallback behavior:

1. When `lst_vol.dat` exists, `lst-tools visualize tracking` renders directly from it.
2. Otherwise, it discovers `kc_*` directories and reads `growth_rate_with_nfact_amps.dat` from each completed case.
3. In fallback mode, it writes PNGs to `alpi_contours_tracking/` in the current working directory and uses a shared contour scale across all discovered cases.

!!! note
		Install the visualization dependency in the same Python environment as
		`lst-tools` when you want to render PNG outputs.