# `process modes`

Match tracking ridge curves across neighboring beta cases and build separate
surfaces for each matched mode:

```bash
lst-tools process tracking --maxima
lst-tools process modes
lst-tools process modes --dir /path/to/cases --out mode_surfaces
lst-tools process modes --nx 1000
lst-tools process modes --force
```

The default input directory is the current directory. Relative output paths
are resolved inside it. A nonempty output directory is not overwritten unless
`--force` is supplied; forced processing replaces that output directory.

The command reads `nfac_max_mode_*/kc_*.dat` and
`alpi_max_mode_*/kc_*.dat` separately. It matches adjacent-beta ridges by
frequency agreement over their common streamwise range, **not** by their
local `mode_001`/`mode_002` folder numbers. Weak-overlap and ambiguous
matches become new mode IDs instead of being forced together. Matching for
N-factor and alpha-imaginary ridges is independent. N-factor ridge extraction
and envelope selection both maximize `Nfac3`; the other N-factor definitions
remain available as corresponding output fields.

Each `nfac_max_mode_*.dat` or `alpi_max_mode_*.dat` output is a Tecplot ordered
`(s, Beta)` surface containing every source-ridge quantity, including frequency,
real and imaginary alpha, amplitudes, N-factors, and Reynolds-number scalings,
plus a `valid` mask.
Both families use the same uniform 500-point `s` grid spanning the source
curves' combined minimum and maximum; `--nx` changes its resolution. Matching
is performed on the original ridge coordinates before this output interpolation.
Beta remains at the computed case values, without interpolation.
Values outside observed ridge segments are zero-filled with `valid = 0` for
Tecplot/preplot compatibility; measured zero values retain `valid = 1`.
No data is extrapolated. `mode_assignments.csv` lists the
source curve, assigned global mode, beta, and relative frequency matching
cost for auditing. A blank matching cost marks a new mode.

For every matched surface, the command also writes a one-dimensional
`nfac_envelope_mode_*.dat` or `alpi_envelope_mode_*.dat` curve. At each `s`,
the envelope selects the largest valid value across beta and retains the beta
and every corresponding quantity from that surface point. The envelope is
computed separately for each physical mode; it never switches between matched
mode surfaces. Locations without support are zero-filled with `valid = 0`.

Check the assignment report and plotted surfaces before interpreting mode
identity near branch crossings.