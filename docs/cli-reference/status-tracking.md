# `status tracking`

Scan `kc_*` cases for solver run logs without connecting to a scheduler:

```bash
lst-tools status tracking [DIRECTORY]
lst-tools status tracking [DIRECTORY] --all-issues
```

`DIRECTORY` defaults to the current directory. The command summarizes cases
without a `run.log`, logs with no solver finish evidence, and logs containing a
`Total Time ... sec` footer. Finished cases get a single line, even if local
solver iterations did not converge along the way. For unfinished cases it lists
the first ten `Method did not converge` messages, including the station number,
frequency, and x-coordinate when the matching initialization progress line
provides one. Use `--all-issues` to show every issue in unfinished cases.
The summary counts convergence issues only from unfinished cases.

**Finished means only that the solver reached its timing footer.** It does not
mean every local solve converged, nor does it establish scheduler exit status.
Without a footer, the run may be ongoing, interrupted, or failed; status reports
an unfinished run with convergence issues as "stopped after nonconvergence".
Otherwise it reports incomplete/unknown. No files are modified.