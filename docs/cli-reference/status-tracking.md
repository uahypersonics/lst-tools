# `status tracking`

Scan `kc_*` cases for solver run logs without connecting to a scheduler:

```bash
lst-tools status tracking [DIRECTORY]
lst-tools status tracking [DIRECTORY] --all-issues
```

`DIRECTORY` defaults to the current directory. The command summarizes cases
without a `run.log`, logs with no solver finish evidence, and logs containing a
`Total Time ... sec` footer. It lists the first ten `Method did not converge`
messages per case, including the station number, frequency, and x-coordinate
when the matching initialization progress line provides one. Use `--all-issues`
to show all recorded issues.

**Finished means only that the solver reached its timing footer.** It does not
mean every local solve converged, nor does it establish scheduler exit status.
Without a footer, the run may be ongoing, interrupted, or failed; status reports
an unfinished run with convergence issues as "stopped after nonconvergence".
Otherwise it reports incomplete/unknown. No files are modified.