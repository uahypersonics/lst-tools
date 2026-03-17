"""Pure-function parsers for HPC command output.

Every function here is deterministic: ``(str) -> value``.
They have **no** I/O and are trivially unit-testable.
"""

from __future__ import annotations

import re
from collections import Counter


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def _most_common_int(vals: list[int]) -> tuple[int, dict[int, int]]:
    """Return ``(most_common_value, full_histogram)``."""
    c = Counter(vals)
    best = c.most_common(1)[0][0]
    return best, dict(c)


# ------------------------------------------------------------------
# time conversion
# ------------------------------------------------------------------
def coerce_time_to_hms(val: float | str | None) -> str:
    """Convert *val* (hours as ``float``, or ``"HH:MM:SS"``) to ``HH:MM:SS``."""
    if isinstance(val, str) and val:
        parts = val.split(":")
        if len(parts) == 3 and all(p.isdigit() for p in parts):
            return val

    h = float(val if val not in (None, "") else 1.0)
    if h < 0:
        h = 1.0

    total = int(round(h * 3600))
    HH, rem = divmod(total, 3600)
    MM, SS = divmod(rem, 60)
    return f"{HH:02d}:{MM:02d}:{SS:02d}"


# ------------------------------------------------------------------
# SLURM CPU detection helpers
# ------------------------------------------------------------------
def parse_slurm_cpus_env(raw: str) -> list[int]:
    """Parse ``SLURM_JOB_CPUS_PER_NODE`` value, e.g. ``"128(x4)"``."""
    parts: list[int] = []
    for token in raw.split(","):
        m = re.match(r"(\d+)\(x(\d+)\)", token)
        if m:
            n, rep = int(m.group(1)), int(m.group(2))
            parts.extend([n] * rep)
        else:
            try:
                parts.append(int(token))
            except ValueError:
                pass
    return parts


def parse_sinfo_cpus(output: str) -> list[int]:
    """Parse ``sinfo -h -o '%n %c'`` output → list of per-node CPU counts."""
    cpus: list[int] = []
    for ln in output.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) >= 2:
            try:
                cpus.append(int(parts[-1]))
            except ValueError:
                pass
    return cpus


def parse_lscpu_cpus(output: str) -> int | None:
    """Parse ``lscpu`` output → total CPU count, or ``None``."""
    for ln in output.splitlines():
        if ln.lower().startswith("cpu(s):"):
            toks = ln.split()
            for t in reversed(toks):
                if t.isdigit():
                    return int(t)
    return None


# ------------------------------------------------------------------
# PBS CPU detection helpers
# ------------------------------------------------------------------
def parse_pbs_nodefile(text: str) -> list[int] | None:
    """Parse ``$PBS_NODEFILE`` contents → per-host CPU counts, or ``None``."""
    hosts = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not hosts:
        return None
    counts = Counter(hosts)
    return list(counts.values())


def parse_pbsnodes_cpus(output: str) -> list[int]:
    """Parse ``pbsnodes -ajS`` or ``pbsnodes -a`` output → CPU counts."""
    cpus: list[int] = []
    for ln in output.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("vnode") or ln.startswith("-"):
            continue
        parts = ln.split()
        if len(parts) >= 7:
            ncpus_field = parts[6]
            if "/" in ncpus_field:
                try:
                    total = int(ncpus_field.split("/")[-1])
                    cpus.append(total)
                    continue
                except ValueError:
                    pass
        m = re.search(r"resources_available\.ncpus\s*=\s*(\d+)", ln)
        if m:
            cpus.append(int(m.group(1)))
    return cpus


# ------------------------------------------------------------------
# resource / allocation parsers
# ------------------------------------------------------------------
def parse_va_output(output: str) -> list[dict[str, object]]:
    """Parse UA HPC ``va`` command output → list of account-resource dicts."""
    pi_sections = output.split("PI: ")[1:]  # skip preamble
    rows: list[dict[str, object]] = []

    for section in pi_sections:
        lines = section.strip().split("\n")
        if not lines:
            continue

        pi_line = lines[0].strip()
        pi_parts = pi_line.split()
        if len(pi_parts) < 4 or pi_parts[1] != "Total" or pi_parts[2] != "time:":
            continue

        pi_total_time = pi_parts[3]

        try:
            tp = pi_total_time.split(":")
            pi_total_hours = int(tp[0]) + int(tp[1]) / 60 + int(tp[2]) / 3600
        except Exception:
            continue

        # find Group / account
        account_name = None
        account_used_hours = 0.0

        for line in lines:
            if line.strip().startswith("Group: "):
                parts = line.strip().split()
                if len(parts) >= 2:
                    account_name = parts[1]
                    try:
                        used_str = (
                            line.split("Time used:")[1]
                            .split("Time encumbered:")[0]
                            .strip()
                        )
                        up = used_str.split(":")
                        account_used_hours = (
                            int(up[0]) + int(up[1]) / 60 + int(up[2]) / 3600
                        )
                    except Exception:
                        pass
                break

        if not account_name:
            continue

        remaining_hours = pi_total_hours - account_used_hours
        percent_remain = (
            (remaining_hours / pi_total_hours * 100) if pi_total_hours > 0 else 0
        )

        rows.append(
            {
                "system": "puma",
                "account": account_name,
                "allocated": int(pi_total_hours),
                "used": int(account_used_hours),
                "remaining": int(remaining_hours),
                "percent_remain": percent_remain,
                "partition": "standard",
                "qos": None,
            }
        )

        # high-priority QOS sub-section
        if "High Priority QOS" not in section:
            continue

        qos_name: str | None = None
        qos_allocation: float | None = None
        qos_used: float | None = None
        qos_remaining: float | None = None

        for line in lines:
            if "user_qos_" in line and line.strip().endswith(":"):
                qos_name = line.strip().rstrip(":")
            elif "Total time:" in line and qos_name:
                try:
                    qts = line.split("Total time:")[1].strip().split(":")
                    qos_allocation = int(qts[0]) + int(qts[1]) / 60 + int(qts[2]) / 3600
                except Exception:
                    pass
            elif "Time used:" in line and qos_name and qos_allocation:
                try:
                    qus = (
                        line.split("Time used:")[1]
                        .split("Time encumbered:")[0]
                        .strip()
                        .split(":")
                    )
                    qos_used = int(qus[0]) + int(qus[1]) / 60 + int(qus[2]) / 3600
                except Exception:
                    pass
            elif "Time remaining:" in line and qos_name and qos_allocation:
                try:
                    qrs = line.split("Time remaining:")[1].strip().split(":")
                    qos_remaining = (
                        int(qrs[0]) + int(qrs[1]) / 60 + int(qrs[2]) / 3600
                    )
                except Exception:
                    pass

        if (
            qos_name
            and qos_allocation
            and qos_used is not None
            and qos_remaining is not None
        ):
            qos_pct = (
                (qos_remaining / qos_allocation * 100) if qos_allocation > 0 else 0
            )
            rows.append(
                {
                    "system": "puma",
                    "account": account_name,
                    "allocated": int(qos_allocation),
                    "used": int(qos_used),
                    "remaining": int(qos_remaining),
                    "percent_remain": qos_pct,
                    "partition": "high_priority",
                    "qos": qos_name,
                }
            )

    return rows


def parse_show_usage_output(output: str) -> list[dict[str, object]]:
    """Parse PBS ``show_usage`` output → list of account-resource dicts."""
    rows: list[dict[str, object]] = []
    started = False

    for ln in output.splitlines():
        if ln.strip().startswith("==="):
            started = True
            continue
        if not started or not ln.strip():
            continue

        parts = ln.split()
        if len(parts) < 6:
            continue

        system = parts[0]
        account = parts[1]

        try:
            allocated = int(parts[2])
            used = int(parts[3])
            remaining = int(parts[4])
            percent_remain = float(parts[5].rstrip("%"))
        except Exception:
            continue

        rows.append(
            {
                "system": system,
                "account": account,
                "allocated": allocated,
                "used": used,
                "remaining": remaining,
                "percent_remain": percent_remain,
                "partition": None,
                "qos": None,
            }
        )

    return rows
