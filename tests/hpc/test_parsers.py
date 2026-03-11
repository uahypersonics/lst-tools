"""Tests for lst_tools.hpc._parsers — all pure functions, no I/O."""

from __future__ import annotations


from lst_tools.hpc._parsers import (
    coerce_time_to_hms,
    parse_lscpu_cpus,
    parse_pbs_nodefile,
    parse_pbsnodes_cpus,
    parse_show_usage_output,
    parse_sinfo_cpus,
    parse_slurm_cpus_env,
    parse_va_output,
)


# ------------------------------------------------------------------
# coerce_time_to_hms
# ------------------------------------------------------------------
class TestCoerceTimeToHms:
    def test_float_one_hour(self):
        assert coerce_time_to_hms(1.0) == "01:00:00"

    def test_float_half_hour(self):
        assert coerce_time_to_hms(0.5) == "00:30:00"

    def test_passthrough_valid_string(self):
        assert coerce_time_to_hms("02:30:00") == "02:30:00"

    def test_none_defaults_to_one_hour(self):
        assert coerce_time_to_hms(None) == "01:00:00"

    def test_negative_defaults_to_one_hour(self):
        assert coerce_time_to_hms(-5) == "01:00:00"

    def test_large_value(self):
        assert coerce_time_to_hms(100) == "100:00:00"

    def test_fractional(self):
        assert coerce_time_to_hms(1.5) == "01:30:00"


# ------------------------------------------------------------------
# SLURM CPU env parsing
# ------------------------------------------------------------------
class TestParseSlurmCpusEnv:
    def test_simple(self):
        assert parse_slurm_cpus_env("128") == [128]

    def test_repeated(self):
        assert parse_slurm_cpus_env("128(x4)") == [128, 128, 128, 128]

    def test_comma_separated(self):
        assert parse_slurm_cpus_env("96,96") == [96, 96]

    def test_mixed(self):
        result = parse_slurm_cpus_env("64(x2),32")
        assert result == [64, 64, 32]


# ------------------------------------------------------------------
# sinfo parsing
# ------------------------------------------------------------------
class TestParseSinfoCpus:
    def test_basic(self):
        output = "node001 128\nnode002 128\nnode003 64\n"
        assert parse_sinfo_cpus(output) == [128, 128, 64]

    def test_empty(self):
        assert parse_sinfo_cpus("") == []


# ------------------------------------------------------------------
# lscpu parsing
# ------------------------------------------------------------------
class TestParseLscpuCpus:
    def test_typical(self):
        output = (
            "Architecture:          x86_64\n"
            "CPU(s):                128\n"
            "Thread(s) per core:    2\n"
        )
        assert parse_lscpu_cpus(output) == 128

    def test_no_match(self):
        assert parse_lscpu_cpus("nothing here") is None


# ------------------------------------------------------------------
# PBS nodefile
# ------------------------------------------------------------------
class TestParsePbsNodefile:
    def test_typical(self):
        text = "node1\nnode1\nnode1\nnode2\nnode2\nnode2\n"
        result = parse_pbs_nodefile(text)
        assert result is not None
        assert sorted(result) == [3, 3]

    def test_empty(self):
        assert parse_pbs_nodefile("") is None


# ------------------------------------------------------------------
# pbsnodes parsing
# ------------------------------------------------------------------
class TestParsePbsnodesCpus:
    def test_tabular(self):
        # pbsnodes -ajS: column 6 (0-indexed) is the ncpus "free/total" field
        output = (
            "vnode           state   mem    ncpus   nmics   ngpus  ncpus_ft\n"
            "----\n"
            "node001  free  512gb  48  0  0  0/48\n"
            "node002  free  512gb  48  0  0  0/48\n"
        )
        assert parse_pbsnodes_cpus(output) == [48, 48]


# ------------------------------------------------------------------
# VA output
# ------------------------------------------------------------------
VA_FIXTURE = """\
Some header text

PI: parent_1381 Total time: 150000:00:00
  Group: chader Time used: 17184:56:00 Time encumbered: 0:00:00
  Some other line
"""

VA_FIXTURE_HP = """\
PI: parent_1381 Total time: 150000:00:00
  Group: chader Time used: 17184:56:00 Time encumbered: 0:00:00
  High Priority QOS sub-allocation:
    user_qos_chader:
    Total time: 10000:00:00
    Time used: 500:00:00 Time encumbered: 0:00:00
    Time remaining: 9500:00:00
"""


class TestParseVaOutput:
    def test_basic(self):
        rows = parse_va_output(VA_FIXTURE)
        assert len(rows) == 1
        assert rows[0]["account"] == "chader"
        assert rows[0]["allocated"] == 150000
        assert rows[0]["partition"] == "standard"

    def test_high_priority(self):
        rows = parse_va_output(VA_FIXTURE_HP)
        assert len(rows) == 2
        hp = [r for r in rows if r["partition"] == "high_priority"]
        assert len(hp) == 1
        assert hp[0]["qos"] == "user_qos_chader"


# ------------------------------------------------------------------
# show_usage output
# ------------------------------------------------------------------
SHOW_USAGE_FIXTURE = """\
System   Account  Allocated  Used  Remaining  %Remain
===================================================================
puma     grp1     100000     20000 80000      80.00%
puma     grp2     50000      50000 0          0.00%
"""


class TestParseShowUsageOutput:
    def test_basic(self):
        rows = parse_show_usage_output(SHOW_USAGE_FIXTURE)
        assert len(rows) == 2
        assert rows[0]["account"] == "grp1"
        assert rows[0]["remaining"] == 80000
        assert rows[1]["percent_remain"] == 0.0
