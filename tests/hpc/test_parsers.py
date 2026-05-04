"""Tests for lst_tools.hpc._parsers — all pure functions, no I/O."""

from __future__ import annotations


from lst_tools.hpc._parsers import (
    _most_common_int,
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
# helpers
# ------------------------------------------------------------------
class TestMostCommonInt:
    def test_returns_most_common_value_and_histogram(self):
        best, hist = _most_common_int([48, 64, 48, 32, 48, 64])

        assert best == 48
        assert hist == {48: 3, 64: 2, 32: 1}


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

    def test_empty_string_defaults_to_one_hour(self):
        assert coerce_time_to_hms("") == "01:00:00"

    def test_invalid_string_is_coerced_as_float_input(self):
        assert coerce_time_to_hms("1.25") == "01:15:00"


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

    def test_ignores_invalid_tokens(self):
        result = parse_slurm_cpus_env("64(x2),garbage,32,bad(xq)")
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

    def test_skips_blank_and_invalid_lines(self):
        output = "\nnode001 not-a-number\nmalformed\nnode002 96\n"
        assert parse_sinfo_cpus(output) == [96]


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

    def test_ignores_non_numeric_cpu_line(self):
        output = "CPU(s):                one-twenty-eight\nCPU op-mode(s): 32-bit\n"
        assert parse_lscpu_cpus(output) is None


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

    def test_ignores_blank_lines_and_whitespace(self):
        text = " node1 \n\nnode1\n  node2\n"
        result = parse_pbs_nodefile(text)
        assert result is not None
        assert sorted(result) == [1, 2]


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

    def test_falls_back_to_resources_available_pattern(self):
        output = (
            "node001\n"
            "resources_available.ncpus = 64\n"
            "node002\n"
            "resources_available.ncpus = 72\n"
        )
        assert parse_pbsnodes_cpus(output) == [64, 72]

    def test_skips_invalid_tabular_cpu_field(self):
        output = "node001  free  512gb  48  0  0  broken/total\n"
        assert parse_pbsnodes_cpus(output) == []

    def test_invalid_slash_field_falls_back_to_regex_parse(self):
        output = (
            "node001 free 512gb 48 0 0 broken/total\n"
            "resources_available.ncpus = 40\n"
        )
        assert parse_pbsnodes_cpus(output) == [40]


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

    def test_skips_section_with_invalid_total_time(self):
        output = "PI: parent_1381 Total time: invalid\n  Group: chader Time used: 1:00:00 Time encumbered: 0:00:00\n"
        assert parse_va_output(output) == []

    def test_skips_section_without_group(self):
        output = "PI: parent_1381 Total time: 10:00:00\n  No group line here\n"
        assert parse_va_output(output) == []

    def test_keeps_standard_row_when_group_usage_is_unparseable(self):
        output = (
            "PI: parent_1381 Total time: 10:00:00\n"
            "  Group: chader Time used: invalid Time encumbered: 0:00:00\n"
        )
        rows = parse_va_output(output)

        assert len(rows) == 1
        assert rows[0]["used"] == 0
        assert rows[0]["remaining"] == 10

    def test_skips_incomplete_high_priority_subsection(self):
        output = (
            "PI: parent_1381 Total time: 20:00:00\n"
            "  Group: chader Time used: 5:00:00 Time encumbered: 0:00:00\n"
            "  High Priority QOS sub-allocation:\n"
            "    user_qos_chader:\n"
            "    Total time: invalid\n"
        )
        rows = parse_va_output(output)

        assert len(rows) == 1
        assert rows[0]["partition"] == "standard"

    def test_skips_section_with_malformed_pi_header(self):
        output = "PI: parent_1381 wrong header\n  Group: chader Time used: 1:00:00 Time encumbered: 0:00:00\n"
        assert parse_va_output(output) == []

    def test_skips_section_with_short_group_line(self):
        output = (
            "PI: parent_1381 Total time: 10:00:00\n"
            "  Group:\n"
        )
        assert parse_va_output(output) == []

    def test_skips_high_priority_row_when_qos_used_is_unparseable(self):
        output = (
            "PI: parent_1381 Total time: 20:00:00\n"
            "  Group: chader Time used: 5:00:00 Time encumbered: 0:00:00\n"
            "  High Priority QOS sub-allocation:\n"
            "    user_qos_chader:\n"
            "    Total time: 10:00:00\n"
            "    Time used: invalid Time encumbered: 0:00:00\n"
            "    Time remaining: 5:00:00\n"
        )
        rows = parse_va_output(output)

        assert len(rows) == 1
        assert rows[0]["partition"] == "standard"

    def test_skips_high_priority_row_when_qos_remaining_is_unparseable(self):
        output = (
            "PI: parent_1381 Total time: 20:00:00\n"
            "  Group: chader Time used: 5:00:00 Time encumbered: 0:00:00\n"
            "  High Priority QOS sub-allocation:\n"
            "    user_qos_chader:\n"
            "    Total time: 10:00:00\n"
            "    Time used: 2:00:00 Time encumbered: 0:00:00\n"
            "    Time remaining: invalid\n"
        )
        rows = parse_va_output(output)

        assert len(rows) == 1
        assert rows[0]["partition"] == "standard"


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

    def test_skips_lines_before_header_and_malformed_rows(self):
        output = (
            "preamble\n"
            "System   Account  Allocated  Used  Remaining  %Remain\n"
            "===================================================================\n"
            "too-short row\n"
            "puma grp1 bad 20000 80000 80.00%\n"
            "puma grp2 50000 10000 40000 80.00%\n"
        )

        rows = parse_show_usage_output(output)

        assert len(rows) == 1
        assert rows[0]["account"] == "grp2"
