from lst_tools.config.check_consistency import (
    check_consistency,
    format_report,
    Issue,
    IssueLevel,
    get,
)


class TestCheckConsistency:
    """Test suite for check_consistency module"""

    def test_get_helper_function(self):
        """Test the get helper function for dotted-path access"""
        config = {
            "lst": {
                "solver": {"type": "navier-stokes"},
                "options": {"geometry_switch": 0},
            },
            "geometry": {"type": 0},
        }

        # Test successful access
        assert get(config, "lst.solver.type") == "navier-stokes"
        assert get(config, "geometry.type") == 0
        assert get(config, "lst.options.geometry_switch") == 0

        # Test default value when path doesn't exist
        assert get(config, "non.existent.path") is None
        assert get(config, "non.existent.path", "default") == "default"

        # Test when intermediate path is not a dict
        config["lst"]["solver"] = "not_a_dict"
        assert get(config, "lst.solver.type") is None

    def test_geometry_type_vs_switch_both_missing(self):
        """Test case where both geometry.type and lst.options.geometry_switch are missing"""
        config = {}

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "geometry.type"
        assert (
            "both geometry.type and lst.options.geometry_switch are missing"
            in errors[0].message
        )

    def test_geometry_type_missing_geometry_switch_present(self):
        """Test case where geometry.type is missing but lst.options.geometry_switch is set"""
        config = {"lst": {"options": {"geometry_switch": 0}}}

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "geometry.type"
        assert "geometry.type is missing" in errors[0].message

    def test_geometry_type_present_geometry_switch_missing(self):
        """Test case where geometry.type is set but lst.options.geometry_switch is missing"""
        config = {"geometry": {"type": 0}}

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.options.geometry_switch"
        assert "lst.options.geometry_switch was not set" in errors[0].message

    def test_geometry_type_0_switch_inconsistent(self):
        """Test flat plate geometry (type=0) with inconsistent geometry_switch"""
        config = {
            "geometry": {"type": 0},
            "lst": {"options": {"geometry_switch": 1}},  # Should be 0
        }

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.options.geometry_switch"
        assert (
            "geometry.type is 0 (flat plate) but lst.options.geometry_switch is 1"
            in errors[0].message
        )

    def test_geometry_type_1_switch_inconsistent(self):
        """Test cylinder geometry (type=1) with inconsistent geometry_switch"""
        config = {
            "geometry": {"type": 1},
            "lst": {"options": {"geometry_switch": 1}},  # Should be 0
        }

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.options.geometry_switch"
        assert (
            "geometry.type is 1 (cylinder) but lst.options.geometry_switch is 1"
            in errors[0].message
        )

    def test_geometry_type_2_switch_inconsistent(self):
        """Test cone geometry (type=2) with inconsistent geometry_switch"""
        config = {
            "geometry": {"type": 2},
            "lst": {"options": {"geometry_switch": 0}},  # Should be 1
        }

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.options.geometry_switch"
        assert (
            "geometry.type is 2 (cone) but lst.options.geometry_switch is 0"
            in errors[0].message
        )

    def test_geometry_type_3_switch_inconsistent(self):
        """Test generalized axisymmetric geometry (type=3) with inconsistent geometry_switch"""
        config = {
            "geometry": {"type": 3},
            "lst": {"options": {"geometry_switch": 0}},  # Should be 1
        }

        errors, warnings = check_consistency(config, enabled=["geometry_switch_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.options.geometry_switch"
        assert (
            "geometry.type is 3 (generalized axisymmetric) but lst.options.geometry_switch is 0"
            in errors[0].message
        )

    def test_consistent_geometry_configurations(self):
        """Test all consistent geometry type and switch combinations"""
        consistent_configs = [
            {
                "geometry": {"type": 0},
                "lst": {"options": {"geometry_switch": 0}},
            },  # Flat plate
            {
                "geometry": {"type": 1},
                "lst": {"options": {"geometry_switch": 0}},
            },  # Cylinder
            {
                "geometry": {"type": 2},
                "lst": {"options": {"geometry_switch": 1}},
            },  # Cone
            # Generalized axisymmetric
            {"geometry": {"type": 3}, "lst": {"options": {"geometry_switch": 1}}},
        ]

        for config in consistent_configs:
            errors, warnings = check_consistency(
                config, enabled=["geometry_switch_check"]
            )
            assert len(errors) == 0
            assert len(warnings) == 0

    def test_check_consistency_with_disabled_checks(self):
        """Test that checks can be selectively disabled"""
        config = {}  # Will trigger error when geometry_switch_check is run

        # Run with specific check disabled
        errors, warnings = check_consistency(config, enabled=[])

        assert len(errors) == 0
        assert len(warnings) == 0

    def test_format_report_with_errors_and_warnings(self):
        """Test format_report with both errors and warnings"""
        errors = [
            Issue(
                level=IssueLevel.ERROR,
                path="test.path1",
                message="Test error message",
                hint="Fix it",
            )
        ]
        warnings = [
            Issue(
                level=IssueLevel.WARNING,
                path="test.path2",
                message="Test warning message",
                hint="Consider fixing",
            )
        ]

        report = format_report(errors, warnings)
        assert "Consistency errors:" in report
        assert "Test error message" in report
        assert "Consistency warnings:" in report
        assert "Test warning message" in report

    def test_format_report_with_only_errors(self):
        """Test format_report with only errors"""
        errors = [
            Issue(
                level=IssueLevel.ERROR,
                path="test.path",
                message="Test error message",
                hint="Fix it",
            )
        ]
        warnings = []

        report = format_report(errors, warnings)
        assert "Consistency errors:" in report
        assert "Test error message" in report
        assert "Consistency warnings:" not in report

    def test_format_report_with_only_warnings(self):
        """Test format_report with only warnings"""
        errors = []
        warnings = [
            Issue(
                level=IssueLevel.WARNING,
                path="test.path",
                message="Test warning message",
                hint="Consider fixing",
            )
        ]

        report = format_report(errors, warnings)
        assert "Consistency warnings:" in report
        assert "Test warning message" in report
        assert "Consistency errors:" not in report

    def test_format_report_no_issues(self):
        """Test format_report with no issues"""
        errors = []
        warnings = []

        report = format_report(errors, warnings)
        assert report == "no consistency issues found"

    def test_issue_string_representation(self):
        """Test Issue class string representation"""
        # Test error without hint
        error_issue = Issue(level=IssueLevel.ERROR, path="test.path", message="Test error")
        error_str = str(error_issue)
        assert "[ERROR] test.path: Test error" == error_str

        # Test warning with hint
        warn_issue = Issue(
            level=IssueLevel.WARNING, path="test.path", message="Test warning", hint="Fix hint"
        )
        warn_str = str(warn_issue)
        assert "[WARNING] test.path: Test warning\n Fix hint" == warn_str

    def test_generalized_flag_check_flat_plate_consistent(self):
        """Test generalized flag check for flat plate with correct setting"""
        config = {
            "geometry": {"type": 0},
            "lst": {"solver": {"generalized": 1}},
        }

        errors, warnings = check_consistency(config, enabled=["generalized_flag_check"])

        assert len(errors) == 0
        assert len(warnings) == 0

    def test_generalized_flag_check_flat_plate_inconsistent(self):
        """Test generalized flag check for flat plate with wrong setting"""
        config = {
            "geometry": {"type": 0},
            "lst": {"solver": {"generalized": 0}},
        }

        errors, warnings = check_consistency(config, enabled=["generalized_flag_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.solver.generalized"
        assert (
            "geometry.type is 0 (flat plate) but lst.solver.generalized is 0"
            in errors[0].message
        )

    def test_generalized_flag_check_conical_geometry_warning(self):
        """Test generalized flag check warns for conical geometries with generalized=1"""
        config = {
            "geometry": {"type": 2},  # cone
            "lst": {"solver": {"generalized": 1}},
        }

        errors, warnings = check_consistency(config, enabled=["generalized_flag_check"])

        assert len(errors) == 0
        assert len(warnings) == 1
        assert warnings[0].level is IssueLevel.WARNING
        assert warnings[0].path == "lst.solver.generalized"
        assert (
            "geometry.type is 2 (cone) but lst.solver.generalized is 1"
            in warnings[0].message
        )

    def test_generalized_flag_check_missing_flag_for_flat_plate(self):
        """Test generalized flag check when flag is missing for flat plate"""
        config = {
            "geometry": {"type": 0},
            # lst.solver.generalized is missing
        }

        errors, warnings = check_consistency(config, enabled=["generalized_flag_check"])

        assert len(errors) == 1
        assert len(warnings) == 0
        assert errors[0].level is IssueLevel.ERROR
        assert errors[0].path == "lst.solver.generalized"
        assert (
            "lst.solver.generalized is not set for flat plate geometry"
            in errors[0].message
        )
