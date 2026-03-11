import sys
import pytest


class TestLSTTools:
    """Test suite for lst_tools main package"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        # Save original modules
        original_modules = dict(sys.modules)

        # Remove lst_tools modules
        modules_to_remove = [k for k in sys.modules.keys() if k.startswith("lst_tools")]
        for module in modules_to_remove:
            del sys.modules[module]

        yield

        # Restore original modules
        sys.modules.clear()
        sys.modules.update(original_modules)

    def test_imports(self):
        """Test that all expected functions are imported correctly"""
        from lst_tools import (
            read_flow_conditions,
            curvature,
            curvilinear_coordinate,
            surface_angle,
        )

        assert read_flow_conditions is not None
        assert curvature is not None
        assert curvilinear_coordinate is not None
        assert surface_angle is not None
