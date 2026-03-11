import os
import sys
from lst_tools.data_io.read_flow_conditions import read_flow_conditions, _first_number

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")


class TestReadFlowConditions:
    """Tests the read_flow_conditions functionality"""

    def test_read_flow_condition(self):
        """
        Ensure that it properly reads flow conditions
        """
        from mocks import MOCK_FLOW_CONDITIONS_DAT, skip_if_missing

        skip_if_missing(MOCK_FLOW_CONDITIONS_DAT)
        result = read_flow_conditions(fpath=MOCK_FLOW_CONDITIONS_DAT)
        assert result == {
            "rgas": 287.15,
            "cp": 1005.0250000000001,
            "cv": 717.8750000000001,
            "gamma": 1.4,
            "pr": 0.71,
            "mach": 5.3,
            "pres_0": 1362869.3601819817,
            "pres_inf": 1827.723761743531,
            "temp_0": 450.0,
            "temp_inf": 67.99637352674526,
            "dens_0": 10.547095866906432,
            "dens_inf": 0.0936086509584126,
            "mu": 4.5824752572e-06,
            "re1": 17900000.0,
            "uvel_inf": 876.2684459642295,
            "a_inf": 165.33366904985462,
            "lref": 1.0,
            "tref": 0.001141202795337,
            "h0": 0.4522612500000001,
            "eta": 3.633793358036811e-06,
            "tau_eta": 2.697343841514764e-07,
        }

    def test_first_number(self):
        """
        Extracts the first number found
        """
        assert _first_number("hello world") is None
        assert _first_number("") is None
        assert _first_number("1") == 1
        assert _first_number("1.1") == 1.1
        assert _first_number("1e-9") == 1e-9
        assert _first_number(".3") == 0.3
        assert _first_number(".3.4") == 0.3
        assert _first_number("1.") == 1
