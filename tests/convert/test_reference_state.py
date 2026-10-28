"""Tests for the meanflow reference state: one state scales the profiles and every station header."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lst_tools.config.schema import Config, MeanflowConversion
from lst_tools.convert.lastrac import convert_meanflow, resolve_reference_state
from lst_tools.core import Flow, Grid
from lst_tools.geometry.kinds import GeometryKind

# --------------------------------------------------
# test data: a dimensional baseflow (2 wall-normal points x 3 stations)
# --------------------------------------------------
FREESTREAM = {
    "re1": 1.08e7,
    "pres_inf": 611.0,
    "temp_inf": 51.2,
    "uvel_inf": 861.0,
    "dens_inf": 0.0416,
}
CUSTOM = {
    "uvel_ref": 800.0,
    "temp_ref": 120.0,
    "dens_ref": 0.03,
    "re1_ref": 5.0e6,
}
RGAS = 287.15


def build_config(meanflow: dict | None = None) -> Config:
    """Flat-plate config with the freestream above and the given meanflow_conversion keys."""

    # meanflow_conversion keys on top of the defaults
    meanflow_dict = {"i_s": 0, "i_e": None, "d_i": 1, "set_v_zero": False}
    meanflow_dict.update(meanflow or {})

    config = Config.from_dict(
        {
            "geometry": {"type": GeometryKind.FLAT_PLATE, "is_body_fitted": True, "l_ref": 1.0},
            "flow_conditions": {**FREESTREAM, "rgas": RGAS},
            "meanflow_conversion": meanflow_dict,
        }
    )
    return config


def dimensional_flow() -> Flow:
    """Dimensional profiles: velocities in m/s, temperature in K, pressure in Pa."""

    # flat wall at y = 0, one outer point at y = 0.1
    grid = Grid(
        x=np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]),
        y=np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]),
        z=None,
        cfg={},
    )
    fields = {
        "uvel": np.array([[0.0, 0.0, 0.0], [700.0, 760.0, 820.0]]),
        "vvel": np.array([[0.0, 0.0, 0.0], [5.0, 6.0, 7.0]]),
        "temp": np.array([[300.0, 300.0, 300.0], [140.0, 130.0, 120.0]]),
        "pres": np.array([[900.0, 950.0, 1000.0], [900.0, 950.0, 1000.0]]),
    }
    flow = Flow(grid=grid, fields=fields, attrs={})
    return flow


def run_conversion(config: Config, flow: Flow) -> MagicMock:
    """Run convert_meanflow with geometry helpers and the writer mocked; return the writer."""

    # geometry of a flat plate: no curvature, no surface angle
    n_stations = flow.grid.x.shape[1]
    zeros = np.zeros(n_stations)
    progress_ctx = MagicMock()
    progress_ctx.__enter__.return_value = lambda: None

    with (
        patch("lst_tools.convert.lastrac.LastracWriter") as writer_class,
        patch("lst_tools.convert.lastrac.radius", return_value=zeros),
        patch("lst_tools.convert.lastrac.curvature", return_value=zeros),
        patch("lst_tools.convert.lastrac.surface_angle", return_value=zeros),
        patch(
            "lst_tools.convert.lastrac.curvilinear_coordinate",
            return_value=np.arange(n_stations, dtype=float),
        ),
        patch("lst_tools.convert.lastrac.progress", return_value=progress_ctx),
    ):
        writer = MagicMock()
        writer_class.return_value = writer
        convert_meanflow(grid=flow.grid, flow=flow, out="meanflow.bin", cfg=config)

    return writer


def station_records(writer: MagicMock) -> list[dict]:
    """Collect each station's header values and its written vectors.

    Per station the converter writes, in order: eta, uvel, vvel, wvel, temp, pres.
    """

    # headers in station order
    headers = [call.kwargs for call in writer.write_station_header.call_args_list]

    # vectors in station order, six per station
    vectors = [np.asarray(call.args[0]) for call in writer.write_station_vector.call_args_list]
    names = ("eta", "uvel", "vvel", "wvel", "temp", "pres")
    records = []
    for index, header in enumerate(headers):
        station_vectors = vectors[6 * index : 6 * index + 6]
        record = dict(zip(names, station_vectors, strict=True))
        record["header"] = header
        records.append(record)

    return records


# --------------------------------------------------
# consistency: profiles x header reference = dimensional input, at every station
# --------------------------------------------------
@pytest.mark.parametrize(
    ("meanflow", "expected"),
    [
        ({}, (FREESTREAM["uvel_inf"], FREESTREAM["temp_inf"], FREESTREAM["dens_inf"], FREESTREAM["re1"])),
        (
            {"reference": "custom", **CUSTOM},
            (CUSTOM["uvel_ref"], CUSTOM["temp_ref"], CUSTOM["dens_ref"], CUSTOM["re1_ref"]),
        ),
    ],
)
def test_profiles_and_station_headers_use_the_same_reference(meanflow: dict, expected: tuple) -> None:
    """Every station header carries the reference the profiles were divided by."""

    # convert the dimensional baseflow
    flow = dimensional_flow()
    writer = run_conversion(build_config(meanflow), flow)
    records = station_records(writer)
    uvel_ref, temp_ref, dens_ref, re1_ref = expected

    # check every station
    assert len(records) == 3
    for i_station, record in enumerate(records):
        header = record["header"]

        # header reference state
        assert header["stat_uvel"] == pytest.approx(uvel_ref)
        assert header["stat_temp"] == pytest.approx(temp_ref)
        assert header["stat_dens"] == pytest.approx(dens_ref)
        assert header["re1"] == pytest.approx(re1_ref)

        # written profile times the header reference gives back the dimensional input
        np.testing.assert_allclose(
            record["uvel"] * header["stat_uvel"], flow.field("uvel")[:, i_station], rtol=1e-12
        )
        np.testing.assert_allclose(
            record["temp"] * header["stat_temp"], flow.field("temp")[:, i_station], rtol=1e-12
        )
        dynamic_pressure = header["stat_dens"] * header["stat_uvel"] ** 2
        np.testing.assert_allclose(
            record["pres"] * dynamic_pressure, flow.field("pres")[:, i_station], rtol=1e-12
        )


def test_custom_reference_pressure_from_perfect_gas() -> None:
    """The file header pressure of a custom reference is dens_ref * rgas * temp_ref."""

    # convert with a custom reference
    writer = run_conversion(build_config({"reference": "custom", **CUSTOM}), dimensional_flow())

    # check the file header
    header_kwargs = writer.write_header.call_args.kwargs
    expected = CUSTOM["dens_ref"] * RGAS * CUSTOM["temp_ref"]
    assert header_kwargs["stat_pres"] == pytest.approx(expected)


def test_freestream_reference_pressure_is_pres_inf() -> None:
    """The file header pressure of the freestream reference is pres_inf."""

    # convert with the default (freestream) reference
    writer = run_conversion(build_config(), dimensional_flow())

    # check the file header
    assert writer.write_header.call_args.kwargs["stat_pres"] == pytest.approx(FREESTREAM["pres_inf"])


# --------------------------------------------------
# resolving and validating the reference state
# --------------------------------------------------
def test_custom_reference_missing_values_rejected() -> None:
    """A custom reference names every value it is missing."""

    # custom reference with only a velocity
    config = build_config({"reference": "custom", "uvel_ref": 800.0})

    with pytest.raises(ValueError, match="temp_ref.*dens_ref.*re1_ref"):
        resolve_reference_state(config)


def test_freestream_reference_missing_value_rejected() -> None:
    """The freestream reference needs dens_inf (and the other freestream values)."""

    # freestream without a density
    config = build_config()
    config.flow_conditions.dens_inf = None

    with pytest.raises(ValueError, match="flow_conditions.dens_inf"):
        resolve_reference_state(config)


def test_non_positive_reference_rejected() -> None:
    """Reference values must be positive."""

    # custom reference with a zero temperature
    config = build_config({"reference": "custom", **CUSTOM, "temp_ref": 0.0})

    with pytest.raises(ValueError, match="temp_ref must be positive"):
        resolve_reference_state(config)


def test_meanflow_conversion_defaults_and_unknown_reference() -> None:
    """Conversion nondimensionalizes by the freestream by default; unknown references raise."""

    # defaults
    defaults = MeanflowConversion.from_dict({})
    assert defaults.nondimensionalize is True
    assert defaults.reference == "freestream"

    # unknown reference
    with pytest.raises(ValueError, match="'freestream' or 'custom'"):
        MeanflowConversion.from_dict({"reference": "edge"})


def test_failed_check_writes_no_file(tmp_path) -> None:
    """Non-dimensional input with nondimensionalize = true fails before the output file is opened."""

    # non-dimensional input (velocities ~ 1) under the default nondimensionalize = true
    flow = dimensional_flow()
    flow.fields["uvel"] = flow.fields["uvel"] / FREESTREAM["uvel_inf"]
    output_path = tmp_path / "meanflow.bin"
    zeros = np.zeros(3)

    with (
        patch("lst_tools.convert.lastrac.radius", return_value=zeros),
        patch("lst_tools.convert.lastrac.curvature", return_value=zeros),
        patch("lst_tools.convert.lastrac.surface_angle", return_value=zeros),
        patch("lst_tools.convert.lastrac.curvilinear_coordinate", return_value=np.arange(3.0)),
        pytest.raises(ValueError, match="double scaling"),
    ):
        convert_meanflow(grid=flow.grid, flow=flow, out=output_path, cfg=build_config())

    # check: no partial meanflow file left behind
    assert not output_path.exists()
