import numpy as np
import pint
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.pre_production.core.day_length import day_length_forsythe


@pytest.fixture()
def simple_forcing() -> xr.DataArray:
    """Simple data with shape (10, 10, 10, 3)."""
    coords_layer = coordinates.new_layer()
    coords_latitude = coordinates.new_latitude([-90, 0, 90])
    coords_longitude = coordinates.new_longitude([-180, 0, 180])
    coords_time = coordinates.new_time(
        xr.cftime_range(start="2000-01-01", end="2000-01-02", freq="D")
    )
    return xr.DataArray(
        coords={
            "time": coords_time,
            "latitude": coords_latitude,
            "longitude": coords_longitude,
            "layer": coords_layer,
        },
        dims=["time", "latitude", "longitude", "layer"],
        data=np.full(
            (
                len(coords_time),
                len(coords_latitude),
                len(coords_longitude),
                len(coords_layer),
            ),
            1,
            dtype=float,
        ),
    )


@pytest.fixture()
def tolerance() -> pint.Quantity:
    """The tolerance for the tests."""
    return 1 * pint.application_registry.minute


class TestDayLength:

    def test_day_length_forsythe_p_zero(self, tolerance: pint.Quantity):
        zero_hour = 0 * pint.application_registry.hour
        twelve_hour = 12 * pint.application_registry.hour
        twenty_four_hour = 24 * pint.application_registry.hour

        # START YEAR
        assert np.isclose(day_length_forsythe(0, 0, 0), twelve_hour, atol=tolerance)
        assert np.isclose(day_length_forsythe(90, 0, 0), zero_hour, atol=tolerance)
        assert np.isclose(
            day_length_forsythe(-90, 0, 0), twenty_four_hour, atol=tolerance
        )
        # MID YEAR
        assert np.isclose(day_length_forsythe(0, 180, 0), twelve_hour, atol=tolerance)
        assert np.isclose(
            day_length_forsythe(90, 180, 0), twenty_four_hour, atol=tolerance
        )
        assert np.isclose(day_length_forsythe(-90, 180, 0), zero_hour, atol=tolerance)
        # NO LEAP
        assert np.isclose(day_length_forsythe(0, 365, 0), twelve_hour, atol=tolerance)
        assert np.isclose(day_length_forsythe(90, 365, 0), zero_hour, atol=tolerance)
        assert np.isclose(
            day_length_forsythe(-90, 365, 0), twenty_four_hour, atol=tolerance
        )
        # ALL LEAP
        assert np.isclose(day_length_forsythe(0, 366, 0), twelve_hour, atol=tolerance)
        assert np.isclose(day_length_forsythe(90, 366, 0), zero_hour, atol=tolerance)
        assert np.isclose(
            day_length_forsythe(-90, 366, 0), twenty_four_hour, atol=tolerance
        )
