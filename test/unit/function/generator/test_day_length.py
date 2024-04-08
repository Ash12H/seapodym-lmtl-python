import cf_xarray  # noqa: F401  # noqa: F401
import numpy as np
import pint
import pytest
import xarray as xr

from seapopym.function.generator.day_length import day_length_forsythe, day_length_kernel
from seapopym.standard import coordinates
from seapopym.standard.labels import CoordinatesLabels


@pytest.fixture()
def simple_forcing() -> xr.DataArray:
    """Simple data with shape (10, 10, 10, 3)."""
    coords_layer = coordinates.new_layer()
    coords_latitude = coordinates.new_latitude([-90, 0, 90])
    coords_longitude = coordinates.new_longitude([-180, 0, 180])
    coords_time = coordinates.new_time(xr.cftime_range(start="2000-01-01", end="2000-01-02", freq="D"))
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


class TestMaskByFgroup:
    def test_simple_working(self, state_preprod_fg4_t4d_y1_x1_z3):
        kernel = day_length_kernel()
        results = kernel.run(state_preprod_fg4_t4d_y1_x1_z3)
        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf.coords
        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape

        assert results.dtype == float

        assert np.all(results > 0)
        assert np.all(results <= 24)

    def test_simple_working_with_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {CoordinatesLabels.functional_group: 1}
        data = state_preprod_fg4_t4d_y1_x1_z3.chunk(chunk)
        kernel = day_length_kernel(chunk=chunk)
        results = kernel.run(data).compute()
        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf.coords
        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape

        assert results.dtype == float

        assert np.all(results > 0)
        assert np.all(results <= 24)


class TestDayLength:
    def test_day_length_forsythe_p_zero(self, tolerance: pint.Quantity):
        zero_hour = 0 * pint.application_registry.hour
        twelve_hour = 12 * pint.application_registry.hour
        twenty_four_hour = 24 * pint.application_registry.hour

        def wrap_to_hour(x, y, z):
            return day_length_forsythe(x, y, z) * pint.application_registry.hour

        # START YEAR
        assert np.isclose(wrap_to_hour(0, 0, 0), twelve_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(90, 0, 0), zero_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(-90, 0, 0), twenty_four_hour, atol=tolerance)
        # MID YEAR
        assert np.isclose(wrap_to_hour(0, 180, 0), twelve_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(90, 180, 0), twenty_four_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(-90, 180, 0), zero_hour, atol=tolerance)
        # NO LEAP
        assert np.isclose(wrap_to_hour(0, 365, 0), twelve_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(90, 365, 0), zero_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(-90, 365, 0), twenty_four_hour, atol=tolerance)
        # ALL LEAP
        assert np.isclose(wrap_to_hour(0, 366, 0), twelve_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(90, 366, 0), zero_hour, atol=tolerance)
        assert np.isclose(wrap_to_hour(-90, 366, 0), twenty_four_hour, atol=tolerance)
