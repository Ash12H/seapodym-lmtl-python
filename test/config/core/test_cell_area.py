import numpy as np
import pint
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.config.core import cell_area


@pytest.fixture()
def simple_forcing() -> xr.DataArray:
    """Simple data with shape (10, 10, 10, 3)."""
    coords_layer = coordinates.new_layer()
    coords_latitude = coordinates.new_latitude(np.arange(0, 10, 1))
    coords_longitude = coordinates.new_longitude(np.arange(0, 10, 1))
    coords_time = coordinates.new_time(
        xr.cftime_range(start="2000-01-01", end="2000-01-10", freq="D")
    )
    forcing = xr.DataArray(
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
    forcing.data[:, :5, :5, :] = np.nan
    return forcing


@pytest.fixture()
def tolerance() -> pint.Quantity:
    """The tolerance for the tests."""
    return 10 * pint.application_registry.meters


class TestCellArea:

    def test_haversine_distance(self, tolerance: pint.Quantity):

        zero_degree = 0 * pint.application_registry.degrees
        one_degree = 1 * pint.application_registry.degrees

        hav_dist_zero = cell_area.haversine_distance(
            min_latitude=zero_degree,
            max_latitude=zero_degree,
            min_longitude=zero_degree,
            max_longitude=zero_degree,
        )
        assert hav_dist_zero.units == pint.application_registry.meters
        assert np.isclose(
            hav_dist_zero, 0 * pint.application_registry.m, atol=tolerance
        )

        hav_dist_equator = cell_area.haversine_distance(
            min_latitude=zero_degree,
            max_latitude=zero_degree,
            min_longitude=zero_degree,
            max_longitude=one_degree,
        )

        assert np.isclose(
            hav_dist_equator, 111_200 * pint.application_registry.m, atol=tolerance
        )

        hav_dist_equator = cell_area.haversine_distance(
            min_latitude=90 * one_degree,
            max_latitude=90 * one_degree,
            min_longitude=zero_degree,
            max_longitude=one_degree,
        )

        assert np.isclose(
            hav_dist_equator, 0 * pint.application_registry.m, atol=tolerance
        )

        hav_dist_equator = cell_area.haversine_distance(
            min_latitude=-90 * one_degree,
            max_latitude=-90 * one_degree,
            min_longitude=zero_degree,
            max_longitude=one_degree,
        )

        assert np.isclose(
            hav_dist_equator, 0 * pint.application_registry.m, atol=tolerance
        )

        with pytest.raises(ValueError):  # noqa: PT011
            cell_area.haversine_distance(
                min_latitude=zero_degree,
                max_latitude=-zero_degree,
                min_longitude=zero_degree,
                max_longitude=180 * one_degree,
            )
