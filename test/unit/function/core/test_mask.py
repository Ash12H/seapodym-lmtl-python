import numpy as np
import pytest
import xarray as xr

from seapopym.function.core import mask
from seapopym.standard import coordinates


@pytest.fixture()
def simple_forcing() -> xr.DataArray:
    """Simple data with shape (10, 10, 10, 3)."""
    coords_layer = coordinates.new_layer()
    coords_latitude = coordinates.new_latitude(np.arange(0, 10, 1))
    coords_longitude = coordinates.new_longitude(np.arange(0, 10, 1))
    coords_time = coordinates.new_time(xr.cftime_range(start="2000-01-01", end="2000-01-10", freq="D"))
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


class TestLandmask:
    def test_landmask(self, simple_forcing):
        res = mask.landmask_from_nan(simple_forcing)
        # Keep spatial dimensions only
        assert res.shape == simple_forcing.isel(time=0).shape
        # Only boolean values
        assert res.dtype == bool
