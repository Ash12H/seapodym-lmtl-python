import numpy as np
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.config.core import landmask


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


class TestLandmask:

    def test_landmask(self, simple_forcing):
        mask = landmask.landmask_from_nan(simple_forcing)
        assert mask.shape == simple_forcing.isel(time=0).shape
        assert set(mask.data.flatten()) == {0, 1}
        assert set(mask.data.flatten()) == {False, True}
