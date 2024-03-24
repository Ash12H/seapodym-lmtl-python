import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.pre_production import compute_daylength
from seapopym.standard import coordinates
from seapopym.standard.labels import ConfigurationLabels


@pytest.fixture()
def wrong_time():
    return xr.DataArray(
        dims=("time",),
        coords={"time": xr.cftime_range(start="2020", freq="D", periods=2)},
        data=xr.cftime_range(start="2020", freq="D", periods=2),
    )


@pytest.fixture()
def right_time():
    return coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=2))


@pytest.fixture()
def latitude():
    return coordinates.new_latitude(np.arange(-90, 0, 90))


@pytest.fixture()
def longitude():
    return coordinates.new_longitude(np.arange(-120, 0, 120))


class TestMaskByFgroup:
    def test_wrong_time(self, wrong_time, latitude, longitude):
        with pytest.raises(ValueError, match="time must have a attrs={..., 'axis':'T', ...}"):
            compute_daylength(
                time=wrong_time,
                latitude=latitude,
                longitude=longitude,
            )

    def test_right_time(self, right_time, latitude, longitude):
        daylength = compute_daylength(
            time=right_time,
            latitude=latitude,
            longitude=longitude,
        )

        assert isinstance(daylength, xr.DataArray)
        for dim in ("T", "Y", "X"):
            assert dim in daylength.cf.coords
        assert daylength.shape == (right_time.cf["T"].size, latitude.cf["Y"].size, longitude.cf["X"].size)
        assert daylength.dtype == float
        assert np.all(daylength > 0)
        assert np.all(daylength <= 24)
