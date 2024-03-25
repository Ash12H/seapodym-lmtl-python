import cf_xarray  # noqa: F401
import numpy as np
import pint
import pint_xarray  # noqa: F401
import pytest
import xarray as xr

from seapopym.function.generator.pre_production import average_temperature_by_fgroup
from seapopym.standard import coordinates
from seapopym.standard.labels import CoordinatesLabels

time = coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=1))
latitude = coordinates.new_latitude(np.array([0]))
longitude = coordinates.new_longitude(np.array([0]))
fgroup = xr.DataArray(
    dims=(CoordinatesLabels.functional_group,),
    coords={CoordinatesLabels.functional_group: np.arange(4)},
    data=np.arange(4, dtype=int),
)
layer = coordinates.new_layer()


@pytest.fixture()
def daylength():
    return (
        xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            data=np.full((time.size, latitude.size, longitude.size), 0.5, dtype=float),
        )
        .pint.quantify("day")
        .pint.dequantify()
    )


@pytest.fixture()
def mask():
    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group, "latitude", "longitude"),
        coords={CoordinatesLabels.functional_group: fgroup, "latitude": latitude, "longitude": longitude},
        data=np.full((fgroup.size, latitude.size, longitude.size), True, dtype=bool),
    )


@pytest.fixture()
def day_layer():
    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group,),
        coords={CoordinatesLabels.functional_group: np.arange(4)},
        data=np.array([1, 1, 2, 2], dtype=int),
    )


@pytest.fixture()
def night_layer():
    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group,),
        coords={CoordinatesLabels.functional_group: np.arange(4)},
        data=np.array([1, 2, 2, 1], dtype=int),
    )


@pytest.fixture()
def temperature():
    data = [np.full((time.size, latitude.size, longitude.size), i, dtype=float) for i in range(layer.size)]
    data = np.stack(data, axis=-1)
    data = xr.DataArray(
        dims=("time", "latitude", "longitude", "layer"),
        coords={"time": time, "latitude": latitude, "longitude": longitude, "layer": layer},
        data=data,
    )
    return (data * pint.application_registry("degC")).pint.dequantify()


class TestMaskByFgroup:
    def test_mask_by_fgroup(
        self,
        daylength: xr.DataArray,
        mask: xr.DataArray,
        day_layer: xr.DataArray,
        night_layer: xr.DataArray,
        temperature: xr.DataArray,
    ):
        fgroup_mask = average_temperature_by_fgroup(
            daylength=daylength, mask=mask, day_layer=day_layer, night_layer=night_layer, temperature=temperature
        )

        assert isinstance(fgroup_mask, xr.DataArray)
        for dim in (CoordinatesLabels.functional_group, "T", "Y", "X"):
            assert dim in fgroup_mask.cf.coords

        assert fgroup_mask.shape == (fgroup.size, time.size, latitude.size, longitude.size)

        assert fgroup_mask.dtype == float

        assert np.array_equal(fgroup_mask.cf.isel(T=0, Y=0, X=0), [0, 0.5, 1.0, 0.5])
