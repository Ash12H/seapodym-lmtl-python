import cf_xarray  # noqa: F401
import numpy as np
import pint
import pint_xarray  # noqa: F401
import pytest
import xarray as xr

from seapopym.standard import coordinates
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels


@pytest.fixture()
def time_4days() -> xr.DataArray:
    return coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=4))


@pytest.fixture()
def latitude_single():
    return coordinates.new_latitude(np.array([0]))


@pytest.fixture()
def longitude_single():
    return coordinates.new_longitude(np.array([0]))


@pytest.fixture()
def fgroup_4():
    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group,),
        coords={CoordinatesLabels.functional_group: np.arange(4)},
        data=np.arange(4, dtype=int),
    )


@pytest.fixture()
def layer():
    return coordinates.new_layer()


@pytest.fixture()
def daylength(time_4days, latitude_single, longitude_single):
    return (
        xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_4days, "latitude": latitude_single, "longitude": longitude_single},
            data=np.full((time_4days.size, latitude_single.size, longitude_single.size), 0.5, dtype=float),
        )
        .pint.quantify("day")
        .pint.dequantify()
    )


@pytest.fixture()
def mask_fgroup(fgroup_4, latitude_single, longitude_single):
    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group, "latitude", "longitude"),
        coords={
            CoordinatesLabels.functional_group: fgroup_4,
            "latitude": latitude_single,
            "longitude": longitude_single,
        },
        data=np.full((fgroup_4.size, latitude_single.size, longitude_single.size), True, dtype=bool),
    )


@pytest.fixture()
def global_mask(latitude_single, longitude_single, layer):
    return xr.DataArray(
        dims=("latitude", "longitude", "layer"),
        coords={"latitude": latitude_single, "longitude": longitude_single, "layer": layer},
        data=np.full((latitude_single.size, longitude_single.size, layer.size), True, dtype=bool),
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
def temperature(time_4days, latitude_single, longitude_single, layer):
    data = [
        np.full((time_4days.size, latitude_single.size, longitude_single.size), i, dtype=float)
        for i in range(layer.size)
    ]
    data = np.stack(data, axis=-1)
    data = xr.DataArray(
        dims=("time", "latitude", "longitude", "layer"),
        coords={"time": time_4days, "latitude": latitude_single, "longitude": longitude_single, "layer": layer},
        data=data,
    )
    return (data * pint.application_registry("degC")).pint.dequantify()


@pytest.fixture()
def state_preprod_fg4_t4d_y1_x1(daylength, day_layer, night_layer, temperature, mask_fgroup, global_mask):
    """
    A dataset for preproduction state.
    Dims : 4 functional groups, 4 days, 1 latitude, 1 longitude, 3 layers.
    """
    return xr.Dataset(
        {
            PreproductionLabels.day_length: daylength,
            PreproductionLabels.mask_by_fgroup: mask_fgroup,
            PreproductionLabels.global_mask: global_mask,
            ConfigurationLabels.day_layer: day_layer,
            ConfigurationLabels.night_layer: night_layer,
            ConfigurationLabels.temperature: temperature,
        }
    )
