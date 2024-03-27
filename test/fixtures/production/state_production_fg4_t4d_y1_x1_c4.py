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
        attrs={"long_name": "Functional group", "standard_name": "functional_group"},
    )


@pytest.fixture()
def cohort_4():
    return coordinates.new_cohort(np.arange(4))


@pytest.fixture()
def mask_temperature(fgroup_4, time_4days, latitude_single, longitude_single, cohort_4):
    coordinates = {
        CoordinatesLabels.functional_group: fgroup_4,
        "time": time_4days,
        "latitude": latitude_single,
        "longitude": longitude_single,
        CoordinatesLabels.cohort: cohort_4,
    }
    return xr.DataArray(
        dims=coordinates.keys(),
        coords=coordinates,
        data=np.full(tuple(i.size for i in coordinates.values()), True, dtype=bool),
    )


@pytest.fixture()
def primary_production_by_fgroup(fgroup_4, time_4days, latitude_single, longitude_single):
    coordinates = {
        CoordinatesLabels.functional_group: fgroup_4,
        "time": time_4days,
        "latitude": latitude_single,
        "longitude": longitude_single,
    }
    return xr.DataArray(
        dims=coordinates.keys(),
        coords=coordinates,
        data=np.full(tuple(i.size for i in coordinates.values()), 1.0, dtype=float),
    )


@pytest.fixture()
def initial_condition_production(fgroup_4, latitude_single, longitude_single, cohort_4):
    coordinates = {
        CoordinatesLabels.functional_group: fgroup_4,
        "latitude": latitude_single,
        "longitude": longitude_single,
        CoordinatesLabels.cohort: cohort_4,
    }
    return xr.DataArray(
        dims=coordinates.keys(),
        coords=coordinates,
        data=np.full(tuple(i.size for i in coordinates.values()), 1.0, dtype=float),
    )


@pytest.fixture()
def timestep_number(fgroup_4, cohort_4):
    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group, CoordinatesLabels.cohort),
        coords={CoordinatesLabels.functional_group: fgroup_4, CoordinatesLabels.cohort: cohort_4},
        data=np.full((4, 4), 1, dtype=int),
    )


@pytest.fixture()
def state_production_fg4_t4d_y1_x1_c4(
    mask_temperature, primary_production_by_fgroup, initial_condition_production, timestep_number
):
    """
    A dataset for preproduction state.
    Dims : 4 functional groups, 4 days, 1 latitude, 1 longitude, 4 cohorts.
    """
    # TODO(Jules): Prepare dataset for production process
    return xr.Dataset(
        {
            PreproductionLabels.mask_temperature: mask_temperature,
            PreproductionLabels.primary_production_by_fgroup: primary_production_by_fgroup,
            ConfigurationLabels.initial_condition_production: initial_condition_production,
            ConfigurationLabels.timesteps_number: timestep_number,
        }
    )
