"""All the functions used to generate or modify the forcings."""

from typing import Iterable

import xarray as xr
from dask.distributed import Client

from seapodym_lmtl_python.pre_production.core import landmask

# TODO(Jules): standardize the parameters names(inv_lambda_max, inv_lambda_rate, tr_max, tr_rate, ...)

# --- Pre production functions --- #


def mask_by_fgroup(day_layers: xr.DataArray, night_layers: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """


def compute_daylength(
    time: xr.DataArray,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
) -> xr.DataArray:
    """
    Use the grid and the time to generate the daylength. Should have a look to a the C++ model -> a python lib is
    available.

    Output
    ------
    - daylength [time, latitude, longitude]

    ---
    Information available in `xarray.cftime_range()` function.
    """
    pass


def average_temperature_by_fgroup(
    daylength: xr.DataArray,
    mask: xr.DataArray,
    day_layer: xr.DataArray,
    night_layer: xr.DataArray,
    temperature: xr.DataArray,
) -> xr.DataArray:
    """
    Depend on:
    - compute_daylength
    - mask_by_fgroup.

    Input
    -----
    - mask_by_fgroup()      [time, latitude, longitude]
    - compute_daylength()   [functional_group, latitude, longitude]
    - day/night_layer       [functional_group]
    - temperature           [time, latitude, longitude, layer]

    Output
    ------
    - avg_temperature [functional_group, time, latitude, longitude]
    """
    pass


def apply_coefficient_to_primary_production(
    primary_production: xr.DataArray,
    functional_group_coefficient: xr.DataArray,
) -> xr.DataArray:
    """
    It is equivalent to generate the fisrt cohort of pre-production.

    Input
    -----
    - primary_production [time, latitude, longitude]
    - functional_group_coefficient [functional_group]

    Output
    ------
    - primary_production [functional_group, time, latitude, longitude]
    """
    pass


def min_temperature_by_cohort(
    mean_timestep: xr.DataArray,
    tr_max: xr.DataArray,
    tr_rate: xr.DataArray,
) -> xr.DataArray:
    """
    Define the minimal temperature of a cohort to be recruited.

    Input
    -----
    - mean_timestep [functional_group, cohort_age]
    - tr_max [functional_group]
    - tr_rate [functional_group]

    Output
    ------
    - min_temperature_by_cohort [functional_group, cohort_age] : a datarray with cohort_age as coordinate and
    minimum temperature as value.
    """
    pass


def mask_temperature_by_cohort_by_functional_group(
    min_temperature_by_cohort: xr.DataArray, average_temperature: xr.DataArray
) -> xr.DataArray:
    """
    It uses the min_temperature_by_cohort.

    Depend on
    ---------
    - min_temperature_by_cohort()
    - average_temperature_by_fgroup()

    Input
    -----
    - min_temperature_by_cohort [cohort_age]
    - average_temperature [functional_group, time, latitude, longitude]

    Output
    ------
    - mask_temperature_by_cohort_by_functional_group [functional_group, time, latitude, longitude, cohort_age]

    NOTE(Jules): Warning : average temperature by functional group (because of daily vertical migration) and not by
    layer. We therefore have a function with a high cost in terms of computation and memory space.

    """
    pass


def compute_cell_area(latitude: xr.DataArray, longitude: xr.DataArray):
    """
    Compute the cell area from the latitude and longitude.

    Input
    ------
    - latitude [latitude]
    - longitude [longitude]
    - (Grid ?)

    Output
    ------
    - cell_area [latitude, longitude]
    """
    pass


def compute_mortality_field(
    average_temperature: xr.DataArray, inv_lambda_max: xr.DataArray, inv_lambda_rate: xr.DataArray
) -> xr.DataArray:
    """
    Use the relation between temperature and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature_by_fgroup()

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - inv_lambda_max [functional_group]
    - inv_lambda_rate [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]
    """
    pass
