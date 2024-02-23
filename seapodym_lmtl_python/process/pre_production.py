"""All the functions used to generate or modify the forcings."""

from typing import Callable, Iterable

import xarray as xr
from dask.distributed import Client

# TODO(Jules): standardize the parameters names(inv_lambda_max, inv_lambda_rate, tr_max, tr_rate, ...)

# --- Pre production functions --- #


def landmask_by_fgroup(
    day_layers: Iterable[int], night_layers: Iterable[int], landmask: xr.DataArray
) -> xr.DataArray:
    """
    The `landmask` has at least 3 dimensions (lat, lon, layer). We are only using the nan cells to generate the
    landmask by functional group.

    `landmask` can be:
    - temperature
    - user landmask

    Output
    ------
    - landmask  [functional_group, latitude, longitude, layer] -> boolean
    """
    pass


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

    TODO(Jules): 2. This forcing can be given by the user. That way, this function isn't called.

    Information available in `xarray.cftime_range()` function.
    """
    pass


def average_temperature_by_fgroup(
    daylength: xr.DataArray,
    landmask: xr.DataArray,
    day_layer: xr.DataArray,
    night_layer: xr.DataArray,
    temperature: xr.DataArray,
) -> xr.DataArray:
    """
    Depend on:
    - compute_daylength
    - landmask_by_fgroup.

    Input
    -----
    - landmask_by_fgroup()  [time, latitude, longitude]
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
    global_coefficient: float,
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

    TODO(Jules): Answer the question about the primary_production which is defined in m**3.
    """
    pass


def min_temperature_by_cohort(
    cohort_coordinates: xr.DataArray,
    tr_max: float,
    tr_rate: float,
) -> xr.DataArray:
    """
    Define the minimal temperature of a cohort to be recruited.

    Input
    -----
    - cohort_coordinates [cohort_age] (The coordinates)
    - tr_max float
    - tr_rate float

    Output
    ------
    - min_temperature_by_cohort {cohort_age: min_temperature} : a datarray with cohort_age as coordinate and
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
    average_temperature: xr.DataArray, inv_lambda_max: float, inv_lambda_rate: float
) -> xr.DataArray:
    """
    Use the relation between temperature and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature_by_fgroup()

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - inv_lambda_max float
    - inv_lambda_rate float

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]
    """
    pass


# --- Wrapper --- #


def process(
    client: Client, configuration: xr.Dataset, kernel: None | list[Callable]
) -> xr.Dataset:
    """
    Wraps all the pre-production functions.

    It generates all the forcings used in the production and post-production processes.

    Parameters
    ----------
    client : None | Client
        The dask client to use.
    configuration : xr.Dataset
        The model configuration that contains both forcing and parameters.
    kernel : None | list[Callable]
        The list of pre-production functions to use. If None, the default list is used.

    """
    if kernel is None:
        kernel = [
            landmask_by_fgroup,
            compute_daylength,
            average_temperature_by_fgroup,
            apply_coefficient_to_primary_production,
            min_temperature_by_cohort,
            mask_temperature_by_cohort_by_functional_group,
            compute_cell_area,
            compute_mortality_field,
        ]

    # ...   TODO(Jules): Apply all pre-production functions using forcing and configuration parameters. The results are
    # ...   then integrated into the latter.

    # COPIED FROM LMTL FILE ------------------

    # landmask = client.submit(landmask_by_fgroup, None, None, None)
    # day_length = client.submit(compute_daylength, None, None, None)
    # avg_temperature = client.submit(
    #     average_temperature_by_fgroup,
    #     day_length,
    #     landmask,
    #     None,
    #     None,
    #     None,
    # )
    # pre_production = client.submit(
    #     apply_coefficient_to_primary_production, None, None, None
    # )
    # min_temperature = client.submit(min_temperature_by_cohort, None, None, None)
    # mask_temperature = client.submit(
    #     mask_temperature_by_cohort_by_functional_group,
    #     min_temperature,
    #     avg_temperature,
    # )
    # cell_area = client.submit(compute_cell_area, None, None)
    # mortality = client.submit(compute_mortality_field, avg_temperature, None, None)

    return configuration
