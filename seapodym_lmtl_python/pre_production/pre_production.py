"""All the functions used to generate or modify the forcings."""

import numpy as np
import xarray as xr

from seapodym_lmtl_python.configuration.no_transport.configuration import NoTransportLabels
from seapodym_lmtl_python.pre_production.core import day_length

# TODO(Jules): standardize the parameters names(inv_lambda_max, inv_lambda_rate, tr_max, tr_rate, ...)

# --- Pre production functions --- #


def mask_by_fgroup(day_layers: xr.DataArray, night_layers: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """
    masks = []
    for i in day_layers[NoTransportLabels.fgroup]:
        day_pos = day_layers.sel(functional_group=i)
        night_pos = night_layers.sel(functional_group=i)

        day_mask = mask.cf.sel(Z=day_pos)
        night_mask = mask.cf.sel(Z=night_pos)
        masks.append(day_mask & night_mask)

    return xr.DataArray(
        coords={
            NoTransportLabels.fgroup: day_layers[NoTransportLabels.fgroup],
            mask.cf["Y"].name: mask.cf["Y"],
            mask.cf["X"].name: mask.cf["X"],
        },
        dims=(NoTransportLabels.fgroup, mask.cf["Y"].name, mask.cf["X"].name),
        data=masks,
        attrs={
            "long_name": "mask",
            "flag_values": [0, 1],
            "flag_meanings": {0: "land", 1: "ocean"},
        },
    )


def compute_daylength(time: xr.DataArray, latitude: xr.DataArray, longitude: xr.DataArray) -> xr.DataArray:
    """
    Use the grid and the time to generate the daylength.

    Should have a look to https://stackoverflow.com/questions/38986527/sunrise-and-sunset-time-in-python. Some libraries
    exist that do the job.

    Output
    ------
    - daylength [time, latitude, longitude]

    ---
    Information available in `xarray.cftime_range()` function.
    """
    return day_length.mesh_day_length(time, latitude, longitude, dask=True)


# TODO(Jules): Pourquoi le daylength est-il en coordonnée et non en variable ?
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
    average_temperature = []
    for fgroup in day_layer[NoTransportLabels.fgroup]:
        day_temperature = temperature.cf.sel(Z=day_layer.sel({NoTransportLabels.fgroup: fgroup}))
        night_temperature = temperature.cf.sel(Z=night_layer.sel({NoTransportLabels.fgroup: fgroup}))
        mean_temperature = ((daylength * day_temperature) + ((24 - daylength) * night_temperature)) / 24
        if "Z" in mean_temperature.cf:
            mean_temperature = mean_temperature.cf.drop_vars("Z")
        mean_temperature = mean_temperature.where(mask.sel({NoTransportLabels.fgroup: fgroup}))
        average_temperature.append(mean_temperature)
    return xr.concat(average_temperature, dim=NoTransportLabels.fgroup)


def apply_coefficient_to_primary_production(
    primary_production: xr.DataArray, functional_group_coefficient: xr.DataArray
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
    return xr.concat((i * primary_production for i in functional_group_coefficient), dim=NoTransportLabels.fgroup)


def min_temperature_by_cohort(mean_timestep: xr.DataArray, tr_max: xr.DataArray, tr_rate: xr.DataArray) -> xr.DataArray:
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
    return np.log(mean_timestep / tr_max) / tr_rate


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
    return average_temperature >= min_temperature_by_cohort


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
