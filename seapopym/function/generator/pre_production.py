"""All the functions used to generate or modify the forcings."""
from __future__ import annotations

import numpy as np
import xarray as xr

from seapopym.function.core import cell_area, day_length
from seapopym.standard.labels import CoordinatesLabels
from seapopym.standard.units import StandardUnitsLabels, check_units


def mask_by_fgroup(day_layers: xr.DataArray, night_layers: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """
    masks = []
    for i in day_layers[CoordinatesLabels.functional_group]:
        day_pos = day_layers.sel(functional_group=i)
        night_pos = night_layers.sel(functional_group=i)

        day_mask = mask.cf.sel(Z=day_pos)
        night_mask = mask.cf.sel(Z=night_pos)
        masks.append(day_mask & night_mask)

    return xr.DataArray(
        coords={
            CoordinatesLabels.functional_group: day_layers[CoordinatesLabels.functional_group],
            mask.cf["Y"].name: mask.cf["Y"],
            mask.cf["X"].name: mask.cf["X"],
        },
        dims=(CoordinatesLabels.functional_group, mask.cf["Y"].name, mask.cf["X"].name),
        data=masks,
        # TODO(Jules): Inherite from mask ?
        attrs={
            "standard_name": "mask",
            "long_name": "mask",
            "flag_values": [0, 1],
            "flag_meanings": "0:land, 1:ocean",
        },
        name="mask_by_fgroup",
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
    return day_length.mesh_day_length(time, latitude, longitude)


def average_temperature(
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
    - compute_daylength()   [functional_group, latitude, longitude] in day
    - day/night_layer       [functional_group]
    - temperature           [time, latitude, longitude, layer] in degC

    Output
    ------
    - avg_temperature [functional_group, time, latitude, longitude] in degC
    """
    temperature = check_units(temperature, StandardUnitsLabels.temperature.units)
    daylength = check_units(daylength, StandardUnitsLabels.time.units)

    average_temperature = []
    for fgroup in day_layer[CoordinatesLabels.functional_group]:
        day_temperature = temperature.cf.sel(Z=day_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        night_temperature = temperature.cf.sel(Z=night_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        mean_temperature = (daylength * day_temperature) + ((1 - daylength) * night_temperature)
        if "Z" in mean_temperature.cf:
            mean_temperature = mean_temperature.cf.drop_vars("Z")
        mean_temperature = mean_temperature.where(mask.sel({CoordinatesLabels.functional_group: fgroup}))
        average_temperature.append(mean_temperature)

    average_temperature = xr.concat(average_temperature, dim=CoordinatesLabels.functional_group, combine_attrs="drop")
    average_temperature.name = "average_temperature"
    return average_temperature.assign_attrs(
        {
            "long_name": "average sea temperature by fonctional group",
            "standard_name": "sea water temperature",
            "description": (
                "Average temperature by functional group according to their layer position during day and night."
            ),
            "units": str(StandardUnitsLabels.temperature.units),
        }
    )


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
    primary_production = check_units(primary_production, StandardUnitsLabels.production.units)
    pp_by_fgroup_gen = (i * primary_production for i in functional_group_coefficient)
    pp_by_fgroup = xr.concat(pp_by_fgroup_gen, dim=CoordinatesLabels.functional_group, combine_attrs="drop")
    pp_by_fgroup.name = "primary_production_by_fgroup"
    return pp_by_fgroup.assign_attrs(
        {
            "standard_name": "primary production",
            "long_name": "primary production by functional group",
            "description": "Primary production by functional group according to their energy transfert coefficient.",
            "units": str(StandardUnitsLabels.production.units),
        }
    )


def min_temperature(mean_timestep: xr.DataArray, tr_max: xr.DataArray, tr_rate: xr.DataArray) -> xr.DataArray:
    """
    Define the minimal temperature of a cohort to be recruited.

    Input
    -----
    - mean_timestep [functional_group, cohort_age]
    - tr_max [functional_group]
    - tr_rate [functional_group]

    Output
    ------
    - min_temperature [functional_group, cohort_age] : a datarray with cohort_age as coordinate and
    minimum temperature as value.
    """
    result = np.log(mean_timestep / tr_max) / tr_rate
    result.name = "min_temperature"
    return result.assign_attrs(
        {
            "standard_name": "minimum temperature",
            "long_name": "minimum temperature by cohort",
            "description": "Minimum temperature to recruit a cohort according to its age.",
            "units": str(StandardUnitsLabels.temperature.units),
        }
    )


def mask_temperature_by_cohort_by_functional_group(
    min_temperature: xr.DataArray, average_temperature: xr.DataArray
) -> xr.DataArray:
    """
    It uses the min_temperature.

    Depend on
    ---------
    - min_temperature()
    - average_temperature()

    Input
    -----
    - min_temperature [cohort_age]
    - average_temperature [functional_group, time, latitude, longitude]

    Output
    ------
    - mask_temperature_by_cohort_by_functional_group [functional_group, time, latitude, longitude, cohort_age]

    NOTE(Jules): Warning : average temperature by functional group (because of daily vertical migration) and not by
    layer. We therefore have a function with a high cost in terms of computation and memory space.

    """
    average_temperature = check_units(average_temperature, StandardUnitsLabels.temperature.units)
    min_temperature = check_units(min_temperature, StandardUnitsLabels.temperature.units)
    mask_temperature_by_fgroup = average_temperature >= min_temperature
    mask_temperature_by_fgroup.name = "mask_temperature_by_cohort_by_functional_group"
    return mask_temperature_by_fgroup.assign_attrs(
        {
            "standard_name": "mask",
            "long_name": "cohort recruitment mask by functional group",
            "description": "Mask to recruit a cohort according to the temperature.",
            "flag_values": [0, 1],
            "flag_meanings": "0:not recruited, 1:recruited",
        }
    )


def compute_cell_area(
    latitude: xr.DataArray, longitude: xr.DataArray, resolution: float | tuple[float, float]
) -> xr.DataArray:
    """
    Compute the cell area from the latitude and longitude.

    Input
    ------
    - latitude [latitude]
    - longitude [longitude]
    - resolution

    Output
    ------
    - cell_area [latitude, longitude]s
    """
    resolution = np.asarray(resolution)
    resolution = float(resolution) if resolution.size == 1 else tuple(resolution)
    cell_surface_area = cell_area.mesh_cell_area(latitude, longitude, resolution)
    cell_surface_area.name = "cell_area"
    return cell_surface_area.assign_attrs(
        {
            "standard_name": "cell_area",
            "long_name": "cell area",
            "description": "Cell area computed from the latitude and longitude centroid.",
            "units": str(StandardUnitsLabels.height.units**2),
        }
    )


def compute_mortality_field(
    average_temperature: xr.DataArray, inv_lambda_max: xr.DataArray, inv_lambda_rate: xr.DataArray, timestep: float
) -> xr.DataArray:
    """
    Use the relation between temperature and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature()

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - inv_lambda_max [functional_group]
    - inv_lambda_rate [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]
    """
    average_temperature = check_units(average_temperature, StandardUnitsLabels.temperature)
    mortality_field = np.exp(-timestep * (np.exp(inv_lambda_rate * average_temperature) / inv_lambda_max))
    mortality_field.name = "mortality_field"
    return mortality_field.assign_attrs(
        {
            "standard_name": "mortality",
            "long_name": "mortality coefficient",
            "description": "Mortality coefficient according to the temperature.",
        }
    )
