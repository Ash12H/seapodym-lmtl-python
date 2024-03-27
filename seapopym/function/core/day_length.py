"""Core functions used to calculate the length of the day."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pint
import xarray as xr

from seapopym.standard.units import StandardUnitsLabels

DAY_IN_HOUR = pint.application_registry("day").to("hour").magnitude


def day_length_forsythe(latitude: float, day_of_the_year: int, p: int = 0) -> float:
    """
    Compute the day length for a given latitude, day of the year and twilight angle.

    NOTE Jules : Seapodym fish (CSimtunaFunc::daylength_twilight)
    The CBM model of Forsythe et al, Ecological Modelling 80 (1995) 87-95
    p - angle between the sun position and the horizon, in degrees :
        - 6  => civil twilight
        - 12 => nautical twilight
        - 18 => astronomical twilight

    Returns
    -------
    float
        The day length in HOURS.

    """
    # revolution angle for the day of the year
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (day_of_the_year - 186)))
    # sun's declination angle, or the angular distance at solar noon between the
    # Sun and the equator, from the Eartch orbit revolution angle
    phi = np.arcsin(0.39795 * np.cos(theta))
    # daylength computed according to 'p'
    arg = (np.sin(np.pi * p / 180) + np.sin(latitude * np.pi / 180) * np.sin(phi)) / (
        np.cos(latitude * np.pi / 180) * np.cos(phi)
    )

    arg = np.clip(arg, -1.0, 1.0)

    return DAY_IN_HOUR - (DAY_IN_HOUR / np.pi) * np.arccos(arg)


def mesh_day_length(
    time: xr.DataArray,
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    angle_horizon_sun: int = 0,
) -> xr.DataArray:
    """
    Compute the day length according to coordinates.

    Parameters
    ----------
    time : xr.DataArray
        The time. This coordinates must contains a the following attributes: {'axis':'T'}.
    latitude : xr.DataArray
        The latitude.
    longitude : xr.DataArray
        The longitude.
    angle_horizon_sun : int, optional
        The angle between the sun position and the horizon, in degrees. Default is 0.

    Returns
    -------
    xr.DataArray
        The day length that can be passed to a dataset as coordinates ((time, latitude, longitude), day_length).

    """
    if isinstance(time, xr.DataArray):
        try:
            time_index = time.indexes[time.cf["T"].name]
        except KeyError as e:
            error_message = "time must have a attrs={..., 'axis':'T', ...}"
            raise ValueError(error_message) from e

    if isinstance(time_index, xr.CFTimeIndex):
        day_of_year = time_index.dayofyear
    elif isinstance(time_index, pd.DatetimeIndex):
        day_of_year = time_index.day_of_year
    else:
        error_message = "time must be a xr.CFTimeIndex or a pd.DatetimeIndex"
        raise TypeError(error_message)

    cell_latitude = np.tile(latitude, (time_index.size, longitude.size, 1)).transpose(0, 2, 1)
    cell_time = np.tile(day_of_year, (latitude.size, longitude.size, 1)).transpose(2, 0, 1)
    data = day_length_forsythe(cell_latitude, cell_time, p=angle_horizon_sun)

    mesh_in_hour = xr.DataArray(
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        dims=["time", "latitude", "longitude"],
        data=data,
        name="day_length",
        attrs={
            "long_name": "Day length",
            "standard_name": "day_length",
            "description": f"Day length at the surface using Forsythe's method with p={angle_horizon_sun}",
            "units": "hour",
        },
    )

    return mesh_in_hour.pint.quantify().pint.to(StandardUnitsLabels.time.units).pint.dequantify()
