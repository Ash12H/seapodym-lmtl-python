"""Module specific to the cell surface area computation."""

from __future__ import annotations

import cf_xarray.units  # noqa: F401
import numpy as np
import pint
import pint_xarray  # noqa: F401
import xarray as xr

EARTH_RADIUS = 6_371_000 * pint.application_registry.meters


def haversine_distance(
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
) -> float:
    """
    Calculate the great circle distance between two points on the earth (specified in
    decimal degrees).

    Warning:
    -------
    If the longitude distance is greater than 180 degrees, the function will return the shortest distance between the
    two points.

    Wikipedia : https://en.wikipedia.org/wiki/Haversine_formula

    """
    min_latitude = np.deg2rad(min_latitude)
    max_latitude = np.deg2rad(max_latitude)
    min_longitude = np.deg2rad(min_longitude)
    max_longitude = np.deg2rad(max_longitude)

    dlat = max_latitude - min_latitude
    dlon = max_longitude - min_longitude

    # # Haversine formula
    hav_theta = (np.sin(dlat / 2) ** 2) + np.cos(min_latitude) * np.cos(
        max_latitude
    ) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(hav_theta))


def cell_borders_length(
    latitude: float,
    resolution: float,
) -> tuple[float, float]:
    """
    Calculate the edge length of a cell (in kilometers) using its centroid latitude
    position and its resolution (in degrees).
    """
    longitude = 0

    lat_len = haversine_distance(
        min_latitude=(latitude - (resolution / 2)),
        max_latitude=(latitude + (resolution / 2)),
        min_longitude=longitude,
        max_longitude=longitude,
    )
    lon_len = haversine_distance(
        min_latitude=latitude,
        max_latitude=latitude,
        min_longitude=longitude,
        max_longitude=resolution,
    )
    return lat_len, lon_len


def cell_area(
    latitude: float,
    resolution: float,
) -> float:
    """
    Return the cell surface area (squared meters) according to its centroid latitude position and resolution (in
    degrees).
    """
    lat_len, lon_len = cell_borders_length(latitude, resolution)
    return lat_len * lon_len


def mesh_cell_area(
    latitude: xr.DataArray,
    longitude: xr.DataArray,
    resolution: float,
) -> xr.DataArray:
    """
    Expand the cell_area function to a meshgrid of latitude and longitude.

    Parameters
    ----------
    latitude : np.ndarray
        Latitude values (float values assume degrees).
    longitude : np.ndarray
        Longitude values (float values assume degrees).
    resolution : float
        Resolution of the grid (float values assume degrees).

    Returns
    -------
    xr.DataArray
        A DataArray containing the cell area for each grid cell.

    """
    cell_y = cell_area(latitude=latitude, resolution=resolution)
    mesh_cell_area = np.tile(cell_y, (len(longitude), 1)).T

    return xr.DataArray(
        coords={
            "latitude": latitude,
            "longitude": longitude,
            "cell_area": (("latitude", "longitude"), mesh_cell_area),
        },
        dims=["latitude", "longitude"],
        attrs={
            "units": "m**2",
            "long_name": "area of grid cell",
            "standard_name": "cell_area",
        },
        data=mesh_cell_area,
    )
