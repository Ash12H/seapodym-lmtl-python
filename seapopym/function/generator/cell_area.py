"""Module specific to the cell surface area computation."""

from __future__ import annotations

from typing import Iterable

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import compute_cell_area_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels

EARTH_RADIUS = 6_371_000 * StandardUnitsLabels.height.units


def haversine_distance(min_latitude: float, max_latitude: float, min_longitude: float, max_longitude: float) -> float:
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
    hav_theta = (np.sin(dlat / 2) ** 2) + np.cos(min_latitude) * np.cos(max_latitude) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS.magnitude * np.arcsin(np.sqrt(hav_theta))  # meters


def cell_borders_length(latitude: float, resolution: float | tuple[float, float]) -> tuple[float, float]:
    """
    Calculate the edge length of a cell (in kilometers) using its centroid latitude
    position and its resolution (in degrees).

    Parameters
    ----------
    latitude : float
        Latitude of the cell centroid.
    resolution : float or tuple
        Resolution of the grid in degrees. If a float, the resolution is assumed to be the same for latitude and
        longitude. If a tuple, the first value is the resolution for latitude and the second value is the resolution for
        longitude.

    """
    if isinstance(resolution, Iterable):
        res_lat = resolution[0]
        res_lon = resolution[1]
    else:
        res_lat = res_lon = resolution
    longitude = 0

    lat_len = haversine_distance(
        min_latitude=(latitude - (res_lat / 2)),
        max_latitude=(latitude + (res_lat / 2)),
        min_longitude=longitude,
        max_longitude=longitude,
    )
    lon_len = haversine_distance(
        min_latitude=latitude,
        max_latitude=latitude,
        min_longitude=longitude,
        max_longitude=res_lon,
    )
    return lat_len, lon_len


def cell_area(latitude: float, resolution: float | tuple[float, float]) -> float:
    """
    Return the cell surface area (squared meters) according to its centroid latitude position and resolution (in
    degrees).

    Parameters
    ----------
    latitude : float
        Latitude of the cell centroid.
    resolution : float or tuple
        Resolution of the grid in degrees. If a float, the resolution is assumed to be the same for latitude and
        longitude. If a tuple, the first value is the resolution for latitude and the second value is the resolution for
        longitude.

    """
    lat_len, lon_len = cell_borders_length(latitude, resolution)
    return lat_len * lon_len


# NOTE(Jules):  In the futur, this function can be used with different cell_area methods. In that case, we should
#               provide a generic `mesh_cell_area` function that will call the right method according to the method
#               provided in parameter. (dependency injection)
def mesh_cell_area(
    latitude: xr.DataArray, longitude: xr.DataArray, resolution: float | tuple[float, float]
) -> xr.DataArray:
    """
    Expand the cell_area function to a meshgrid of latitude and longitude.

    Parameters
    ----------
    latitude : np.ndarray
        Latitude values (float values assume degrees).
    longitude : np.ndarray
        Longitude values (float values assume degrees).
    resolution : float or tuple
        Resolution of the grid in degrees. If a float, the resolution is assumed to be the same for latitude and
        longitude. If a tuple, the first value is the resolution for latitude and the second value is the resolution for
        longitude.

    Returns
    -------
    xr.DataArray
        A DataArray containing the cell area for each grid cell.

    """
    cell_y = cell_area(latitude=latitude.data, resolution=resolution)
    mesh_cell_area = np.tile(cell_y, (int(longitude.size), 1)).T
    return xr.DataArray(
        coords={
            "latitude": latitude,
            "longitude": longitude,
        },
        dims=["latitude", "longitude"],
        attrs={
            "long_name": "area of grid cell",
            "standard_name": "cell_area",
            "units": str(StandardUnitsLabels.height.units**2),
        },
        data=mesh_cell_area,
    )


def _cell_area_helper(state: xr.Dataset) -> xr.DataArray:
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
    resolution = (state[ConfigurationLabels.resolution_latitude], state[ConfigurationLabels.resolution_longitude])
    resolution = np.asarray(resolution)
    resolution = float(resolution) if resolution.size == 1 else tuple(resolution)
    return mesh_cell_area(state.cf[CoordinatesLabels.Y], state.cf[CoordinatesLabels.X], resolution)


def cell_area_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.cell_area,
        dims=[CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=compute_cell_area_desc,
        chunks=chunk,
    )


def cell_area_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = cell_area_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.cell_area,
        template=template,
        function=_cell_area_helper,
    )
