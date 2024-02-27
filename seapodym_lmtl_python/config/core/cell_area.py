"""Module specific to the cell surface area computation."""

from __future__ import annotations

import cf_xarray.units  # noqa: F401
import numpy as np
import pint
import pint_xarray  # noqa: F401
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates

EARTH_RADIUS = 6_371_000 * pint.application_registry.meters


def haversine_distance(
    min_latitude: pint.Quantity,
    max_latitude: pint.Quantity,
    min_longitude: pint.Quantity,
    max_longitude: pint.Quantity,
) -> pint.Quantity:
    """
    Calculate the great circle distance between two points on the earth (specified in
    decimal degrees).

    Wikipedia : https://en.wikipedia.org/wiki/Haversine_formula
    """
    ureg = pint.UnitRegistry()
    min_latitude = min_latitude.to(ureg.radians).magnitude
    max_latitude = max_latitude.to(ureg.radians).magnitude
    min_longitude = min_longitude.to(ureg.radians).magnitude
    max_longitude = max_longitude.to(ureg.radians).magnitude

    dlat = max_latitude - min_latitude
    dlon = max_longitude - min_longitude

    if (
        dlon >= (180 * ureg.degrees).to(ureg.radians).magnitude
        or dlon < (-180 * ureg.degrees).to(ureg.radians).magnitude
    ):
        dlon = (dlon * ureg.radians).to(ureg.degrees).magnitude
        error_msg = (
            f"The longitude distance ( = {dlon}) must be in [-180, 180[ interval. The result of the haversine formula "
            "is the shortest distance between the two points. "
        )
        raise ValueError(error_msg)

    # # Haversine formula
    hav_theta = (np.sin(dlat / 2) ** 2) + np.cos(min_latitude) * np.cos(
        max_latitude
    ) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(hav_theta))


def cell_borders_length(
    latitude: pint.Quantity,
    resolution: pint.Quantity,
) -> tuple[pint.Quantity, pint.Quantity]:
    """
    Calculate the edge length of a cell (in kilometers) using its centroid latitude
    position and its resolution (in degrees).
    """
    longitude = 0 * pint.application_registry.degrees

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
    latitude: float | pint.Quantity | np.ndarray,
    resolution: float | pint.Quantity | np.ndarray,
) -> pint.Quantity | np.ndarray:
    """
    Return the cell surface area (squared meters) according to its centroid latitude position and resolution (in
    degrees).
    """
    latitude = (
        latitude.to("degrees")
        if isinstance(latitude, pint.Quantity)
        else latitude * pint.application_registry.degrees
    )
    resolution = (
        resolution.to("degrees")
        if isinstance(resolution, pint.Quantity)
        else resolution * pint.application_registry.degrees
    )
    lat_len, lon_len = cell_borders_length(latitude, resolution)
    return lat_len * lon_len


def mesh_cell_area(
    latitude: np.ndarray,
    longitude: np.ndarray,
    resolution: float | pint.Quantity,
) -> xr.DataArray:
    """
    Expand the cell_area function to a meshgrid of latitude and longitude.

    Parameters
    ----------
    latitude : np.ndarray
        Latitude values (float values assume degrees).
    longitude : np.ndarray
        Longitude values (float values assume degrees).
    resolution : float | pint.Quantity
        Resolution of the grid (float values assume degrees).

    Returns
    -------
    xr.DataArray
        A DataArray containing the cell area for each grid cell. Quantified with pint as "squared meters".

    """
    cell_y = cell_area(latitude=latitude, resolution=resolution)
    mesh_cell_area = np.tile(cell_y, (len(longitude), 1)).T

    latitude = coordinates.new_latitude(latitude)
    longitude = coordinates.new_longitude(longitude)

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
    ).pint.quantify()
