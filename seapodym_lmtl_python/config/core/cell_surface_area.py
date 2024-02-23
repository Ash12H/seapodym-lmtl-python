"""Module specific to the cell surface area computation."""

from __future__ import annotations

import numpy as np
import pint

EARTH_RADIUS = 6371 * pint.application_registry.kilometers


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
    latitude: float | pint.Quantity,
    resolution: float | pint.Quantity,
) -> pint.Quantity | np.ndarray:
    """
    Return the cell surface area according to its centroid latitude position and
    resolution (in degrees).
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
