"""Defines the structure of the coordinates of the data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import cf_xarray.units  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

from seapopym.standard.labels import CoordinatesLabels, SeaLayers

if TYPE_CHECKING:
    import numpy as np


def list_available_dims(data: xr.Dataset | xr.DataArray) -> list[str]:
    """Return the standard name of all available coordinates in the data."""
    return [coord for coord in CoordinatesLabels.ordered() if coord in data.cf]


def reorder_dims(data: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Follow the standard order of dimensions for a xarray.Dataset or xarray.DataArray."""
    return data.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")


def new_latitude(latitude_data: np.ndarray) -> xr.DataArray:
    """Create a new latitude coordinate."""
    attributs = {"long_name": "latitude", "standard_name": "latitude", "units": "degrees_north", "axis": "Y"}
    latitude = xr.DataArray(
        coords=[("latitude", latitude_data, attributs)],
        dims=["latitude"],
    )
    return latitude.cf["Y"]


def new_longitude(longitude_data: Iterable) -> xr.DataArray:
    """Create a new longitude coordinate."""
    attributs = {"long_name": "longitude", "standard_name": "longitude", "units": "degrees_east", "axis": "X"}
    longitude = xr.DataArray(
        coords=[("longitude", longitude_data, attributs)],
        dims=["longitude"],
    )
    return longitude.cf["X"]


def new_layer(layer_data: Iterable | None = None) -> xr.DataArray:
    """Create a new layer coordinate."""
    if layer_data is None:
        layer_data = [layer.depth for layer in SeaLayers]
    attributs = {
        "long_name": "layer",
        "standard_name": "layer",
        "positive": "down",
        "axis": "Z",
        "flag_values": layer_data,
        "flag_meanings": " ".join([layer.standard_name for layer in SeaLayers]),
    }
    layer = xr.DataArray(coords=(("layer", layer_data, attributs),), dims=["layer"])
    return layer.cf["Z"]


def new_time(time_data: Iterable) -> xr.DataArray:
    """Create a new time coordinate."""
    time = xr.DataArray(
        coords=[("time", time_data, {"long_name": "time", "standard_name": "time", "axis": "T"})], dims=["time"]
    )
    return time.cf["T"]


def new_cohort(cohort_data: Iterable) -> xr.DataArray:
    """Create a new cohort coordinate."""
    attributs = {"long_name": "cohort", "standard_name": "cohort"}
    cohort = xr.DataArray(
        coords=[("cohort", cohort_data, attributs)],
        dims=["cohort"],
    )
    return cohort.cf["cohort"]
