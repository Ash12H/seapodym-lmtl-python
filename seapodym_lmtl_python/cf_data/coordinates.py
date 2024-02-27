"""Defines the structure of the coordinates of the data."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Iterable, Literal

import cf_xarray.units  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

if TYPE_CHECKING:
    import numpy as np


class SeaLayers(Enum):
    """Enumerate the sea layers."""

    # NOTE(Jules): The following order of the layers declaration is important.
    ## Since python 3.4 this order is preserved.
    EPI = ("epipelagic", 1)
    UPMESO = ("upper-mesopelagic", 2)
    LOWMESO = ("lower-mesopelagic", 3)

    @property
    def standard_name(
        self: SeaLayers,
    ) -> Literal["epipelagic", "upper-mesopelagic", "lower-mesopelagic"]:
        """Return the standard_name of the sea layer."""
        return self.value[0]

    @property
    def depth(self: SeaLayers) -> Literal[1, 2, 3]:
        """Return the depth of the sea layer."""
        return self.value[1]


def new_latitude(latitude_data: np.ndarray) -> xr.DataArray:
    """Create a new latitude coordinate."""
    return xr.DataArray(
        coords={"latitude": latitude_data},
        dims=["latitude"],
        data=latitude_data,
        attrs={
            "long_name": "latitude",
            "standard_name": "latitude",
            "units": "degrees_north",
            "axis": "Y",
        },
    )


def new_longitude(longitude_data: Iterable) -> xr.DataArray:
    """Create a new longitude coordinate."""
    return xr.DataArray(
        coords={"longitude": longitude_data},
        dims=["longitude"],
        data=longitude_data,
        attrs={
            "long_name": "longitude",
            "standard_name": "longitude",
            "units": "degrees_east",
            "axis": "X",
        },
    )


def new_layer(layer_data: Iterable | None = None) -> xr.DataArray:
    """Create a new layer coordinate."""
    if layer_data is None:
        layer_data = [layer.depth for layer in SeaLayers]
    return xr.DataArray(
        coords={"layer": layer_data},
        dims=["layer"],
        data=layer_data,
        attrs={
            "long_name": "layer",
            "standard_name": "layer",
            "positive": "down",
            "axis": "Z",
            "flag_values": layer_data,
            "flag_meanings": " ".join([layer.standard_name for layer in SeaLayers]),
        },
    )


def new_time(time_data: Iterable) -> xr.DataArray:
    """Create a new time coordinate."""
    return xr.DataArray(
        coords={"time": time_data},
        dims=["time"],
        data=time_data,
        attrs={"long_name": "time", "standard_name": "time", "axis": "T"},
    )
