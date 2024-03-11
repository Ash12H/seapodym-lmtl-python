"""This module contains the function that generate a landmask from any forcing data."""

import xarray as xr


def landmask_from_nan(forcing: xr.DataArray) -> xr.DataArray:
    """Create a landmask from a forcing data array."""
    mask = forcing.isel(time=0).notnull().reset_coords("time", drop=True)
    mask.name = "mask"
    mask.attrs = {
        "long_name": "mask",
        "flag_values": [0, 1],
        "flag_meanings": "0:land, 1:ocean",
    }
    return mask


# NOTE(Jules):  Other functions can be implemented here. For example, a function that creates a landmask from a user
#               text file.
