"""Core functions to generate a landmask from any forcing data."""

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.standard.attributs import global_mask_desc


def landmask_from_nan(forcing: xr.DataArray) -> xr.DataArray:
    """Create a landmask from a forcing data array."""
    mask = forcing.cf.isel(T=0).notnull().cf.reset_coords("T", drop=True)
    mask.name = "mask"
    mask.attrs = global_mask_desc
    return mask


# NOTE(Jules):  Other functions can be implemented here. For example, a function that creates a landmask from a user
#               text file.
