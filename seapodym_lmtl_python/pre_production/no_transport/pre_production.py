"""Wrap the pre_production computation to follow the no-transport model."""

import xarray as xr

from seapodym_lmtl_python.pre_production import pre_production
from seapodym_lmtl_python.pre_production.core.landmask import landmask_from_nan
from seapodym_lmtl_python.standard.coordinates import list_available_dims
from seapodym_lmtl_python.standard.labels import ConfigurationLabels, PreproductionLabels


def mask_global(state: xr.Dataset):
    def _wrap_landmask_from_nan(state: xr.Dataset):
        return landmask_from_nan(state[ConfigurationLabels.temperature])

    coords = state[ConfigurationLabels.temperature].cf.isel(T=0).cf.reset_coords("T", drop=True).coords
    dims = coords.keys()

    template = xr.DataArray(dims=list(dims), coords=coords).chunk()
    return xr.map_blocks(_wrap_landmask_from_nan, state, template=template)
