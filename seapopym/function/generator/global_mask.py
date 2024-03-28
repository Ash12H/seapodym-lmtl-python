"""A landmask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import xarray as xr

from seapopym.function.core.landmask import landmask_from_nan
from seapopym.function.core.template import apply_map_block
from seapopym.standard.attributs import global_mask_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels


def global_mask(state: xr.Dataset, chunk: dict | None = None) -> xr.DataArray:
    """Wrap the landmask computation with a map_block function."""

    def _global_mask_from_nan(state: xr.Dataset) -> xr.DataArray:
        return landmask_from_nan(state[ConfigurationLabels.temperature])

    max_dims = [CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z]
    return apply_map_block(
        function=_global_mask_from_nan,
        state=state,
        name=PreproductionLabels.global_mask,
        dims=max_dims,
        attributs=global_mask_desc,
        chunk=chunk,
    )
