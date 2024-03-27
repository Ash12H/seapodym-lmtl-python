"""A landmask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import xarray as xr

from seapopym.function.core.landmask import landmask_from_nan
from seapopym.function.core.template import apply_map_block, generate_template
from seapopym.standard.attributs import global_mask_desc
from seapopym.standard.coordinates import reorder_dims
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels


def global_mask(state: xr.Dataset, chunk: dict | None = None) -> xr.DataArray:
    """Wrap the landmask computation with a map_block function."""
    max_dims = [CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z]
    data = reorder_dims(state[ConfigurationLabels.temperature])
    return apply_map_block(
        function=landmask_from_nan,
        state=data,
        template=max_dims,
        attributs=global_mask_desc,
        chunk=chunk,
    )
