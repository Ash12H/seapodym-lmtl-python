"""A landmask computation wrapper. Use xarray.map_block."""

import xarray as xr

from seapopym.function.core.landmask import landmask_from_nan
from seapopym.function.core.template import generate_template
from seapopym.standard.attributs import global_mask_desc
from seapopym.standard.coordinates import reorder_dims
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels


def global_mask(state: xr.Dataset, chunk: dict) -> xr.DataArray:
    """Wrap the landmask computation with a map_block function."""
    max_dims = [CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z]
    template_mask = generate_template(state=state, dims=max_dims, attributs=global_mask_desc, chunk=chunk)
    data = reorder_dims(state[ConfigurationLabels.temperature])

    return xr.map_blocks(landmask_from_nan, data, template=template_mask)
