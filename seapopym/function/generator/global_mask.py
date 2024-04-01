"""A landmask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import xarray as xr

from seapopym.function.core.mask import landmask_from_nan
from seapopym.function.core.template import Template, apply_map_block
from seapopym.standard.attributs import global_mask_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels


def global_mask(state: xr.Dataset, chunk: dict | None = None) -> xr.DataArray:
    """Wrap the landmask computation with a map_block function."""

    def _global_mask_from_nan(state: xr.Dataset) -> xr.DataArray:
        return landmask_from_nan(state[ConfigurationLabels.temperature])

    template = Template(
        name=PreproductionLabels.global_mask,
        dims=[CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z],
        attributs=global_mask_desc,
        chunk=chunk,
    )

    return apply_map_block(function=_global_mask_from_nan, state=state, template=template)
