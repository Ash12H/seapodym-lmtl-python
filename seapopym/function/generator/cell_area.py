"""A temperature mask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.cell_area import mesh_cell_area
from seapopym.function.core.template import apply_map_block
from seapopym.standard.attributs import compute_cell_area_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels


def _cell_area_helper(state: xr.Dataset) -> xr.DataArray:
    """
    Compute the cell area from the latitude and longitude.

    Input
    ------
    - latitude [latitude]
    - longitude [longitude]
    - resolution

    Output
    ------
    - cell_area [latitude, longitude]s
    """
    resolution = (state[ConfigurationLabels.resolution_latitude], state[ConfigurationLabels.resolution_longitude])
    resolution = np.asarray(resolution)
    resolution = float(resolution) if resolution.size == 1 else tuple(resolution)
    return mesh_cell_area(state.cf[CoordinatesLabels.Y], state.cf[CoordinatesLabels.X], resolution)


def cell_area(state: xr.Dataset, chunk: dict | None = None) -> xr.DataArray:
    """Wrap the average temperature by functional group computation with a map_block function."""
    max_dims = [CoordinatesLabels.Y, CoordinatesLabels.X]
    return apply_map_block(
        function=_cell_area_helper,
        state=state,
        name=PreproductionLabels.cell_area,
        dims=max_dims,
        attributs=compute_cell_area_desc,
        chunk=chunk,
    )
