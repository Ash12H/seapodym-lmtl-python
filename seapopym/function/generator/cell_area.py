"""A temperature mask computation wrapper. Use xarray.map_block."""

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.cell_area import mesh_cell_area
from seapopym.function.core.template import generate_template
from seapopym.standard.attributs import compute_cell_area_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels


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
    cell_surface_area = mesh_cell_area(state.cf[CoordinatesLabels.Y], state.cf[CoordinatesLabels.X], resolution)
    cell_surface_area.name = "cell_area"
    return cell_surface_area


def cell_area(state: xr.Dataset, chunk: dict) -> xr.DataArray:
    """Wrap the average temperature by functional group computation with a map_block function."""
    max_dims = [CoordinatesLabels.Y, CoordinatesLabels.X]
    template_mask_temperature = generate_template(
        state=state, dims=max_dims, attributs=compute_cell_area_desc, chunk=chunk
    )
    return xr.map_blocks(_cell_area_helper, state, template=template_mask_temperature)
