"""A landmask computation wrapper. Use xarray.map_block."""

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.day_length import mesh_day_length
from seapopym.function.core.template import generate_template
from seapopym.standard.attributs import day_length_desc
from seapopym.standard.labels import CoordinatesLabels


def day_length(state: xr.Dataset, chunk: dict, angle_horizon_sun: int = 0) -> xr.DataArray:
    """Wrap the day length computation with a map_block function."""

    def _wrapper_mesh_day_lengths(state: xr.DataArray) -> xr.DataArray:
        return mesh_day_length(
            state.cf[CoordinatesLabels.time],
            state.cf[CoordinatesLabels.Y],
            state.cf[CoordinatesLabels.X],
            angle_horizon_sun,
        )

    max_dims = [CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
    template_day_length = generate_template(
        state=state, dims=max_dims, attributs=day_length_desc(angle_horizon_sun), chunk=chunk
    )
    return xr.map_blocks(_wrapper_mesh_day_lengths, state, template=template_day_length)
