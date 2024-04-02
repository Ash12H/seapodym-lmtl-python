"""A landmask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.day_length import mesh_day_length
from seapopym.function.core.template import Template, apply_map_block
from seapopym.standard.attributs import day_length_desc
from seapopym.standard.labels import CoordinatesLabels, PreproductionLabels


def day_length(state: xr.Dataset, chunk: dict | None = None, angle_horizon_sun: int = 0) -> xr.DataArray:
    """Wrap the day length computation with a map_block function."""

    def _wrapper_mesh_day_lengths(state: xr.DataArray) -> xr.DataArray:
        return mesh_day_length(
            state.cf[CoordinatesLabels.time],
            state.cf[CoordinatesLabels.Y],
            state.cf[CoordinatesLabels.X],
            angle_horizon_sun,
        )

    template = Template(
        name=PreproductionLabels.day_length,
        dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        attributs=day_length_desc(angle_horizon_sun=angle_horizon_sun),
        chunk=chunk,
    )

    return apply_map_block(function=_wrapper_mesh_day_lengths, state=state, template=template)
