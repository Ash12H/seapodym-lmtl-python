"""A temperature mask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.template import apply_map_block, generate_template
from seapopym.standard.attributs import mask_temperature_desc
from seapopym.standard.labels import CoordinatesLabels, PreproductionLabels
from seapopym.standard.units import StandardUnitsLabels, check_units


def _mask_temperature_helper(state: xr.Dataset) -> xr.DataArray:
    """
    It uses the min_temperature.

    Depend on
    ---------
    - min_temperature()
    - average_temperature()

    Input
    -----
    - min_temperature [cohort_age]
    - average_temperature [functional_group, time, latitude, longitude]

    Output
    ------
    - mask_temperature_by_cohort_by_functional_group [functional_group, time, latitude, longitude, cohort_age]

    NOTE(Jules): Warning : average temperature by functional group (because of daily vertical migration) and not by
    layer. We therefore have a function with a high cost in terms of computation and memory space.

    """
    average_temperature = check_units(
        state[PreproductionLabels.avg_temperature_by_fgroup], StandardUnitsLabels.temperature.units
    )
    min_temperature = check_units(state[PreproductionLabels.min_temperature], StandardUnitsLabels.temperature.units)
    mask_temperature_by_fgroup = average_temperature >= min_temperature
    mask_temperature_by_fgroup.name = "mask_temperature_by_cohort_by_functional_group"
    return mask_temperature_by_fgroup


def mask_temperature(state: xr.Dataset, chunk: dict | None = None) -> xr.DataArray:
    """Wrap the average temperature by functional group computation with a map_block function."""
    max_dims = [
        CoordinatesLabels.functional_group,
        CoordinatesLabels.time,
        CoordinatesLabels.Y,
        CoordinatesLabels.X,
        CoordinatesLabels.cohort,
    ]
    return apply_map_block(
        function=_mask_temperature_helper,
        state=state,
        dims=max_dims,
        attributs=mask_temperature_desc,
        chunk=chunk,
    )
