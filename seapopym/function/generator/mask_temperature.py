"""A temperature mask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import mask_temperature_desc
from seapopym.standard.labels import CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels, check_units

if TYPE_CHECKING:
    import xarray as xr


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
        state[ForcingLabels.avg_temperature_by_fgroup], StandardUnitsLabels.temperature.units
    )
    min_temperature = check_units(state[ForcingLabels.min_temperature], StandardUnitsLabels.temperature.units)
    return average_temperature >= min_temperature


def mask_temperature_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.mask_temperature,
        dims=[
            CoordinatesLabels.functional_group,
            CoordinatesLabels.time,
            CoordinatesLabels.Y,
            CoordinatesLabels.X,
            CoordinatesLabels.cohort,
        ],
        attrs=mask_temperature_desc,
        chunks=chunk,
    )


def mask_temperature_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = mask_temperature_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.mask_temperature,
        template=template,
        function=_mask_temperature_helper,
    )
