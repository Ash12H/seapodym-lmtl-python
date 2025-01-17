"""A temperature mask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import mortality_field_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.units import StandardUnitsLabels, check_units

if TYPE_CHECKING:
    import xarray as xr


def _mortality_field_helper(state: xr.Dataset) -> xr.DataArray:
    """
    Use the relation between temperature and mortality to generate the mortality field.

    Depend on
    ---------
    - average_temperature()

    Input
    ------
    - average_temperature [functional_group, time, latitude, longitude]
    - inv_lambda_max [functional_group]
    - inv_lambda_rate [functional_group]

    Output
    ------
    - mortality_field [functional_group, time, latitude, longitude]
    """
    average_temperature = state[ForcingLabels.avg_temperature_by_fgroup]
    inv_lambda_max = state[ConfigurationLabels.inv_lambda_max]
    inv_lambda_rate = state[ConfigurationLabels.inv_lambda_rate]
    timestep = state[ConfigurationLabels.timestep]

    average_temperature = check_units(average_temperature, StandardUnitsLabels.temperature)
    return np.exp(-timestep * np.exp(-inv_lambda_rate * average_temperature) / inv_lambda_max)


def mortality_field_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.mortality_field,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=mortality_field_desc,
        chunks=chunk,
    )


def mortality_field_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = mortality_field_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.mortality_field,
        template=template,
        function=_mortality_field_helper,
    )
