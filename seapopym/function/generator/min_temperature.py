"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import min_temperature_by_cohort_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels


def _min_temperature_by_cohort_helper(state: xr.Dataset) -> xr.DataArray:
    """
    Define the minimal temperature of a cohort to be recruited.

    Input
    -----
    - mean_timestep [functional_group, cohort_age]
    - tr_max [functional_group]
    - tr_rate [functional_group]

    Output
    ------
    - min_temperature [functional_group, cohort_age] : a datarray with cohort_age as coordinate and
    minimum temperature as value.
    """
    return (
        np.log(state[ConfigurationLabels.mean_timestep] / state[ConfigurationLabels.temperature_recruitment_max])
        / state[ConfigurationLabels.temperature_recruitment_rate]
    )


def min_temperature_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.min_temperature,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.cohort],
        attrs=min_temperature_by_cohort_desc,
        chunks=chunk,
    )


def min_temperature_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = min_temperature_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.min_temperature,
        template=template,
        function=_min_temperature_by_cohort_helper,
    )
