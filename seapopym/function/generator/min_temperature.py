"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import min_temperature_by_cohort_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels
from seapopym.standard.types import SeapopymForcing


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


# def min_temperature(state: xr.Dataset, chunk: dict | None = None, lazy: ForcingName | None = None) -> SeapopymForcing:
#     """Wrap the average temperature by functional group computation with a map_block function."""
#     class_type = Template if lazy is None else TemplateLazy
#     template_attributs = {
#         "name": PreproductionLabels.min_temperature,
#         "dims": [CoordinatesLabels.functional_group, CoordinatesLabels.cohort],
#         "attributs": min_temperature_by_cohort_desc,
#         "chunk": chunk,
#     }
#     if lazy is not None:
#         template_attributs["model_name"] = lazy
#     template = class_type(**template_attributs)

#     return apply_map_block(function=_min_temperature_by_cohort_helper, state=state, template=template)


def min_temperature_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=PreproductionLabels.min_temperature,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.cohort],
        attrs=min_temperature_by_cohort_desc,
        chunks=chunk,
    )


def min_temperature_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> SeapopymForcing:
    if template is None:
        template = min_temperature_template(chunk=chunk)
    return KernelUnits(
        name=PreproductionLabels.min_temperature,
        template=template,
        function=_min_temperature_by_cohort_helper,
    )
