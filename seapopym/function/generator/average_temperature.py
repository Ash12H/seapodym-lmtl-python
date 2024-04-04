"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.template import Template, TemplateLazy, apply_map_block
from seapopym.standard.attributs import average_temperature_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels
from seapopym.standard.types import ForcingName, SeapopymForcing
from seapopym.standard.units import StandardUnitsLabels, check_units


def _average_temperature_by_fgroup_helper(state: xr.Dataset) -> xr.DataArray:
    """
    Depend on:
    - compute_daylength
    - mask_by_fgroup.

    Input
    -----
    - mask_by_fgroup()      [time, latitude, longitude]
    - compute_daylength()   [functional_group, latitude, longitude] in day
    - day/night_layer       [functional_group]
    - temperature           [time, latitude, longitude, layer] in degC

    Output
    ------
    - avg_temperature [functional_group, time, latitude, longitude] in degC
    """
    temperature = check_units(state[ConfigurationLabels.temperature], StandardUnitsLabels.temperature.units)
    day_length = check_units(state[PreproductionLabels.day_length], StandardUnitsLabels.time.units)
    mask_by_fgroup = state[PreproductionLabels.mask_by_fgroup]
    day_layer = state[ConfigurationLabels.day_layer]
    night_layer = state[ConfigurationLabels.night_layer]

    average_temperature = []
    for fgroup in day_layer[CoordinatesLabels.functional_group]:
        day_temperature = temperature.cf.sel(Z=day_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        night_temperature = temperature.cf.sel(Z=night_layer.sel({CoordinatesLabels.functional_group: fgroup}))
        mean_temperature = (day_length * day_temperature) + ((1 - day_length) * night_temperature)
        if "Z" in mean_temperature.cf:
            mean_temperature = mean_temperature.cf.drop_vars("Z")
        mean_temperature = mean_temperature.where(mask_by_fgroup.sel({CoordinatesLabels.functional_group: fgroup}))
        average_temperature.append(mean_temperature)

    return xr.concat(average_temperature, dim=CoordinatesLabels.functional_group.value)


def average_temperature(
    state: xr.Dataset, chunk: dict | None = None, lazy: ForcingName | None = None
) -> SeapopymForcing:
    """Wrap the average temperature by functional group computation with a map_block function."""
    class_type = Template if lazy is None else TemplateLazy
    template_attributs = {
        "name": PreproductionLabels.avg_temperature_by_fgroup,
        "dims": [CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        "attributs": average_temperature_by_fgroup_desc,
        "chunk": chunk,
    }
    if lazy is not None:
        template_attributs["model_name"] = lazy
    template = class_type(**template_attributs)

    return apply_map_block(function=_average_temperature_by_fgroup_helper, state=state, template=template)
