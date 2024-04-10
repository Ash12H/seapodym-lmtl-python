"""
This module contains the post-production function used to compute the biomass.
They are run after the production process.
"""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.function.generator.biomass.compiled_functions import biomass_sequence
from seapopym.standard.attributs import biomass_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from seapopym.standard.types import SeapopymForcing


def _biomass_helper(state: xr.Dataset) -> xr.DataArray:
    """Wrap the biomass computation arround the Numba function `biomass_sequence`."""

    def _format_fields(forcing: SeapopymForcing) -> SeapopymForcing:
        """Format the fields to be used in the biomass computation."""
        return np.nan_to_num(forcing.data, 0.0).astype(np.float64)

    state = CoordinatesLabels.order_data(state)
    recruited = _format_fields(state[ForcingLabels.recruited])
    mortality = _format_fields(state[ForcingLabels.mortality_field])
    if ConfigurationLabels.initial_condition_biomass in state:
        initial_conditions = _format_fields(state[ConfigurationLabels.initial_condition_biomass])
    else:
        initial_conditions = None
    biomass = biomass_sequence(recruited=recruited, mortality=mortality, initial_conditions=initial_conditions)
    return xr.DataArray(
        dims=state[ForcingLabels.mortality_field].dims,
        coords=state[ForcingLabels.mortality_field].coords,
        data=biomass,
    )


def biomass_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.biomass,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=biomass_desc,
        chunks=chunk,
    )


def biomass_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = biomass_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.biomass,
        template=template,
        function=_biomass_helper,
    )
