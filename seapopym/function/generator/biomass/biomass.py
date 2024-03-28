"""
This module contains the post-production function used to compute the biomass.
They are run after the production process.
"""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.template import apply_map_block
from seapopym.function.generator.biomass.compiled_functions import biomass_sequence
from seapopym.standard.attributs import biomass_desc
from seapopym.standard.labels import (
    ConfigurationLabels,
    CoordinatesLabels,
    PostproductionLabels,
    PreproductionLabels,
    ProductionLabels,
)


def _biomass_helper(state: xr.Dataset) -> xr.DataArray:
    """Wrap the biomass computation arround the Numba function `biomass_sequence`."""

    def _format_fields(state: xr.Dataset) -> xr.Dataset:
        """Format the fields to be used in the biomass computation."""
        return np.nan_to_num(state.data, 0.0).astype(np.float64)

    state = state.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")
    recruited = state[ProductionLabels.recruited].sum(CoordinatesLabels.cohort)
    recruited = _format_fields(recruited)
    mortality = _format_fields(state[PreproductionLabels.mortality_field])
    if ConfigurationLabels.initial_condition_biomass in state:
        initial_conditions = _format_fields(state[ConfigurationLabels.initial_condition_biomass])
    else:
        initial_conditions = None

    biomass = biomass_sequence(recruited=recruited, mortality=mortality, initial_conditions=initial_conditions)

    return xr.DataArray(
        biomass,
        coords=state[PreproductionLabels.mortality_field].coords,
        dims=state[PreproductionLabels.mortality_field].dims,
    )


def biomass(state: xr.Dataset, chunk: dict | None = None) -> xr.DataArray:
    """Wrap the biomass cumputation with a map_block function."""
    max_dims = [CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
    return apply_map_block(
        function=_biomass_helper,
        state=state,
        dims=max_dims,
        name=PostproductionLabels.biomass,
        attributs=biomass_desc,
        chunk=chunk,
    )
