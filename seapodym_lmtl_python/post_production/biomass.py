"""
This module contains the post-production function used to compute the biomass.
They are run after the production process.
"""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from numba import jit

from seapodym_lmtl_python.configuration.no_transport.labels import (
    ConfigurationLabels,
    PostproductionLabels,
    PreproductionLabels,
    ProductionLabels,
    StandardUnitsLabels,
)


@jit()
def biomass_sequence(recruited: np.ndarray, mortality: np.ndarray, initial_conditions: np.ndarray | None) -> np.ndarray:
    """
    Compute the biomass of the recruited individuals.

    Formula :
        B(t) = R(t) + (M(t) * B(t-1))

    Parameters
    ----------
    recruited : np.ndarray
        The number of recruited individuals. Minimum dims : (F, T, <Y, X, ...>). Type : np.float64
    mortality : np.ndarray
        The mortality rate of the recruited individuals. Minimum dims : (F, T, <Y, X, ...>). Type : np.float64
    initial_conditions : np.ndarray
        The initial biomass. Minimum dims : (F, <Y, X, ...>). Type : np.float64
        If None, the initial biomass is set to `np.zeros(recruited[:, 0, ...].shape)`.

    Warning
    -------
    - Be sure to use floats for the input arrays.

    """
    initial_conditions = (
        np.zeros(recruited[:, 0, ...].shape, dtype=np.float64) if initial_conditions is None else initial_conditions
    )
    biomass = np.zeros(recruited.shape)
    biomass[:, 0, ...] = recruited[:, 0, ...] + (mortality[:, 0, ...] * initial_conditions)

    for timestep in range(1, recruited.shape[1]):
        # -> For now, we consider that the mortality is the same for all cohorts because mortality is implicite to
        # pre-production. We can use the sum of all cohorts to compute the production.
        biomass[:, timestep, ...] = recruited[:, timestep, ...] + (
            mortality[:, timestep, ...] * biomass[:, timestep - 1, ...]
        )
    return biomass


def compute_biomass(data: xr.Dataset) -> xr.DataArray:
    """Wrap the biomass computation arround the Numba function `biomass_sequence`."""

    def _format_fields(data: xr.Dataset) -> xr.Dataset:
        """Format the fields to be used in the biomass computation"""
        return np.nan_to_num(data.data, 0.0).astype(np.float64)

    data = data.cf.transpose(ConfigurationLabels.fgroup, "T", "Y", "X", "Z", ConfigurationLabels.cohort)
    recruited = data[ProductionLabels.recruited].sum(ConfigurationLabels.cohort)
    recruited = _format_fields(recruited)
    mortality = _format_fields(data[PreproductionLabels.mortality_field])
    if ConfigurationLabels.initial_condition_biomass in data:
        initial_conditions = _format_fields(data[ConfigurationLabels.initial_condition_biomass])
    else:
        initial_conditions = None

    biomass = biomass_sequence(
        recruited=recruited.astype(np.float64),
        mortality=mortality.astype(np.float64),
        initial_conditions=initial_conditions,
    )

    return xr.DataArray(
        biomass,
        coords=data[PreproductionLabels.mortality_field].coords,
        dims=data[PreproductionLabels.mortality_field].dims,
        attrs={
            "long_name": "biomass",
            "units": str(StandardUnitsLabels.biomass.units),
            "description": "The biomass of the recruited individuals.",
        },
        name=PostproductionLabels.biomass,
    )
