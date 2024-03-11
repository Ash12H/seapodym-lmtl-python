"""
This module contains the post-production function used to compute the biomass.
They are run after the production process.
"""

import numpy as np
import xarray as xr
from numba import jit

from seapodym_lmtl_python.configuration.no_transport.labels import (
    ConfigurationLabels,
    PreproductionLabels,
    ProductionLabels,
)


@jit
def biomass_sequence(recruited: np.ndarray, mortality: np.ndarray) -> np.ndarray:
    """
    Compute the biomass of the recruited individuals.

    Formula :
        B(t) = R(t) + (M(t) * B(t-1))
    """
    biomass = np.zeros(recruited.shape)
    biomass[:, 0, ...] = recruited[:, 0, ...]
    # TODO(Jules) : Add initialisation from configuration
    for timestep in range(1, recruited.shape[1]):
        # TODO(Jules) : Check how to manage the ageing of aggregated cohorts
        # -> For now, we consider that the mortality is the same for all cohorts because mortality is implicite to
        # pre-production. We can use the sum of all cohorts to compute the production.
        biomass[:, timestep, ...] = recruited[:, timestep, ...] + (
            mortality[:, timestep, ...] * biomass[:, timestep - 1, ...]
        )
    return biomass


def compute_biomass(data: xr.Dataset) -> xr.DataArray:
    data = data.cf.transpose(ConfigurationLabels.fgroup, "T", "Y", "X", "Z", ConfigurationLabels.cohort)
    recruited = data[ProductionLabels.recruited].sum(ConfigurationLabels.cohort)
    recruited = np.nan_to_num(recruited.data, 0)
    mortality = np.nan_to_num(data[PreproductionLabels.mortality_field].data, 0)
    biomass = biomass_sequence(recruited, mortality)
    return xr.DataArray(
        biomass,
        coords=data[PreproductionLabels.mortality_field].coords,
        dims=data[PreproductionLabels.mortality_field].dims,
        attrs={
            # TODO(Jules)
        },
    )
