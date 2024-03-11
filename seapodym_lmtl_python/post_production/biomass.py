"""
This module contains the post-production function used to compute the biomass.
They are run after the production process.
"""

from typing import Callable
import xarray as xr
from dask.distributed import Client


def sum_recruitment(recruited: xr.DataArray) -> xr.DataArray:
    """
    Sum the recruitment over the functional_group axis.

    Input
    -----
    - recruited [functional_group, time, latitude, longitude, cohort_age]

    Output
    ------
    - recruitment [functional_group, time, latitude, longitude]

    """


def compute_biomass(
    recruitment: xr.DataArray,
    cell_area: xr.DataArray,
    mortality: xr.DataArray,
) -> xr.DataArray:
    """
    Compute the recruited biomass.

    Input
    -----
    - recruitment [functional_group, time, latitude, longitude] from the `sum_recruitment()` function.
    - cell_area [latitude, longitude]
    - mortality [functional_group, time, latitude, longitude] from the `compute_mortality_field()` function.

    Output
    ------
    - biomass [functional_group, time{select(timestep)}, latitude, longitude]
    """
