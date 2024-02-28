"""This module contains the function used in the **dependent** process. They are run in sequence in timeseries order."""

from __future__ import annotations

from typing import Callable

import xarray as xr
from dask.distributed import Client


def recruitment(
    pre_production: xr.DataArray, recruitment_mask: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Define which part of the pre-production is recruited and return both recuitment and the remaining
    pre-production.

    Input
    -----
    - pre_production [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]
    - recruitment_mask [functional_group, time{select(timestep)}latitude, longitude, cohort_age] :
        coming from the `mask_temperature_by_cohort_by_functional_group()` function.

    Output
    ------
    - recruitment [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]

    """


def remove_recruited(
    pre_production: xr.DataArray, recruitment_mask: xr.DataArray
) -> xr.DataArray:
    """
    Remove the recruited part of the pre-production.

    Input
    -----
    - pre_production [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]
    - recruitment_mask [functional_group, time{select(timestep)}, latitude, longitude, cohort_age] :
        coming from the `mask_temperature_by_cohort_by_functional_group()` function.

    Output
    ------
    - remaining_pre_production [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]

    """


def next_preproduction(
    next_preproduction: xr.DataArray,
    current_unrecruited_preproduction: xr.DataArray,
    max_age: int,
) -> xr.DataArray:
    """
    Return the next pre-production dataarray.

    Input
    -----
    - next_preproduction [functional_group, time{select(next(timestep))}, latitude, longitude, cohort_age]
    - current_unrecruited_preproduction [functional_group, time{select(timestep)}, latitude, longitude, cohort_age]:
        coming from the `recruitment()` function as remaining_pre_production.

    Output
    ------
    - next_preproduction [functional_group, time{select(next(timestep))}, latitude, longitude, cohort_age]

    """


# --- Wrapper --- #


def process(configuration: xr.Dataset, kernel: None | list[Callable]) -> xr.Dataset:
    """
    Wraps all the production functions.

    Parameters
    ----------
    configuration : xr.Dataset
        The model configuration that contains both forcing and parameters.
    kernel : None | list[Callable]
        The list of production functions to use. If None, the default list is used.

    """
    if kernel is None:
        kernel = [recruitment, remove_recruited, next_preproduction]

    # Run the production process
    # ...   def loop_function():
    # ...       for timestep in dataset.time :
    # ...           Apply all functions to timestep
    # ...
    # ...   xarray.dataset.chunk()
    # ...   xarray.dataset.map_block(loop_function)

    return configuration
