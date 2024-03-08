"""This module contains the function used in the **dependent** process. They are run in sequence in timeseries order."""

from __future__ import annotations

from typing import Callable

import xarray as xr
from dask.distributed import Client


def recruitment(pre_production: xr.DataArray, recruitment_mask: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
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


def remove_recruited(pre_production: xr.DataArray, recruitment_mask: xr.DataArray) -> xr.DataArray:
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


# NOTE(Jules): Draft


@jit
def expand_dims(data: np.ndarray, dim_len: int) -> np.ndarray:
    """Add a new dimension to the DataArray and fill it with O."""
    expanded_data = np.full((*data.shape, dim_len), 0)
    expanded_data[..., 0] = data
    return expanded_data


@jit
def ageing(production: np.ndarray, nb_timestep_by_cohort: np.ndarray) -> np.ndarray:
    coefficient_except_last = 1.0 / nb_timestep_by_cohort[:-1]
    production_except_last = production[..., :-1]
    first_as_zero = np.zeros((*production.shape[:-1], 1))
    growing = np.concatenate((first_as_zero, production_except_last * coefficient_except_last), axis=-1)
    staying_except_last = production_except_last * (1 - coefficient_except_last)
    staying = np.concatenate((staying_except_last, production[..., -1:]), axis=-1)
    return growing + staying


# @jit(nopython=False, forceobj=True)
@jit
def time_loop(
    primary_production: np.ndarray, cohorts: np.ndarray, mask_temperature: np.ndarray, timestep_number: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The process done during one timestep.

    Warning:
    -------
    - Be sure to transform nan values into 0.

    """
    next_prepoduction = np.zeros(mask_temperature.shape[1:])  # not time
    output_recruited = np.empty(mask_temperature.shape)  # init my output
    output_preproduction = np.empty(mask_temperature.shape)  # init my output
    output_not_recruited = np.empty(mask_temperature.shape)  # init my output
    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], cohorts.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(~mask_temperature[timestep], pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        output_recruited[timestep] = recruited
        output_not_recruited[timestep] = not_recruited
        output_preproduction[timestep] = pre_production
    return (output_recruited, output_not_recruited, output_preproduction)


def compute_preproduction_numba(data: xr.Dataset) -> xr.DataArray:
    data.cf.transpose("functional_group", "T", "Y", "X", "Z", "cohort")

    results_recruited = []
    results_not_recruited = []
    results_preproduction = []
    for fgroup in data.functional_group:
        logger.info(f"Computing production for Cohort {int(fgroup)}")
        fgroup_data = data.sel(functional_group=fgroup).dropna("cohort")
        primary_production = np.nan_to_num(fgroup_data.primary_production.data, 0.0)
        cohorts = fgroup_data.cohort.data
        mask_temperature = np.nan_to_num(fgroup_data.mask_temperature.data, False)
        timestep_number = fgroup_data.timesteps_number.data

        output_recruited, output_not_recruited, output_preproduction = time_loop(
            primary_production=primary_production,
            cohorts=cohorts,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
        )
        results_recruited.append(
            xr.DataArray(
                coords=fgroup_data.mask_temperature.coords,
                dims=fgroup_data.mask_temperature.dims,
                data=output_recruited,
            )
        )
        results_not_recruited.append(
            xr.DataArray(
                coords=fgroup_data.mask_temperature.coords,
                dims=fgroup_data.mask_temperature.dims,
                data=output_not_recruited,
            )
        )
        results_preproduction.append(
            xr.DataArray(
                coords=fgroup_data.mask_temperature.coords,
                dims=fgroup_data.mask_temperature.dims,
                data=output_preproduction,
            )
        )
    res = xr.Dataset(
        {
            "recruited": xr.concat(results_recruited, dim="functional_group"),
            "not_recruited": xr.concat(results_not_recruited, dim="functional_group"),
            "preproduction": xr.concat(results_preproduction, dim="functional_group"),
        },
        coords=data.coords,
    )
    return res
