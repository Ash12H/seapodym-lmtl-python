"""This module contains the compiled (JIT) functions used by the production generator."""

from __future__ import annotations

import numpy as np
from numba import jit


@jit
def expand_dims(data: np.ndarray, dim_len: int) -> np.ndarray:
    """
    Add a new dimension to the DataArray and fill it with O.

    Parameters
    ----------
    data : np.ndarray
        The data to expand.
    dim_len : int
        The length of the new dimension.

    Returns
    -------
    expanded_data : np.ndarray
        The expanded data.

    """
    expanded_data = np.full((*data.shape, dim_len), 0.0, dtype=np.float64)
    expanded_data[..., 0] = data
    return expanded_data


@jit
def ageing(production: np.ndarray, nb_timestep_by_cohort: np.ndarray) -> np.ndarray:
    """
    Age the production by rolling over part of it to the next age. The proportion of production moved to the next age
    cohort is defined by the inverse of the number of time steps per cohort.

    Parameters
    ----------
    production : np.ndarray
        The production to age.
    nb_timestep_by_cohort : np.ndarray
        The number of timestep by cohort.

    Returns
    -------
    aged_production : np.ndarray
        The aged production.

    """
    coefficient_except_last = 1.0 / nb_timestep_by_cohort[:-1]
    production_except_last = production[..., :-1]
    first_as_zero = np.zeros((*production.shape[:-1], 1), dtype=np.float64)
    growing = np.concatenate((first_as_zero, production_except_last * coefficient_except_last), axis=-1)
    staying_except_last = production_except_last * (1 - coefficient_except_last)
    staying = np.concatenate((staying_except_last, production[..., -1:]), axis=-1)
    return growing + staying


@jit
def time_loop(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
    export_preproduction: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    The processes done during the time range.

    Parameters
    ----------
    primary_production : np.ndarray
        The primary production. Dims : [T, X, Y].
    mask_temperature : np.ndarray
        The temperature mask. Dims : [T, X, Y, Cohort].
    timestep_number : np.ndarray
        The number of timestep. Dims : [Cohort]
    initial_production : np.ndarray | None
        The initial production. Dims : [X, Y, Cohort]
        If None is given then initial_production is set to `np.zeros((T.size, Y.size, X.size))`.
    export_preproduction : np.ndarray | None
        An array containing the time-index (i.e. timestamps) to export the pre-production. If None, the pre-production
        is not exported.

    Returns
    -------
    output_recruited : np.ndarray
        The recruited production. Dims : [T, X, Y, Cohort]
    output_preproduction : np.ndarray
        The pre-production if `export_preproduction` is True, None otherwise. Dims : [T, X, Y, Cohort]


    Warning:
    -------
    - Be sure to transform nan values into 0.
    - The dimensions order of the input arrays must be [Time, Latitude, Longitude, Cohort].

    """
    output_recruited = np.empty(mask_temperature.shape)
    output_preproduction = None
    if export_preproduction is not None:
        exported_preproduction_shape = (export_preproduction.size, *mask_temperature.shape[1:])
        output_preproduction = np.empty(exported_preproduction_shape, dtype=np.float64)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production

    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)

        output_recruited[timestep] = recruited
        if (export_preproduction is not None) and (timestep in export_preproduction):
            output_preproduction[timestep] = pre_production

    return (output_recruited, output_preproduction if export_preproduction is not None else None)


@jit
def production(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> np.ndarray:
    output_recruited = np.empty(mask_temperature.shape)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production
    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        output_recruited[timestep] = recruited
    return np.sum(output_recruited, axis=-1)


@jit
def production_export_preproduction(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_recruited = np.empty(mask_temperature.shape)
    output_preproduction = np.empty(mask_temperature.shape, dtype=np.float64)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production

    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        # if timestep < primary_production.shape[0] - 1:
        not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
        next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)

        output_recruited[timestep] = recruited
        output_preproduction[timestep] = pre_production
    output_preproduction[-1, ...] = next_prepoduction
    return (np.sum(output_recruited, axis=-1), output_preproduction)


@jit
def production_export_initial(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    output_recruited = np.empty(mask_temperature.shape)
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production

    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(np.logical_not(mask_temperature[timestep]), pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        output_recruited[timestep] = recruited
    return (np.sum(output_recruited, axis=-1), next_prepoduction)
