"""This module contains the function used in the **dependent** process. They are run in sequence in timeseries order."""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from numba import jit

from seapodym_lmtl_python.configuration.no_transport.labels import (
    ConfigurationLabels,
    PreproductionLabels,
    ProductionLabels,
)
from seapodym_lmtl_python.logging.custom_logger import logger


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
    expanded_data = np.full((*data.shape, dim_len), 0)
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
    first_as_zero = np.zeros((*production.shape[:-1], 1))
    growing = np.concatenate((first_as_zero, production_except_last * coefficient_except_last), axis=-1)
    staying_except_last = production_except_last * (1 - coefficient_except_last)
    staying = np.concatenate((staying_except_last, production[..., -1:]), axis=-1)
    return growing + staying


@jit
def time_loop(
    primary_production: np.ndarray,
    mask_temperature: np.ndarray,
    timestep_number: np.ndarray,
    initial_production: np.ndarray | None,
    *,
    export_preproduction: bool = False,
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
    export_preproduction : bool
        If True, the pre-production is included in the output.

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
    # INIT OUTPUTS
    output_recruited = np.empty(mask_temperature.shape)
    if export_preproduction:
        output_preproduction = np.empty(mask_temperature.shape)
    # MAIN COMPUTATION
    next_prepoduction = np.zeros(mask_temperature.shape[1:]) if initial_production is None else initial_production
    # TODO(Jules) : Add initialisation from configuration
    # next_prepoduction = initial_biomass(ie. t=-1)
    for timestep in range(primary_production.shape[0]):
        pre_production = expand_dims(primary_production[timestep], timestep_number.size)
        pre_production = pre_production + next_prepoduction
        if timestep < primary_production.shape[0] - 1:
            not_recruited = np.where(~mask_temperature[timestep], pre_production, 0)
            next_prepoduction = ageing(not_recruited, timestep_number)
        recruited = np.where(mask_temperature[timestep], pre_production, 0)
        # UPDATE OUTPUTS
        output_recruited[timestep] = recruited
        if export_preproduction:
            output_preproduction[timestep] = pre_production
    return (output_recruited, output_preproduction if export_preproduction else None)


def compute_preproduction_numba(data: xr.Dataset, *, export_preproduction: bool = False) -> xr.DataArray:
    """
    Compute the pre-production using numba jit.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset.
    export_preproduction : bool
        If True, the pre-production is included in the output dataset.

    Returns
    -------
    output : xr.Dataset
        The output dataset.

    """
    data = data.cf.transpose(ConfigurationLabels.fgroup, "T", "Y", "X", "Z", ConfigurationLabels.cohort)
    results_recruited = []
    if export_preproduction:
        results_preproduction = []

    for fgroup in data[ConfigurationLabels.fgroup]:
        logger.info(f"Computing production for Cohort {int(fgroup)}")
        fgroup_data = data.sel({ConfigurationLabels.fgroup: fgroup}).dropna(ConfigurationLabels.cohort)

        output_recruited, output_preproduction = time_loop(
            primary_production=np.nan_to_num(fgroup_data[PreproductionLabels.primary_production].data, 0.0),
            mask_temperature=np.nan_to_num(fgroup_data[PreproductionLabels.mask_temperature].data, False),
            timestep_number=fgroup_data[ConfigurationLabels.timesteps_number].data,
            # TODO(Jules) : Add initialisation from configuration (ie. initial_production)
            initial_production=None,
            export_preproduction=export_preproduction,
        )

        if export_preproduction:
            results_preproduction.append(
                xr.DataArray(
                    coords=fgroup_data[PreproductionLabels.mask_temperature].coords,
                    dims=fgroup_data[PreproductionLabels.mask_temperature].dims,
                    data=output_preproduction,
                )
            )
        results_recruited.append(
            xr.DataArray(
                coords=fgroup_data[PreproductionLabels.mask_temperature].coords,
                dims=fgroup_data[PreproductionLabels.mask_temperature].dims,
                data=output_recruited,
            )
        )

    results = {
        ProductionLabels.recruited: xr.concat(results_recruited, dim=ConfigurationLabels.fgroup),
    }
    if export_preproduction:
        results[ProductionLabels.preproduction] = xr.concat(results_preproduction, dim=ConfigurationLabels.fgroup)
    return xr.Dataset(results, coords=data.coords)


def compute_production(
    data: xr.Dataset, chunk: dict[str, int | Literal["auto"]] | None = None, *, export_preproduction: bool = False
) -> xr.Dataset:
    """
    The main fonction to compute the production. It is a wrapper around the `compute_preproduction_numba` function.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset.
    chunk : dict[str, int | Literal["auto"]] | None
        The chunk size for the computation. If None, the default chunk size is used {ConfigurationLabels.fgroup: 1}.
    export_preproduction : bool
        If True, the pre-production is included in the output dataset.

    Returns
    -------
    output : xr.Dataset
        The output dataset.

    Warning:
    --------
    - Valide chunk keys are : `{ConfigurationLabels.fgroup:..., "X":..., "Y":...}`. Priority should be given to the
    functional group dimension.

    """
    if chunk is None:
        chunk = {ConfigurationLabels.fgroup: 1}
    data = data.cf.chunk(chunk).unify_chunks()

    template = xr.Dataset({ProductionLabels.recruited: data[PreproductionLabels.mask_temperature]}, coords=data.coords)
    if export_preproduction:
        template[ProductionLabels.preproduction] = data[PreproductionLabels.mask_temperature]

    output = xr.map_blocks(
        compute_preproduction_numba, data, kwargs={"export_preproduction": export_preproduction}, template=template
    )
    output = xr.merge([data, output])
    return output.persist()
