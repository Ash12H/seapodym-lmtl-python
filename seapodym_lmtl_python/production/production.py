"""This module contains the function used in the **dependent** process. They are run in sequence in timeseries order."""

from __future__ import annotations

from typing import Literal

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
from numba import jit

from seapodym_lmtl_python.logging.custom_logger import logger
from seapodym_lmtl_python.standard.labels import (
    ConfigurationLabels,
    PreproductionLabels,
    ProductionLabels,
)
from seapodym_lmtl_python.standard.units import StandardUnitsLabels


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


def _init_forcing(fgroup_data: xr.Dataset, export_preproduction: np.ndarray | None) -> dict[str, np.ndarray]:
    """Initialise the forcing data used in the Numba function that compute production."""

    def standardize_forcing(forcing: xr.DataArray, nan: object = 0.0, dtype: type = np.float64) -> np.ndarray:
        """Refer to Numba documentation about array typing."""
        return np.nan_to_num(x=forcing.data, nan=nan).astype(dtype)

    if ConfigurationLabels.initial_condition_production not in fgroup_data:
        initial_condition = None
    else:
        initial_condition = standardize_forcing(fgroup_data[ConfigurationLabels.initial_condition_production])

    return {
        "primary_production": standardize_forcing(fgroup_data[PreproductionLabels.primary_production_by_fgroup]),
        "mask_temperature": standardize_forcing(fgroup_data[PreproductionLabels.mask_temperature]),
        "timestep_number": standardize_forcing(fgroup_data[ConfigurationLabels.timesteps_number], False, bool),
        "initial_production": initial_condition,
        "export_preproduction": export_preproduction,
    }


def _format_recruited(fgroup_data: xr.Dataset, data: np.ndarray) -> xr.DataArray:
    """Convert the output of the Numba function to a DataArray."""
    recruited_template = fgroup_data[PreproductionLabels.mask_temperature]
    return xr.DataArray(coords=recruited_template.coords, dims=recruited_template.dims, data=data)


def _format_pre_prod(
    fgroup_data: xr.Dataset, export_preproduction: np.ndarray | None, data: np.ndarray
) -> xr.DataArray:
    """Convert the output of the Numba function to a DataArray."""
    preprod_template = fgroup_data[PreproductionLabels.mask_temperature].cf.isel(T=export_preproduction)
    return xr.DataArray(coords=preprod_template.coords, dims=preprod_template.dims, data=data)


def compute_preproduction_numba(data: xr.Dataset, *, export_preproduction: np.ndarray | None = None) -> xr.DataArray:
    """
    Compute the pre-production using a numba jit function.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset.
    export_preproduction : np.ndarray | None
        An array containing the time-index (i.e. timestamps) to export the pre-production. If None, the pre-production
        is not exported.

    Returns
    -------
    output : xr.Dataset
        The output dataset.

    """
    data = data.cf.transpose(ConfigurationLabels.fgroup, "T", "Y", "X", "Z", ConfigurationLabels.cohort)
    results_recruited = []
    if export_preproduction is not None:
        results_preproduction = []

    for fgroup in data[ConfigurationLabels.fgroup]:
        logger.info(f"Computing production for Cohort {int(fgroup)}")

        fgroup_data = data.sel({ConfigurationLabels.fgroup: fgroup}).dropna(ConfigurationLabels.cohort)
        output_recruited, output_preproduction = time_loop(**_init_forcing(fgroup_data, export_preproduction))

        results_recruited.append(_format_recruited(fgroup_data, output_recruited))
        if export_preproduction is not None:
            results_preproduction.append(_format_pre_prod(fgroup_data, export_preproduction, output_preproduction))

    results = {
        ProductionLabels.recruited: xr.concat(results_recruited, dim=ConfigurationLabels.fgroup, combine_attrs="drop")
    }
    if export_preproduction is not None:
        results[ProductionLabels.preproduction] = xr.concat(
            results_preproduction, dim=ConfigurationLabels.fgroup, combine_attrs="drop"
        )
    return xr.Dataset(results, coords=data.coords)


def compute_production(
    data: xr.Dataset,
    chunk: dict[str, int | Literal["auto"]] | None = None,
    *,
    export_preproduction: np.ndarray | None = None,
) -> xr.Dataset:
    """
    The main fonction to compute the production. It is a wrapper around the `compute_preproduction_numba` function.

    Parameters
    ----------
    data : xr.Dataset
        The input dataset.
    chunk : dict[str, int | Literal["auto"]] | None
        The chunk size for the computation. If None, the default chunk size is used {ConfigurationLabels.fgroup: 1}.
    export_preproduction : np.ndarray | None
        An array (dtype=int) containing the time-index (i.e. timestamps) to export the pre-production. If None, the
        pre-production is not exported.

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

    template = xr.Dataset(
        {ProductionLabels.recruited: data[PreproductionLabels.mask_temperature]},
        coords=data.coords,
    )
    if export_preproduction is not None:
        template[ProductionLabels.preproduction] = data[PreproductionLabels.mask_temperature].cf.isel(
            T=export_preproduction
        )

    output = xr.map_blocks(
        compute_preproduction_numba, data, kwargs={"export_preproduction": export_preproduction}, template=template
    )

    attrs_recruited = {
        "standard_name": "production",
        "long_name": "production",
        "units": str(StandardUnitsLabels.production.units),
    }
    output[ProductionLabels.recruited].attrs.clear()
    output[ProductionLabels.recruited].attrs.update(attrs_recruited)

    if export_preproduction is not None:
        attrs_preproduction = {
            "standard_name": "pre-production",
            "long_name": "pre-production",
            "description": "The entire population before recruitment, divided into cohorts.",
            "units": str(StandardUnitsLabels.production.units),
        }
        output[ProductionLabels.preproduction].attrs.clear()
        output[ProductionLabels.preproduction].attrs.update(attrs_preproduction)

    output = xr.merge([data, output])
    return output.persist()
