"""
This module contains the function used to compute the recruited population in  the NotTransport model.
They are run in sequence in timeseries order.
"""

from __future__ import annotations

from typing import Literal

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.core.template import generate_template
from seapopym.function.generator.production.compiled_functions import time_loop
from seapopym.logging.custom_logger import logger
from seapopym.standard.attributs import preproduction_desc, recruited_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels, ProductionLabels


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


def _format_output(
    fgroup_data: xr.Dataset, data: np.ndarray, *, export_preproduction: np.ndarray | None = None
) -> xr.DataArray:
    """Convert the output of the Numba function to a DataArray."""
    template = fgroup_data[PreproductionLabels.mask_temperature]
    if export_preproduction is not None:
        template = template.cf.isel(T=export_preproduction)
    return xr.DataArray(coords=template.coords, dims=template.dims, data=data)


def _production_helper(data: xr.Dataset, *, export_preproduction: np.ndarray | None = None) -> xr.DataArray:
    """
    Compute the production using a numba jit function.

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
    data = data.cf.transpose(*CoordinatesLabels.ordered(), missing_dims="ignore")
    results_recruited = []
    if export_preproduction is not None:
        results_preproduction = []

    for fgroup in data[CoordinatesLabels.functional_group]:
        logger.info(f"Computing production for Cohort {int(fgroup)}")

        fgroup_data = data.sel({CoordinatesLabels.functional_group: fgroup}).dropna(CoordinatesLabels.cohort)
        output_recruited, output_preproduction = time_loop(**_init_forcing(fgroup_data, export_preproduction))

        results_recruited.append(_format_output(fgroup_data, output_recruited))
        if export_preproduction is not None:
            results_preproduction.append(
                _format_output(fgroup_data, output_preproduction, export_preproduction=export_preproduction)
            )

    results = {ProductionLabels.recruited: xr.concat(results_recruited, dim=CoordinatesLabels.functional_group)}
    if export_preproduction is not None:
        results[ProductionLabels.preproduction] = xr.concat(
            results_preproduction, dim=CoordinatesLabels.functional_group
        )
    return xr.Dataset(results)


def production(
    state: xr.Dataset,
    chunk: dict[str, int | Literal["auto"]] | None = None,
    *,
    export_preproduction: np.ndarray | None = None,
) -> xr.Dataset:
    """
    The main fonction to compute the production. It is a wrapper around the `compute_preproduction_numba` function.

    Parameters
    ----------
    state : xr.Dataset
        The input dataset.
    chunk : dict[str, int | Literal["auto"]] | None
        The chunk size for the computation. If None, the default chunk size is used {CoordinatesLabels.functional_group: 1}.
    export_preproduction : np.ndarray | None
        An array (dtype=int) containing the time-index (i.e. timestamps) to export the pre-production. If None, the
        pre-production is not exported.

    Returns
    -------
    output : xr.Dataset
        The output dataset.

    Warning:
    --------
    - Valide chunk keys are : `{CoordinatesLabels.functional_group:..., "X":..., "Y":...}`. Priority should be given to the
    functional group dimension.

    """
    max_dims = (
        CoordinatesLabels.functional_group,
        CoordinatesLabels.time,
        CoordinatesLabels.Y,
        CoordinatesLabels.X,
        CoordinatesLabels.cohort,
    )
    template = {
        ProductionLabels.recruited: generate_template(state, dims=max_dims, attributs=recruited_desc, chunk=chunk)
    }

    if export_preproduction is not None:
        template[ProductionLabels.preproduction] = generate_template(
            state.cf.isel(T=export_preproduction), dims=max_dims, attributs=preproduction_desc, chunk=chunk
        )

    return xr.map_blocks(
        _production_helper, state, kwargs={"export_preproduction": export_preproduction}, template=xr.Dataset(template)
    )
