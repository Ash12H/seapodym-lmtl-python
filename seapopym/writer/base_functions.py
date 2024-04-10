"""All the functions to export the data of the model to a file."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr

from seapopym.standard.labels import ConfigurationLabels, ForcingLabels
from seapopym.standard.types import SeapopymForcing, SeapopymState

if TYPE_CHECKING:
    from seapopym.model.no_transport_model import NoTransportModel


def _helper_export_data(
    data: xr.Dataset,
    path: str | Path,
    engine: Literal["zarr", "netcdf4"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> None:
    """Export the data to a file."""

    def _write(data: xr.Dataset, engine: str) -> None:
        if engine == "zarr":
            data.to_zarr(path, mode="w")
        else:
            data.to_netcdf(path, mode="w")

    if mode == "a":
        msg = "The append mode is not implemented yet."
        raise NotImplementedError(msg)

    path = Path(path)

    if engine not in ["zarr", "netcdf4"]:
        msg = "The engine must be either 'zarr' or 'netcdf4'."
    if mode not in ["w", "a"]:
        msg = "The mode must be either 'w' (create or overwrite) or 'a' (append or create)."
        raise ValueError(msg)

    _write(data, engine)


def _helper_check_state(model: NoTransportModel) -> None:
    """Check if the model has state to export."""
    if not hasattr(model, "state") or model.state is None:
        msg = "The model does not have state to export."
        raise ValueError(msg)


def export_state(
    model: NoTransportModel,
    path: str | Path,
    engine: Literal["zarr", "netcdf"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> None:
    """Export the state of the model to a file."""
    _helper_check_state(model)

    _helper_export_data(model.state, path, engine, mode)


def export_initial_conditions(
    model: NoTransportModel,
    path: str | Path | None = None,
    engine: Literal["zarr", "netcdf"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> SeapopymState | None:
    """Export the initial conditions to a file."""
    _helper_check_state(model)
    if ForcingLabels.preproduction not in model.state:
        msg = "The model does not have production to export."
        raise ValueError(msg)
    if ForcingLabels.biomass not in model.state:
        msg = "The model does not have biomass to export."
        raise ValueError(msg)

    if "T" in model.state[ForcingLabels.preproduction].cf.coords:
        # NOTE(Jules): if production_export_preproduction have been used
        init_prod = model.state[ForcingLabels.preproduction].cf.isel(T=-1)
    else:
        # NOTE(Jules): if production_export_initial have been used
        init_prod = model.state[ForcingLabels.preproduction]

    data_to_export = xr.Dataset(
        {
            ConfigurationLabels.initial_condition_production: init_prod,
            ConfigurationLabels.initial_condition_biomass: model.state[ForcingLabels.biomass].cf.isel(T=-1),
        }
    )

    if path is None:
        return data_to_export

    _helper_export_data(data_to_export, path, engine, mode)
    return None


def export_biomass(
    model: NoTransportModel,
    path: str | Path | None = None,
    engine: Literal["zarr", "netcdf"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> SeapopymForcing | None:
    """Export the biomass to a file."""
    _helper_check_state(model)
    if ForcingLabels.biomass not in model.state:
        msg = "The model does not have biomass to export."
        raise ValueError(msg)

    if path is None:
        return model.state[ForcingLabels.biomass]

    _helper_export_data(model.state[ForcingLabels.biomass], path, engine, mode)
    return None
