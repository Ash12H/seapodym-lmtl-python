"""The writer class for the no transport model."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr

from seapopym.standard.labels import ConfigurationLabels, PostproductionLabels, ProductionLabels

if TYPE_CHECKING:
    from seapopym.model.no_transport_model import NoTransportModel


def _helper_export_data(
    data: xr.Dataset,
    path: str | Path,
    engine: Literal["zarr", "netcdf"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> None:
    """Export the data to a file."""
    path = Path(path)

    if mode == "a" and not path.exists():
        mode = "w"

    if mode == "w":
        if engine == "zarr":
            data.to_zarr(path, mode="w")
        else:
            data.to_netcdf(path)
    elif mode == "a":
        old_data = xr.open_zarr(path) if engine == "zarr" else xr.open_dataset(path)
        data = xr.merge([old_data, data])
        if engine == "zarr":
            data.to_zarr(path, mode="w")
        else:
            data.to_netcdf(path)
    else:
        msg = "The mode must be either 'w' or 'a'."
        raise ValueError(msg)


def export_state(
    self: NoTransportModel,
    path: str | Path,
    engine: Literal["zarr", "netcdf"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> None:
    """Export the state of the model to a file."""
    if not hasattr(self, "state") or self.state is None:
        msg = "The model does not have state to export."
        raise ValueError(msg)

    _helper_export_data(self.state, path, engine, mode)


def export_initial_conditions(
    self: NoTransportModel,
    path: str | Path,
    engine: Literal["zarr", "netcdf"] = "zarr",
    mode: Literal["w", "a"] = "w",
) -> None:
    """Export the initial conditions to a file."""
    if not hasattr(self, "state") or self.state is None:
        msg = "The model does not have initial conditions to export."
        raise ValueError(msg)
    if ProductionLabels.preproduction not in self.state:
        msg = "The model does not have production to export."
        raise ValueError(msg)
    if PostproductionLabels.biomass not in self.state:
        msg = "The model does not have biomass to export."
        raise ValueError(msg)

    data_to_export = xr.Dataset(
        {
            ConfigurationLabels.initial_condition_production: self.state[ProductionLabels.preproduction],
            ConfigurationLabels.initial_condition_biomass: self.state[PostproductionLabels.biomass],
        }
    )
    data_to_export = data_to_export.cf.isel(T=-1)

    _helper_export_data(data_to_export, path, engine, mode)
