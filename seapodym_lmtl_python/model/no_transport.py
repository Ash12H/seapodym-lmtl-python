"""The LMTL model without ADRE equations."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from typing import IO, Callable

import xarray as xr
from dask.distributed import Client, Future

from seapodym_lmtl_python.config import model_configuration
from seapodym_lmtl_python.config.client import init_client_locally
from seapodym_lmtl_python.config.parameters import Parameters
from seapodym_lmtl_python.model.abstract_model import BaseModel
from seapodym_lmtl_python.pre_production import pre_production
from seapodym_lmtl_python.pre_production.core import landmask


class ForcingNames(StrEnum):
    mask_global = "mask"
    mask_by_fgroup = "mask_fgroup"
    day_length = "day_length"
    temperature = "temperature"  # from config.parameters
    primary_production = "primary_production"  # from config.parameters


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(
        self: NoTransportModel,
        parameters: Parameters | None = None,
        client: Client | None = None,
    ) -> None:
        """The constructor of the model allows the user to overcome the default parameters and client behaviors."""
        super().__init__()
        self._parameters = parameters
        self._client = client

    @property
    def parameters(self: NoTransportModel) -> Parameters | None:
        """The parameters structure is an attrs class."""
        return self._parameters

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._client

    @client.setter
    def client(self: NoTransportModel, client: Client) -> None:
        """The dask Client setter."""
        if self._client is not None:
            warning_message = (
                f"The model has already a client running at '{self._client.dashboard_link}'."
                f"\nWe are then closing it and starting the new one at '{client.dashboard_link}'"
            )
            logging.warning(warning_message)
            self._client.close()
            self._client = client
        self._client = client

    def parse(self: NoTransportModel, configuration_file: str | Path | IO) -> None:
        self._parameters: Parameters = super().parse(configuration_file)

    def initialize(self: NoTransportModel) -> None:
        self.client = init_client_locally(self.parameters)

    def generate_configuration(self: NoTransportModel) -> None:
        self.state: xr.Dataset = model_configuration.process(self.parameters)

    def save_configuration(self: NoTransportModel) -> None:
        """Save the configuration."""

    def pre_production(self: NoTransportModel) -> None:
        """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""

        def apply_if_not_already_computed(
            forcing_name: str, function: Callable, *args: list, **kargs: dict
        ) -> Future:
            if forcing_name in self.state:
                return self.state[forcing_name]
            return self.client.submit(function(*args, **kargs))

        # 1. Global mask
        mask = apply_if_not_already_computed(
            ForcingNames.mask_global,
            landmask.landmask_from_nan,
            forcing=self.state[ForcingNames.temperature],
        )

        # 2. Mask by functional group
        mask_fgroup = apply_if_not_already_computed(
            ForcingNames.mask_by_fgroup,
            pre_production.mask_by_fgroup,
            day_layers=self.state[model_configuration.ForcingNames.day_position],
            night_layers=self.state[model_configuration.ForcingNames.night_position],
            mask=mask,
        )

        # 3. Mask by functional group
        day_length = apply_if_not_already_computed(
            ForcingNames.day_length,
            pre_production.compute_daylength,
            time=self.state.cf["T"],
            latitude=self.state.cf["Y"],
            longitude=self.state.cf["X"],
        )

    def production(self: NoTransportModel) -> None:
        """Run the production process that is not explicitly parallel."""

    def post_production(self: NoTransportModel) -> None:
        """Run the post-production process. Mostly parallel but need the production to be computed."""

    def save_output(self: NoTransportModel) -> None:
        """Save the outputs of the model."""

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
