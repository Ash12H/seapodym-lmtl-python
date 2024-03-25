"""The LMTL model without ADRE equations."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

import numpy as np
import xarray as xr

from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.configuration.no_transport.parameter import NoTransportParameters
from seapopym.function import generator
from seapopym.function.generator.biomass import compute_biomass
from seapopym.function.generator.production import compute_production
from seapopym.logging.custom_logger import logger
from seapopym.model.base_model import BaseModel
from seapopym.standard.labels import PreproductionLabels

if TYPE_CHECKING:
    from pathlib import Path

    from dask.distributed import Client


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(
        self: NoTransportModel, configuration: NoTransportConfiguration | NoTransportParameters | None = None
    ) -> None:
        """The constructor of the model allows the user to overcome the default parameters and client behaviors."""
        super().__init__()
        if isinstance(configuration, NoTransportParameters):
            self._configuration = NoTransportConfiguration(configuration)
        elif isinstance(configuration, NoTransportConfiguration):
            self._configuration = configuration
        else:
            msg = "The configuration must be an instance of NoTransportConfiguration or NoTransportParameters."
            raise TypeError(msg)
        self.state = None

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration | None:
        """The parameters structure is an attrs class."""
        return self._configuration

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._configuration.environment_parameters.client.client

    @classmethod
    def parse(cls: NoTransportModel, configuration_file: str | Path | IO) -> NoTransportModel:
        return NoTransportModel(NoTransportConfiguration.parse(configuration_file))

    def initialize(self: NoTransportModel) -> None:
        self.configuration.environment_parameters.client.initialize_client()

    def generate_configuration(self: NoTransportModel) -> None:
        self.state: xr.Dataset = self.configuration.model_parameters

    # TODO(Jules): Rename this method to save_state ?
    def save_configuration(self: NoTransportModel) -> None:
        """Save the configuration."""

    def pre_production(self: NoTransportModel) -> None:
        """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""
        chunk = self.configuration.environment_parameters.chunk.as_dict()
        kernel = {
            PreproductionLabels.global_mask: generator.global_mask,
            PreproductionLabels.mask_by_fgroup: generator.mask_by_fgroup,
            PreproductionLabels.day_length: generator.day_length,
            PreproductionLabels.avg_temperature_by_fgroup: generator.average_temperature,
            PreproductionLabels.primary_production_by_fgroup: generator.apply_coefficient_to_primary_production,
            PreproductionLabels.min_temperature: generator.min_temperature,
            PreproductionLabels.mask_temperature: generator.mask_temperature,
            PreproductionLabels.cell_area: generator.cell_area,
            PreproductionLabels.mortality_field: generator.mortality_field,
        }

        self.state = self.state.chunk(chunk)
        if self.client is not None:
            logger.info("Scattering the data to the workers.")
            self.client.scatter(self.state)
        for name, func in kernel.items():
            self.state[name] = func(self.state, chunk=chunk)

        # self.state.persist()

    def production(self: NoTransportModel) -> None:
        """Run the production process that is not explicitly parallel."""

        def _preproduction_converter() -> np.ndarray | None:
            """
            The production process requires a specific format, we then convert parameters to a np.ndarray that contains
            the indices of the timestamps to export. None is returned if no timestamps are required.
            """
            timestamps = self.configuration.environment_parameters.output.pre_production.timestamps
            data = self.state.cf["T"]

            if timestamps is None:
                return None
            if timestamps == "all":
                return np.arange(data.size)
            if np.all([isinstance(x, int) for x in timestamps]):
                selected_dates = data.isel(time=timestamps)
            elif np.all([isinstance(x, str) for x in timestamps]):
                selected_dates = data.sel(time=timestamps, method="nearest")
            else:
                msg = "The timestamps must be either 'all', a list of integers or a list of strings."
                raise TypeError(msg)
            return np.arange(data.size)[data.isin(selected_dates)]

        self.state = compute_production(
            data=self.state,
            chunk=self.configuration.environment_parameters.chunk.as_dict(),
            export_preproduction=_preproduction_converter(),
        )

    def post_production(self: NoTransportModel) -> None:
        """Run the post-production process. Mostly parallel but need the production to be computed."""
        biomass = xr.map_blocks(compute_biomass, self.state)
        self.state = xr.merge([self.state, biomass]).persist()

    def save_output(self: NoTransportModel) -> None:
        """Save the outputs of the model."""
        # 1. biomass
        # param_biomass = self.configuration.environment_parameters.output.biomass
        # biomass_field = self.state[PostproductionLabels.biomass]

        # 2. production

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        self.configuration.environment_parameters.client.close_client()
