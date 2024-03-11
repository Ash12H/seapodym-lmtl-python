"""The LMTL model without ADRE equations."""

from __future__ import annotations

from pathlib import Path
from typing import IO, Callable

import xarray as xr
from dask.distributed import Client, Future

from seapodym_lmtl_python.configuration.no_transport import client as no_transport_client
from seapodym_lmtl_python.configuration.no_transport.configuration import NoTransportConfiguration
from seapodym_lmtl_python.configuration.no_transport.labels import ConfigurationLabels, PreproductionLabels
from seapodym_lmtl_python.configuration.no_transport.parameters import NoTransportParameters
from seapodym_lmtl_python.logging.custom_logger import logger
from seapodym_lmtl_python.model.base_model import BaseModel
from seapodym_lmtl_python.post_production.biomass import compute_biomass
from seapodym_lmtl_python.pre_production import pre_production
from seapodym_lmtl_python.pre_production.core import landmask
from seapodym_lmtl_python.production.production import compute_production


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(
        self: NoTransportModel,
        configuration: NoTransportConfiguration | NoTransportParameters | None = None,
        client: Client | None = None,
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
        self._client = client

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration | None:
        """The parameters structure is an attrs class."""
        return self._configuration

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
            logger.warning(warning_message)
            self._client.close()
            self._client = client
        self._client = client

    @classmethod
    def parse(cls: NoTransportModel, configuration_file: str | Path | IO) -> NoTransportModel:
        return NoTransportModel(NoTransportConfiguration.parse(configuration_file))

    def initialize(self: NoTransportModel) -> None:
        self.client = no_transport_client.init_client_locally(self.configuration)

    def generate_configuration(self: NoTransportModel) -> None:
        self.state: xr.Dataset = self.configuration.as_dataset()

    def save_configuration(self: NoTransportModel) -> None:
        """Save the configuration."""

    def pre_production(self: NoTransportModel) -> None:
        """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""

        def apply_if_not_already_computed(forcing_name: str, function: Callable, *args: list, **kargs: dict) -> Future:
            if forcing_name in self.state:
                return self.state[forcing_name]
            return self.client.submit(function, *args, **kargs)

        mask = apply_if_not_already_computed(
            PreproductionLabels.mask_global,
            landmask.landmask_from_nan,
            forcing=self.state[PreproductionLabels.temperature],
        )

        mask_fgroup = apply_if_not_already_computed(
            PreproductionLabels.mask_by_fgroup,
            pre_production.mask_by_fgroup,
            day_layers=self.state[ConfigurationLabels.day_layer],
            night_layers=self.state[ConfigurationLabels.night_layer],
            mask=mask,
        )

        day_length = apply_if_not_already_computed(
            PreproductionLabels.day_length,
            pre_production.compute_daylength,
            time=self.state.cf["T"],
            latitude=self.state.cf["Y"],
            longitude=self.state.cf["X"],
        )

        avg_tmp = apply_if_not_already_computed(
            PreproductionLabels.avg_temperature_by_fgroup,
            pre_production.average_temperature_by_fgroup,
            daylength=day_length,
            mask=mask_fgroup,
            day_layer=self.state[ConfigurationLabels.day_layer],
            night_layer=self.state[ConfigurationLabels.day_layer],
            temperature=self.state[PreproductionLabels.temperature],
        )

        primary_production_by_fgroup = apply_if_not_already_computed(
            PreproductionLabels.primary_production_by_fgroup,
            pre_production.apply_coefficient_to_primary_production,
            primary_production=self.state[PreproductionLabels.primary_production],
            functional_group_coefficient=self.state[ConfigurationLabels.energy_transfert],
        )

        min_temperature_by_cohort = apply_if_not_already_computed(
            PreproductionLabels.min_temperature_by_cohort,
            pre_production.min_temperature_by_cohort,
            mean_timestep=self.state[ConfigurationLabels.mean_timestep],
            tr_max=self.state[ConfigurationLabels.temperature_recruitment_max],
            tr_rate=self.state[ConfigurationLabels.temperature_recruitment_rate],
        )

        mask_temperature = apply_if_not_already_computed(
            PreproductionLabels.mask_temperature,
            pre_production.mask_temperature_by_cohort_by_functional_group,
            min_temperature_by_cohort=min_temperature_by_cohort,
            average_temperature=avg_tmp,
        )

        cell_area = apply_if_not_already_computed(
            PreproductionLabels.cell_area,
            pre_production.compute_cell_area,
            latitude=self.state.cf["Y"],
            longitude=self.state.cf["X"],
            resolution=self.state[ConfigurationLabels.resolution],
        )

        mortality_field = apply_if_not_already_computed(
            PreproductionLabels.mortality_field,
            pre_production.compute_mortality_field,
            average_temperature=avg_tmp,
            inv_lambda_max=self.state[ConfigurationLabels.inv_lambda_max],
            inv_lambda_rate=self.state[ConfigurationLabels.inv_lambda_rate],
            timestep=self.state[ConfigurationLabels.timestep],
        )

        # NOTE(Jules): Some forcing are not used in the production-process so we do not keep them in memory.
        results = self.client.gather(
            {
                # PreproductionLabels.mask_global: mask,
                PreproductionLabels.mask_by_fgroup: mask_fgroup,
                # PreproductionLabels.day_length: day_length,
                # PreproductionLabels.avg_temperature_by_fgroup: avg_tmp,
                PreproductionLabels.primary_production_by_fgroup: primary_production_by_fgroup,
                # PreproductionLabels.min_temperature_by_cohort: min_temperature_by_cohort,
                PreproductionLabels.mask_temperature: mask_temperature,
                PreproductionLabels.cell_area: cell_area,
                PreproductionLabels.mortality_field: mortality_field,
            }
        )

        self.state = xr.merge([self.state, results])

    def production(self: NoTransportModel) -> None:
        """Run the production process that is not explicitly parallel."""
        # TODO(Jules): Manage chunk and export_preproduction parameters. Fixed for now:
        chunk = {ConfigurationLabels.fgroup: 1}
        export_preproduction = False

        self.state = compute_production(
            data=self.state,
            chunk=chunk,
            export_preproduction=export_preproduction,
        )

    def post_production(self: NoTransportModel) -> None:
        """Run the post-production process. Mostly parallel but need the production to be computed."""
        biomass = xr.map_blocks(compute_biomass, self.state)
        self.state = xr.merge([self.state, biomass])

    def save_output(self: NoTransportModel) -> None:
        """Save the outputs of the model."""

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        no_transport_client.close_client_locally(self.client)
