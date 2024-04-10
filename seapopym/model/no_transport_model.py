"""The LMTL model without ADRE equations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym.function import generator
from seapopym.function.core.kernel import Kernel
from seapopym.function.generator.mask import apply_mask_to_state
from seapopym.logging.custom_logger import logger
from seapopym.model.base_model import BaseModel
from seapopym.plotter import base_functions as pfunctions
from seapopym.standard.coordinates import reorder_dims
from seapopym.standard.types import SeapopymState
from seapopym.writer import base_functions as wfunctions

if TYPE_CHECKING:
    from dask.distributed import Client

    from seapopym.configuration.no_transport.configuration import NoTransportConfiguration


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(self: NoTransportModel, configuration: NoTransportConfiguration) -> None:
        """The constructor of the model allows the user to overcome the default parameters and client behaviors."""
        self._configuration = configuration
        self.state = apply_mask_to_state(reorder_dims(configuration.model_parameters))

        chunk = self.configuration.environment_parameters.chunk.as_dict()

        self._kernel = Kernel(
            [
                generator.global_mask_kernel(chunk=chunk),
                generator.mask_by_fgroup_kernel(chunk=chunk),
                generator.day_length_kernel(
                    chunk=chunk, angle_horizon_sun=configuration.kernel_parameters.angle_horizon_sun
                ),
                generator.average_temperature_kernel(chunk=chunk),
                generator.apply_coefficient_to_primary_production_kernel(chunk=chunk),
                generator.min_temperature_kernel(chunk=chunk),
                generator.mask_temperature_kernel(chunk=chunk),
                generator.cell_area_kernel(chunk=chunk),
                generator.mortality_field_kernel(chunk=chunk),
                generator.production_kernel(
                    chunk=chunk,
                    export_preproduction=configuration.kernel_parameters.compute_preproduction,
                    export_initial_production=configuration.kernel_parameters.compute_initial_conditions,
                ),
                generator.biomass_kernel(chunk=chunk),
            ]
        )

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration:
        """The configuration getter."""
        return self._configuration

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._configuration.environment_parameters.client.client

    @property
    def kernel(self: NoTransportModel) -> Kernel:
        """The kernel getter."""
        return self._kernel

    @property
    def template(self: NoTransportModel) -> SeapopymState:
        """The template getter."""
        return self.kernel.template(self.state)

    @property
    def expected_memory_usage(self: NoTransportModel) -> int:
        """The expected memory usage getter."""
        return f"The expected memory usage is {self.template.nbytes / 1e6:.2f} MB."

    def initialize_dask(self: NoTransportModel) -> None:
        """Initialize the client and configure the model to run in distributed mode."""
        logger.info("Initializing the client.")
        self.configuration.environment_parameters.client.initialize_client()
        chunk = self.configuration.environment_parameters.chunk.as_dict()
        self.state = self.state.chunk(chunk)
        logger.info("Scattering the data to the workers.")
        self.client.scatter(self.state)

    def run(self: NoTransportModel) -> None:
        """Run the model. Wrapper of the pre-production, production and post-production processes."""
        self.state = self.kernel.run(self.state)
        if self.client is not None:
            self.client.persist(self.state)

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        self.configuration.environment_parameters.client.close_client()

    # --- Export functions --- #

    export_state = wfunctions.export_state
    export_biomass = wfunctions.export_biomass
    export_initial_conditions = wfunctions.export_initial_conditions

    # --- Plot functions --- #

    plot_biomass = pfunctions.plot_biomass
