"""The LMTL model without ADRE equations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from seapopym.function import generator
from seapopym.function.core.kernel import Kernel
from seapopym.function.core.mask import apply_mask_to_state
from seapopym.logging.custom_logger import logger
from seapopym.model.base_model import BaseModel
from seapopym.plotter import base_functions as pfunctions
from seapopym.standard.coordinates import reorder_dims
from seapopym.writer import base_functions as wfunctions

if TYPE_CHECKING:
    import xarray as xr
    from dask.distributed import Client

    from seapopym.configuration.no_transport.configuration import NoTransportConfiguration


class NoTransportModel(BaseModel):
    """Implement the LMTL model without the transport (Advection-Diffusion)."""

    def __init__(self: NoTransportModel, configuration: NoTransportConfiguration) -> None:
        """The constructor of the model allows the user to overcome the default parameters and client behaviors."""
        self._configuration = configuration
        self.state = apply_mask_to_state(reorder_dims(configuration.model_parameters))

    @property
    def configuration(self: NoTransportModel) -> NoTransportConfiguration:
        """The configuration getter."""
        return self._configuration

    @property
    def client(self: NoTransportModel) -> Client | None:
        """The dask Client getter."""
        return self._configuration.environment_parameters.client.client

    # TODO(Jules): Include angle_horizon_sun in the configuration. Then kernel will be a property.
    def kernel(self: NoTransportModel, angle_horizon_sun: float = 0) -> Kernel:
        """The kernel getter."""

        def _preproduction_converter() -> tuple[np.ndarray, xr.DataArray] | tuple[None, None]:
            """
            The production process requires a specific format, we then convert parameters to a np.ndarray that contains
            the indices of the timestamps to export. None is returned if no timestamps are provided.
            """
            if self.configuration.environment_parameters.output.pre_production is None:
                return (None, None)

            timestamps = self.configuration.environment_parameters.output.pre_production.timestamps
            data = self.state.cf["T"]

            if timestamps is None:
                return (None, None)
            if timestamps == "all":
                return (np.arange(data.size), data.cf["T"])
            if np.all([isinstance(x, int) for x in timestamps]):
                selected_dates = data.cf.isel(T=timestamps)
            elif np.all([isinstance(x, str) for x in timestamps]):
                selected_dates = data.cf.sel(T=timestamps, method="nearest")
            else:
                msg = "The timestamps must be either 'all', a list of integers or a list of strings."
                raise TypeError(msg)
            index = np.arange(data.size)[data.isin(selected_dates)]
            coords = data.cf.isel(T=index)
            return (index, coords)

        preproduction_time_index, preproduction_time_coords = _preproduction_converter()

        chunk = self.configuration.environment_parameters.chunk.as_dict()

        return Kernel(
            [
                generator.global_mask_kernel(chunk=chunk),
                generator.mask_by_fgroup_kernel(chunk=chunk),
                generator.day_length_kernel(chunk=chunk, angle_horizon_sun=angle_horizon_sun),
                generator.average_temperature_kernel(chunk=chunk),
                generator.apply_coefficient_to_primary_production_kernel(chunk=chunk),
                generator.min_temperature_kernel(chunk=chunk),
                generator.mask_temperature_kernel(chunk=chunk),
                generator.cell_area_kernel(chunk=chunk),
                generator.mortality_field_kernel(chunk=chunk),
                generator.production_kernel(
                    chunk=chunk,
                    preproduction_time_coords=preproduction_time_coords,
                    preproduction_time_index=preproduction_time_index,
                ),
                generator.biomass_kernel(chunk=chunk),
            ]
        )

    def initialize_dask(self: NoTransportModel) -> None:
        """Initialize the client and configure the model to run in distributed mode."""
        logger.info("Initializing the client.")
        self.configuration.environment_parameters.client.initialize_client()
        chunk = self.configuration.environment_parameters.chunk.as_dict()
        self.state = self.state.chunk(chunk)
        logger.info("Scattering the data to the workers.")
        self.client.scatter(self.state)

    # def _pre_production(self: NoTransportModel) -> None:
    #     """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""

    #     def _apply_functions(state: SeapopymState, kernel: dict, chunk: dict) -> xr.Dataset:
    #         for name, func in kernel.items():
    #             if name in state:
    #                 logger.info(f"{name} already present in the state, skipping the computation")
    #             else:
    #                 logger.info(f"Computing {name}.")
    #                 if callable(func):
    #                     self.state[name] = func(self.state, chunk=chunk)
    #                 else:
    #                     self.state[name] = func[0](self.state, chunk=chunk, **func[1])

    #     logger.debug("Starting the pre-production process.")
    #     kernel = {
    #         PreproductionLabels.global_mask: (generator.global_mask, {"lazy": ConfigurationLabels.temperature}),
    #         PreproductionLabels.mask_by_fgroup: (generator.mask_by_fgroup, {"lazy": ConfigurationLabels.temperature}),
    #         PreproductionLabels.day_length: (generator.day_length, {"lazy": ConfigurationLabels.primary_production}),
    #         PreproductionLabels.avg_temperature_by_fgroup: (
    #             generator.average_temperature,
    #             {"lazy": ConfigurationLabels.primary_production},
    #         ),
    #         PreproductionLabels.primary_production_by_fgroup: (
    #             generator.apply_coefficient_to_primary_production,
    #             {"lazy": ConfigurationLabels.primary_production},
    #         ),
    #         PreproductionLabels.min_temperature: generator.min_temperature,
    #         PreproductionLabels.mask_temperature: (
    #             generator.mask_temperature,
    #             {"lazy": ConfigurationLabels.primary_production},
    #         ),
    #         PreproductionLabels.cell_area: (generator.cell_area, {"lazy": ConfigurationLabels.primary_production}),
    #         PreproductionLabels.mortality_field: (
    #             generator.mortality_field,
    #             {"lazy": ConfigurationLabels.primary_production},
    #         ),
    #     }
    #     chunk = self.configuration.environment_parameters.chunk.as_dict()
    #     _apply_functions(self.state, kernel, chunk)
    #     logger.debug("End of the pre-production process.")

    # def _production(self: NoTransportModel) -> None:
    #     """Run the production process that is not explicitly parallel."""

    #     def _preproduction_converter() -> np.ndarray | None:
    #         """
    #         The production process requires a specific format, we then convert parameters to a np.ndarray that contains
    #         the indices of the timestamps to export. None is returned if no timestamps are provided.
    #         """
    #         logger.debug("Starting the production process.")
    #         timestamps = self.configuration.environment_parameters.output.pre_production.timestamps
    #         data = self.state.cf["T"]

    #         if timestamps is None:
    #             return None
    #         if timestamps == "all":
    #             return np.arange(data.size)
    #         if np.all([isinstance(x, int) for x in timestamps]):
    #             selected_dates = data.cf.isel(T=timestamps)
    #         elif np.all([isinstance(x, str) for x in timestamps]):
    #             selected_dates = data.cf.sel(T=timestamps, method="nearest")
    #         else:
    #             msg = "The timestamps must be either 'all', a list of integers or a list of strings."
    #             raise TypeError(msg)
    #         return np.arange(data.size)[data.isin(selected_dates)]

    #     if self.configuration.environment_parameters.output.pre_production is None:
    #         export_preproduction = None
    #     else:
    #         export_preproduction = _preproduction_converter()

    #     output = production(
    #         state=self.state,
    #         chunk=self.configuration.environment_parameters.chunk.as_dict(),
    #         export_preproduction=export_preproduction,
    #     )
    #     self.state = xr.merge([self.state, output])

    # def _post_production(self: NoTransportModel) -> None:
    #     """Run the post-production process. Mostly parallel but need the production to be computed."""
    #     output = biomass(self.state, chunk=self.configuration.environment_parameters.chunk.as_dict())
    #     self.state = xr.merge([self.state, output])

    def run(self: NoTransportModel) -> None:
        """Run the model. Wrapper of the pre-production, production and post-production processes."""
        self.state = self.kernel().run(self.state)
        self.state.persist()

    def close(self: NoTransportModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
        self.configuration.environment_parameters.client.close_client()

    # --- Export functions --- #

    export_state = wfunctions.export_state
    export_biomass = wfunctions.export_biomass
    export_initial_conditions = wfunctions.export_initial_conditions

    # --- Plot functions --- #

    plot_biomass = pfunctions.plot_biomass
