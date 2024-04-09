"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from typing import IO, TYPE_CHECKING

from seapopym.configuration.base_configuration import BaseConfiguration
from seapopym.configuration.no_transport.configuration_to_dataset import as_dataset
from seapopym.configuration.no_transport.parameter import KernelParameters

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr

    from seapopym.configuration.no_transport.parameter import NoTransportParameters
    from seapopym.configuration.parameters.parameter_environment import EnvironmentParameter


class NoTransportConfiguration(BaseConfiguration):
    """Configuration for the NoTransportModel."""

    def __init__(self: NoTransportConfiguration, parameters: NoTransportParameters) -> None:
        """Create a NoTransportConfiguration object."""
        self._parameters = parameters

    @property
    def model_parameters(self: NoTransportConfiguration) -> xr.Dataset:
        """The xarray.Dataset that stores all the model parameters and forcing."""
        return as_dataset(
            functional_groups=self._parameters.functional_groups_parameters.functional_groups,
            forcing_parameters=self._parameters.forcing_parameters,
        )

    @property
    def environment_parameters(self: NoTransportConfiguration) -> EnvironmentParameter:
        """The attrs dataclass that stores all the environment parameters."""
        return self._parameters.environment_parameters

    @property
    def kernel_parameters(self: NoTransportConfiguration) -> KernelParameters:
        """The attrs dataclass that stores all the kernel parameters."""
        return self._parameters.kernel_parameters

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:  # noqa: ARG003
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)
