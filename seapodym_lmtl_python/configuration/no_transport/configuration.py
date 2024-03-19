"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import xarray as xr

from seapodym_lmtl_python.configuration.base_configuration import BaseConfiguration
from seapodym_lmtl_python.configuration.no_transport.configuration_to_dataset import as_dataset
from seapodym_lmtl_python.configuration.no_transport.parameter_environment import EnvironmentParameter

if TYPE_CHECKING:
    from seapodym_lmtl_python.configuration.no_transport.parameters import NoTransportParameters


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

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)
