"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import attrs
import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapodym_lmtl_python.configuration.base_configuration import BaseConfiguration
from seapodym_lmtl_python.configuration.no_transport.configuration_to_dataset import as_dataset
from seapodym_lmtl_python.configuration.no_transport.labels import ConfigurationLabels

if TYPE_CHECKING:
    from seapodym_lmtl_python.configuration.no_transport.parameters import NoTransportParameters


class NoTransportConfiguration(BaseConfiguration):
    """Configuration for the NoTransportModel."""

    def __init__(self: NoTransportConfiguration, parameters: NoTransportParameters) -> None:
        """Create a NoTransportConfiguration object."""
        self._parameters = parameters

    @property
    def parameters(self: NoTransportConfiguration) -> NoTransportParameters:
        """The attrs dataclass that stores all the model parameters."""
        return self._parameters

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

    def as_dataset(self: NoTransportConfiguration) -> xr.Dataset:
        """Return the configuration as a xarray.Dataset."""
        return as_dataset(
            functional_groups=self.parameters.functional_groups_parameters.functional_groups,
            forcing_parameters=self.parameters.forcing_parameters,
        )
