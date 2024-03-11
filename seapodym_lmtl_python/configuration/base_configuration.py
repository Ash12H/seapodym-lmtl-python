"""Implementation of the base class for all configuration classes."""

from __future__ import annotations

import abc
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import attrs
    import xarray as xr


class BaseConfiguration(abc.ABC):
    """Base class for all configurations."""

    @property
    @abc.abstractmethod
    def parameters(self: BaseConfiguration) -> attrs.Attribute:
        """The attrs dataclass that stores all the model parameters."""
        return self._parameters

    @abc.abstractclassmethod
    def parse(
        cls: BaseConfiguration, configuration_file: str | Path | IO
    ) -> BaseConfiguration:
        """Parse the configuration file and create a BaseConfiguration object."""

    @abc.abstractmethod
    def as_dataset(self: BaseConfiguration) -> xr.Dataset:
        """Return the configuration as a xarray.Dataset."""
