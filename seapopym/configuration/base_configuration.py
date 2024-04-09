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

    @classmethod
    @abc.abstractmethod
    def parse(cls: BaseConfiguration, configuration_file: str | Path | IO) -> BaseConfiguration:
        """Parse the configuration file and create a BaseConfiguration object."""

    @property
    @abc.abstractmethod
    def model_parameters(self: BaseConfiguration) -> xr.Dataset:
        """The xarray.Dataset that stores all the model parameters and forcing."""

    @property
    @abc.abstractmethod
    def environment_parameters(self: BaseConfiguration) -> attrs.Attribute:
        """The attrs dataclass that stores all the environment parameters."""

    @property
    @abc.abstractmethod
    def kernel_parameters(self: BaseConfiguration) -> attrs.Attribute:
        """The attrs dataclass that stores all the kernel parameters."""
