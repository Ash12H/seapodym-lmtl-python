"""Implementation of the base class for all models."""

from __future__ import annotations

import abc
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr


class BaseModel(abc.ABC):
    """The base class for all models."""

    state: xr.Dataset | None

    @property
    @abc.abstractmethod
    def configuration(self: BaseModel) -> object:
        """The structure that store the model parameters."""

    @property
    @abc.abstractmethod
    def client(self: BaseModel) -> object:
        """The client getter."""

    @abc.abstractclassmethod
    def parse(cls: BaseModel, configuration_file: str | Path | IO) -> BaseModel:
        """Parse the configuration file."""

    @abc.abstractmethod
    def initialize_client(self: BaseModel) -> None:
        """
        Initialize local or remote system.

        The client can be a local client or a remote cluster. If the client is not initialized, the model will run
        without parallelism and therefore without dask support.

        ### Xarray documentation on dask :
        Xarray integrates with Dask to support parallel and continuous calculations on datasets that don't fit in
        memory.

        """

    @abc.abstractmethod
    def generate_configuration(self: BaseModel) -> None:
        """Generate the configuration using parameters extracted by parse or given by the user."""

    @abc.abstractmethod
    def run(self: BaseModel) -> None:
        """Run the model."""

    @abc.abstractmethod
    def close(self: BaseModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
