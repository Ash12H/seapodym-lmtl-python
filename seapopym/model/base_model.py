"""Implementation of the base class for all models."""

from __future__ import annotations

import abc
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class BaseModel(abc.ABC):
    """
    Base class for all models.

    The model is run in 8 steps:
    1. Parse the configuration file.
    2. Initialize the Dask client/cluster.
    3. Generate the configuration.
    4. Run the pre-production process.
    (4.bis Save the configuration.)
    5. Run the production process.
    (5.bis Save the configuration.)
    6. Run the biomass process.
    7. Save the outputs.
    8. Close the client.
    """

    @property
    @abc.abstractmethod
    def configuration(self: BaseModel) -> object:
        """The structure that store the model parameters."""

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
    def save_state(self: BaseModel) -> None:
        """
        Save the state of the model.

        State:
        - The input forcing.
        - The parameters of the model.
        - The outputs of the model if they are computed.
        """

    @abc.abstractmethod
    def pre_production(self: BaseModel) -> None:
        """Run the pre-production process. Basicaly, it runs all the parallel functions to speed up the model."""

    @abc.abstractmethod
    def production(self: BaseModel) -> None:
        """Run the production process that is not explicitly parallel."""

    @abc.abstractmethod
    def post_production(self: BaseModel) -> None:
        """Run the post-production process. Mostly parallel but need the production to be computed."""

    @abc.abstractmethod
    def save_output(self: BaseModel) -> None:
        """Save the outputs of the model."""

    @abc.abstractmethod
    def close(self: BaseModel) -> None:
        """Clean up the system. For example, it can be used to close dask.Client."""
