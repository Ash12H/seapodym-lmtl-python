from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from attrs import define, field, validators
from dask.distributed import Client, LocalCluster

from seapodym_lmtl_python.logging.custom_logger import logger


@define
class ChunkParameter:
    """The chunk size of the different dimensions."""

    functional_group: str | int = field(
        default="auto",
        metadata={"description": "The chunk size of the functional group dimension."},
    )
    latitude: str | int | None = field(
        default=None,
        metadata={"description": "The chunk size of the latitude dimension."},
    )
    longitude: str | int | None = field(
        default=None,
        metadata={"description": "The chunk size of the longitude dimension."},
    )
    time: None = field(
        init=False,
        default=None,
        metadata={
            "description": (
                "The chunk size of the time dimension. "
                "Present only to remind us that time is not divisible due to time dependencies."
            )
        },
    )

    def as_dict(self: ChunkParameter, *, with_fgroup: bool = False) -> dict:
        """Format to a dictionary as expected by xarray."""
        chunks = {}
        if self.latitude is not None:
            chunks["latitude"] = self.latitude
        if self.longitude is not None:
            chunks["longitude"] = self.longitude
        if with_fgroup:
            chunks["functional_group"] = self.functional_group
        return chunks


@define
class ClientParameter:
    """
    If an address is provided, the client will be initialized with this address and other parameters will be ignored.
    If no address is provided, a LocalCluster will be initialized with the other parameters.

    For more information about this class check the Dask documentation about LocalCluster and Client.
    """

    n_workers: int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        metadata={"description": "The number of workers."},
    )
    threads_per_worker: int | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        metadata={"description": "The number of threads per worker."},
    )
    memory_limit: str | int | float = field(
        default="auto", metadata={"description": "The memory limit of each worker."}
    )
    client: Client | None = field(init=False, default=None, metadata={"description": "The Dask client."})

    @classmethod
    def from_address(cls: ClientParameter, address: str) -> ClientParameter:
        """
        Create a ClientParameter from an address.

        Example:
        -------
        ```python
        client = Client()
        client_param = ClientParameter.from_address(address=client.scheduler.address)
        print(client_param.client)
        ```

        """
        client = Client(address=address)
        workers = client.scheduler_info()["workers"]
        infos = ((w_info["nthreads"], w_info["memory_limit"]) for w_info in workers.values())
        nthreads, memory_limit = tuple(map(list, zip(*infos)))

        n_workers = int(len(nthreads))
        nthreads = int(np.mean(nthreads))
        memory_limit = int(np.mean(memory_limit))

        client_param = ClientParameter(n_workers=n_workers, threads_per_worker=nthreads, memory_limit=memory_limit)
        client_param.client = client
        return client_param

    def initialize_client(self: ClientParameter) -> None:
        """Initialize the client."""
        if self.client is None:
            self.client = Client(
                n_workers=self.n_workers,
                threads_per_worker=self.threads_per_worker,
                memory_limit=self.memory_limit,
            )
        else:
            msg = "Trying to initialize an already initialized client."
            logger.info(msg)

    def close_client(self: ClientParameter) -> None:
        """Close the client."""
        if self.client is not None:
            self.client.close()
            del self.client
            self.client = None


@define
class BaseOuputForcingParameter:
    """A base class for the output forcing parameter."""

    path: str | Path = field(
        default=Path("./output.nc"),
        converter=Path,
        metadata={"description": "The path where the forcing will be stored."},
    )
    with_parameter: bool = field(
        default=True,
        validator=validators.instance_of(bool),
        metadata={"description": "If True, the forcing will be computed with the parameter forcing."},
    )
    with_forcing: bool = field(
        default=False,
        validator=validators.instance_of(bool),
        metadata={"description": "If True, forcing will be added to the output dataset."},
    )


@define
class BiomassParameter(BaseOuputForcingParameter):
    """The output parameter for the biomass forcing."""


@define
class ProductionParameter(BaseOuputForcingParameter):
    """The output parameter for the production forcing."""


@define
class PreProductionParameter(BaseOuputForcingParameter):
    """The output parameter for the pre-production forcing (i.e. with cohorts)."""

    timestamps: Iterable[str] | Iterable[int] = field(
        default=[-1],
        metadata={"description": "The timestamps for the pre-production forcing."},
    )


@define
class OutputParameter:
    """The output parameter that manage the backup of the output forcings."""

    biomass: BiomassParameter | None = field(
        factory=BiomassParameter,
        validator=validators.instance_of(BiomassParameter),
        metadata={"description": "The output parameter for the biomass forcing."},
    )
    production: ProductionParameter | None = field(
        factory=ProductionParameter,
        validator=validators.instance_of(ProductionParameter),
        metadata={"description": "The output parameter for the production forcing."},
    )
    pre_production: PreProductionParameter | None = field(
        factory=PreProductionParameter,
        validator=validators.instance_of(PreProductionParameter),
        metadata={"description": "The output parameter for the pre-production forcing."},
    )

    def shared_path(self: OutputParameter) -> bool:
        """Check if the path are shared between the different output forcing."""
        return self.biomass.path == self.production.path and self.production.path == self.pre_production.path


@define
class EnvironmentParameter:
    """Manage the different environment parameters of the simulation."""

    chunk: ChunkParameter | None = field(
        factory=ChunkParameter,
        validator=validators.instance_of(ChunkParameter),
        metadata={"description": "The chunk size of the different dimensions."},
    )
    client: ClientParameter | None = field(
        factory=ClientParameter,
        validator=validators.instance_of(ClientParameter),
        metadata={"description": "The client parameter."},
    )
    output: OutputParameter | None = field(
        factory=OutputParameter,
        validator=validators.instance_of(OutputParameter),
        metadata={"description": "The output parameter."},
    )
