from __future__ import annotations

from pathlib import Path
from typing import Iterable

from attrs import define, field, validators


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
    """For more information about this class check the Dask documentation about LocalCluster and Client."""

    adress: str | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
        metadata={"description": "The adress of the dask scheduler."},
    )
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
        default="auto",
        metadata={"description": "The memory limit of each worker."},
    )


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

    biomass: BiomassParameter = field(
        validator=validators.instance_of(BiomassParameter),
        metadata={"description": "The output parameter for the biomass forcing."},
    )
    production: ProductionParameter = field(
        validator=validators.instance_of(ProductionParameter),
        metadata={"description": "The output parameter for the production forcing."},
    )
    pre_production: PreProductionParameter = field(
        validator=validators.instance_of(PreProductionParameter),
        metadata={"description": "The output parameter for the pre-production forcing."},
    )

    def shared_path(self: OutputParameter) -> bool:
        """Check if the path are shared between the different output forcing."""
        return self.biomass.path == self.production.path and self.production.path == self.pre_production.path


@define
class EnvironmentParameter:
    """Manage the different environment parameters of the simulation."""

    chunk: ChunkParameter = field(
        factory=ChunkParameter,
        validator=validators.instance_of(ChunkParameter),
        metadata={"description": "The chunk size of the different dimensions."},
    )
    client: ClientParameter = field(
        factory=ClientParameter,
        validator=validators.instance_of(ClientParameter),
        metadata={"description": "The client parameter."},
    )
    output: OutputParameter = field(
        factory=OutputParameter,
        validator=validators.instance_of(OutputParameter),
        metadata={"description": "The output parameter."},
    )
