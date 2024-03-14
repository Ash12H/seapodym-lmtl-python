"""
This class is used to store the model configuration parameters. It uses the attrs library to define the class
attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from attrs import field, frozen, validators

from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit
from seapodym_lmtl_python.configuration.no_transport.parameter_functional_group import FunctionalGroupUnit
from seapodym_lmtl_python.exception.parameter_exception import (
    CohortTimestepConsistencyError,
    DifferentForcingTimestepError,
)
from seapodym_lmtl_python.logging.custom_logger import logger


@frozen(kw_only=True)
class ForcingParameters:
    """
    This data class is used to store access paths to forcing fields. You can inherit it to add further forcings, but in
    this case you'll need to add new behaviors to the functions and classes that follow.
    """

    temperature: ForcingUnit = field(
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the temperature field."},
    )
    primary_production: ForcingUnit = field(
        validator=validators.instance_of(ForcingUnit),
        metadata={"description": "Path to the primary production field."},
    )
    mask: ForcingUnit | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the mask field."},
    )
    day_length: ForcingUnit | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the day length field."},
    )
    cell_area: ForcingUnit | None = field(
        default=None,
        validator=validators.optional(validators.instance_of(ForcingUnit)),
        metadata={"description": "Path to the cell area field."},
    )

    timestep: int | None = field(
        init=False,
        default=None,
        validator=validators.optional(validators.instance_of(int)),
        metadata={"description": "Common timestep of the fields in day(s)."},
    )

    resolution: float | tuple[float, float] | None = field(
        init=False,
        default=None,
        metadata={"description": "Common space resolution of the fields as (lat, lon) or both if equals."},
    )

    def __attrs_post_init__(self: ForcingParameters) -> None:
        """
        This method is called after the initialization of the class. It is used to check the consistency of the
        forcing fields.
        """

        def _check_timestep(forcings: list[ForcingUnit]) -> int:
            timesteps = {field.timestep for field in forcings}
            if len(timesteps) != 1:
                as_dict = dict(zip([field.forcing.name for field in forcings], [field.timestep for field in forcings]))
                if len(as_dict) != len(timesteps):  # If there are duplicates in the forcing names or None values
                    timesteps = as_dict
                raise DifferentForcingTimestepError(timesteps)
            return timesteps.pop()

        def _check_resolution(forcings: list[ForcingUnit]) -> tuple[float, float]:
            resolutions = {(field.resolution[0], field.resolution[1]) for field in forcings}
            if len(resolutions) != 1:
                min_lat = min(lat for lat, _ in resolutions)
                min_lon = min(lon for _, lon in resolutions)
                msg = (
                    f"The forcings have different resolutions : {resolutions}."
                    f"\nBe aware that stranges behaviors may occur because minimum resolution is taken : "
                    f"{(min_lat,min_lon)}"
                    f"\nYou can extrapolate the fields to the same resolution using the xarray package."
                )
                logger.warning(msg)
            else:
                min_lat, min_lon = resolutions.pop()
            return (min_lat, min_lon)

        forcings = [self.temperature, self.primary_production, self.mask, self.day_length, self.cell_area]
        forcings = [field for field in forcings if field is not None]

        object.__setattr__(self, "timestep", _check_timestep(forcings))
        object.__setattr__(self, "resolution", _check_resolution(forcings))


@frozen(kw_only=True)
class FunctionalGroups:
    """This data class is used to store the parameters of all functional groups."""

    functional_groups: list[FunctionalGroupUnit] = field(metadata={"description": "List of all functional groups."})

    @functional_groups.validator
    def are_all_instance_of_functional_group_unit(
        self: FunctionalGroups, attribute: str, value: list[FunctionalGroupUnit]
    ) -> None:
        """This method is used to check the consistency of the functional groups."""
        if not all(isinstance(fgroup, FunctionalGroupUnit) for fgroup in value):
            msg = "All the functional groups must be instance of FunctionalGroupUnit."
            raise TypeError(msg)


@frozen(kw_only=True)
class NoTransportParameters:
    """This is the main data class. It is used to store the model configuration parameters."""

    path_parameters: ForcingParameters = field(metadata={"description": "All the paths to the forcings."})

    functional_groups_parameters: FunctionalGroups = field(
        metadata={"description": "Parameters of all functional groups."}
    )

    def __attrs_post_init__(self: NoTransportParameters) -> None:
        """
        Check that the timestep of the functional groups is consistent (ie. multiple of) with the timestep of the
        forcings.
        """
        global_timestep = self.path_parameters.timestep
        for fgroup in self.functional_groups_parameters.functional_groups:
            fgroup_timestep = fgroup.functional_type.cohorts_timesteps
            if not np.all([(ts % global_timestep) == 0 for ts in fgroup_timestep]):
                raise CohortTimestepConsistencyError(
                    cohort_name=fgroup.name, cohort_timesteps=fgroup_timestep, global_timestep=global_timestep
                )
