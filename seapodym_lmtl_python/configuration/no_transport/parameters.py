"""
This class is used to store the model configuration parameters. It uses the attrs library to define the class
attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import field, frozen

if TYPE_CHECKING:
    from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit
    from seapodym_lmtl_python.configuration.no_transport.parameter_functional_group import FunctionalGroupUnit


@frozen(kw_only=True)
class ForcingParameters:
    """
    This data class is used to store access paths to forcing fields. You can inherit it to add further forcings, but in
    this case you'll need to add new behaviors to the functions and classes that follow.
    """

    temperature: ForcingUnit = field(
        metadata={"description": "Path to the temperature field."},
    )
    primary_production: ForcingUnit = field(
        metadata={"description": "Path to the primary production field."},
    )
    mask: ForcingUnit | None = field(
        default=None,
        metadata={"description": "Path to the mask field."},
    )
    day_length: ForcingUnit | None = field(
        default=None,
        metadata={"description": "Path to the day length field."},
    )
    cell_area: ForcingUnit | None = field(
        default=None,
        metadata={"description": "Path to the cell area field."},
    )


@frozen(kw_only=True)
class FunctionalGroups:
    """This data class is used to store the parameters of all functional groups."""

    functional_groups: list[FunctionalGroupUnit] = field(metadata={"description": "List of all functional groups."})


@frozen(kw_only=True)
class NoTransportParameters:
    """This is the main data class. It is used to store the model configuration parameters."""

    path_parameters: ForcingParameters = field(metadata={"description": "All the paths to the forcings."})

    functional_groups_parameters: FunctionalGroups = field(
        metadata={"description": "Parameters of all functional groups."}
    )

    timestep: int | None = field(
        init=False,
        default=None,
        metadata={"description": "Common timestep of the fields in day(s)."},
    )

    resolution: float | tuple[float, float] | None = field(
        default=None,
        metadata={"description": "Common space resolution of the fields as (lat, lon) or both if equals."},
    )

    # TODO(Jules): Finish this work

    # def __attrs_post_init__(self: NoTransportParameters) -> None:
    #     timedelta = [
    #         forcing.timestep
    #         for forcing in asdict(self.path_parameters).values()
    #         if forcing is not None and forcing.timestep is not None
    #     ]

    #     for forcing in ["temperature", "primary_production", "mask", "day_length", "cell_area"]:
    #         data = getattr(self, forcing)

    #     if len(set(timedelta)) != 1:
    #         msg = (
    #             f"The time axis is not regular. Differents values of timedelta are found: {timedelta}."
    #             "\nCalendars will be interpolated in the rest of the simulation."
    #         )
    #         logger.warning(msg)
    #         object.__setattr__(self, "timestep", min(timedelta))
    #     else:
    #         object.__setattr__(self, "timestep", timedelta[0])

    #     # TODO(Jules): Do the same for space resolution

    # # TODO(Jules): Validate timestep (ie. min_timestep) is multiple of cohort_timesteps
