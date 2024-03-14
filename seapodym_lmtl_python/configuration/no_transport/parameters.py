"""
This class is used to store the model configuration parameters. It uses the attrs library to define the class
attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import field, frozen

from seapodym_lmtl_python.logging.custom_logger import logger

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

    timestep: int | None = field(
        init=False,
        default=None,
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
        forcings = [self.temperature, self.primary_production, self.mask, self.day_length, self.cell_area]
        forcings = [field for field in forcings if field is not None]
        # timestep
        timesteps = {forcing.timestep for forcing in forcings}
        if len(timesteps) != 1:
            msg = f"The forcings have different timesteps : {timesteps}."
            raise ValueError(msg)
        object.__setattr__(self, "timestep", timesteps.pop())
        # resolution
        resolutions = {(forcing.resolution[0], forcing.resolution[1]) for forcing in forcings}
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
        object.__setattr__(self, "resolution", (min_lat, min_lon))


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

    # TODO(Jules): Check for cohorts timesteps are multiple of the forcing timestep.
