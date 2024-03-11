"""
This class is used to store the model configuration parameters. It uses the attrs library to define the class
attributes.
"""

from __future__ import annotations

from pathlib import Path

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from attrs import Attribute, field, frozen, validators

from seapodym_lmtl_python.logging.custom_logger import logger


def _check_single_forcing_resolution(latitude: xr.DataArray, longitude: xr.DataArray) -> float | tuple[float, float]:
    """Helper function used by PathParametersUnit to check the resolution of a single forcing."""
    lat_resolution = np.asarray(set(np.round(latitude[1:].data - latitude[:-1].data, decimals=5)))
    lon_resolution = np.asarray(set(np.round(longitude[1:].data - longitude[:-1].data, decimals=5)))
    if lat_resolution.size > 1 or lon_resolution.size > 1:
        msg = (
            "The resolution of the forcing is not constant (rounded to E-5). Its value is set to the mean."
            "\nLatitude resolution: {lat_res}\nLongitude resolution: {lon_res}"
        )
        logger.warning(msg)
        lat_resolution = np.mean(lat_resolution)
        lon_resolution = np.mean(lon_resolution)
    if lat_resolution == lon_resolution:
        return lat_resolution
    msg = (
        f"The resolution of the forcing is not the same for latitude ({lat_resolution}) and "
        f"longitude ({lon_resolution})."
    )
    logger.info(msg)
    return (lat_resolution, lon_resolution)


@frozen(kw_only=True)
class PathParametersUnit:
    """This data class is used to store access paths to a forcing field (read with xarray.open_dataset)."""

    forcing_path: Path = field(
        converter=Path,
        metadata={"description": "Path to the temperature field."},
    )

    @forcing_path.validator
    def _path_exists(self: PathParameters, attribute: Attribute, value: Path) -> None:
        """Check if the path exists. If not, raise a ValueError."""
        if not value.exists():
            message = f"Parameter {attribute.name} : {value} does not exist."
            raise ValueError(message)

    name: str = field(
        converter=str,
    )

    @name.validator
    def name_isin_forcing(self: PathParametersUnit, attribute: Attribute, value: str) -> None:
        """Check if the name exists in the forcing file. If not, raise a ValueError."""
        if value not in xr.open_dataset(self.forcing_path):
            message = (
                f"Parameter {attribute.name} : {value} is not in the forcing file '{self.forcing_path}'."
                f"\nAccepted values are : {", ".join(list(xr.open_dataset(self.forcing_path)))}"
            )
            raise ValueError(message)

    resolution: float | tuple[float, float] | None = field(
        default=None,
        metadata={"description": "Space resolution of the field as (lat, lon) or both if equals."},
    )

    timestep: int | None = field(
        default=None,
        metadata={"description": "Timestep of the field in day(s)."},
    )

    def __attrs_post_init__(self: PathParametersUnit) -> None:
        """Compute  and timestep."""
        data = xr.open_dataset(self.forcing_path)[self.name]
        # Check space resolution consistency
        if "X" in data.cf and "Y" in data.cf:
            resolution = _check_single_forcing_resolution(latitude=data.cf["Y"], longitude=data.cf["X"])
            if self.resolution is not None and self.resolution != resolution:
                msg = (
                    f"The given resolution of the forcing is not the same as the computed one."
                    f"\nGiven: {self.resolution}, Computed: {resolution}."
                    f"\nGiven resolution will be used."
                )
                logger.warning(msg)
            else:
                object.__setattr__(self, "resolution", resolution)
        # Check time resolution consistency
        if "T" in data.cf:
            timedelta = pd.TimedeltaIndex(data.cf.indexes["T"][1:] - data.cf.indexes["T"][:-1]).days.unique()
            if len(timedelta) != 1:
                msg = (
                    f"The time axis is not regular. Differents values of timedelta are found: {timedelta}."
                    "\nConsider to use the cftime library to handle special calendar."
                )
                logger.error(msg)
                raise ValueError(msg)
            object.__setattr__(self, "timestep", timedelta[0])


@frozen(kw_only=True)
class PathParameters:
    """
    This data class is used to store access paths to forcing fields. You can inherit it to add further forcings, but in
    this case you'll need to add new behaviors to the functions and classes that follow.

    Example:
    -------
    ```
    @frozen(kw_only=True)
    class PathParametersOptional(PathParameters):
        landmask: Path = field(
            converter=Path,
            validator=[_path_exists],
            metadata={"description": "Path to the mask field."},
        )
    ```

    """

    temperature: PathParametersUnit = field(
        metadata={"description": "Path to the temperature field."},
    )
    primary_production: PathParametersUnit = field(
        metadata={"description": "Path to the primary production field."},
    )
    mask: PathParametersUnit | None = field(
        default=None,
        metadata={"description": "Path to the mask field."},
    )
    day_length: PathParametersUnit | None = field(
        default=None,
        metadata={"description": "Path to the day length field."},
    )
    cell_area: PathParametersUnit | None = field(
        default=None,
        metadata={"description": "Path to the cell area field."},
    )


@frozen(kw_only=True)
class FunctionalGroupUnitRelationParameters:
    """
    This data class is used to store the parameters linked to the relation between temperature and functional
    group.
    """

    inv_lambda_max: float = field(
        validator=[validators.gt(0)],
        converter=float,
        metadata={"description": "Value of 1/lambda when temperature is 0°C."},
    )
    inv_lambda_rate: float = field(
        validator=[
            validators.gt(0),
        ],
        converter=float,
        metadata={"description": "Rate of the inverse of the mortality."},
    )
    temperature_recruitment_rate: float = field(
        validator=[
            validators.lt(0),
        ],
        converter=float,
        metadata={"description": "Rate of the recruitment time."},
    )
    temperature_recruitment_max: float = field(
        validator=[
            validators.gt(0),
        ],
        converter=float,
        metadata={"description": "Maximum value of the recruitment time (temperature is 0°C).", "units": "day"},
    )
    cohorts_timesteps: list[int] | None = field(
        metadata={"description": "The number of timesteps in the cohort. Useful for cohorts aggregation."},
    )

    @cohorts_timesteps.validator
    def _cohorts_timesteps_equal_tr_max(
        self: FunctionalGroupUnitRelationParameters, attribute: Attribute, value: list[int]
    ) -> None:
        if np.sum(value) != np.ceil(self.temperature_recruitment_max):
            message = (
                f"Parameter {attribute.name} : {value} does not sum (= {np.sum(value)}) to the maximum recruitment "
                f"time {np.ceil(self.temperature_recruitment_max)}."
            )
            raise ValueError(message)

    def __attrs_post_init__(self: FunctionalGroupUnitRelationParameters) -> None:
        """Ensure that the last cohort contains a single timestep."""
        if self.cohorts_timesteps[-1] != 1:
            previous = np.copy(self.cohorts_timesteps)
            new = np.copy(previous)
            new[-1] = new[-1] - 1
            new = np.concatenate([new, [1]])
            object.__setattr__(self, "cohorts_timesteps", new)
            msg = (
                "The last cohort timesteps must be equal to 1. It has been modified to follow the standard behavior."
                f"\nPrevious :{previous}\nNew : {new}"
            )
            logger.warning(msg)


@frozen(kw_only=True)
class FunctionalGroupUnitMigratoryParameters:
    """This data class is used to store the parameters liked to the migratory behavior of a single functional group."""

    day_layer: int = field(
        validator=[validators.gt(0)],
        metadata={"description": "Layer position during day."},
    )
    night_layer: int = field(
        validator=[validators.gt(0)],
        metadata={"description": "Layer position during night."},
    )


@frozen(kw_only=True)
class FunctionalGroupUnit:
    """Represent a functional group."""

    name: str = field(metadata={"description": "The name of the function group."})
    energy_transfert: float = field(
        validator=[
            validators.ge(0),
            validators.le(1),
        ],
        converter=float,
        metadata={"description": "Energy transfert coefficient between primary production and functional group."},
    )
    functional_type: FunctionalGroupUnitRelationParameters = field(
        metadata={"description": "Parameters linked to the relation between temperature and the functional group."}
    )
    migratory_type: FunctionalGroupUnitMigratoryParameters = field(
        metadata={"description": "Parameters linked to the migratory behavior of the functional group."}
    )


@frozen(kw_only=True)
class FunctionalGroups:
    """This data class is used to store the parameters of all functional groups."""

    functional_groups: list[FunctionalGroupUnit] = field(metadata={"description": "List of all functional groups."})


@frozen(kw_only=True)
class NoTransportParameters:
    """This is the main data class. It is used to store the model configuration parameters."""

    path_parameters: PathParameters = field(metadata={"description": "All the paths to the forcings."})
    functional_groups_parameters: FunctionalGroups = field(
        metadata={"description": "Parameters of all functional groups."}
    )

    timestep: int | None = field(
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
