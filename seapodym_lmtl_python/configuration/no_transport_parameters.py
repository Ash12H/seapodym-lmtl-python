"""
This class is used to store the model configuration parameters. It uses the attrs library to define the class
attributes.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr
from attrs import Attribute, field, frozen, validators


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
            validators.gt(0),
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
