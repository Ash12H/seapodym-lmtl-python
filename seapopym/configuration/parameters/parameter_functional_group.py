"""This module define the data classes used to store the parameters of a functional group in the no transport model."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from attrs import Attribute, field, frozen, validators

from seapopym.exception.parameter_exception import TimestepInDaysError
from seapopym.logging.custom_logger import logger


@frozen(kw_only=True)
class FunctionalGroupUnitMigratoryParameters:
    """This data class is used to store the parameters liked to the migratory behavior of a single functional group."""

    day_layer: int = field(
        converter=int,
        metadata={"description": "Layer position during day."},
    )
    night_layer: int = field(
        converter=int,
        metadata={"description": "Layer position during night."},
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
    inv_lambda_rate: float = field(converter=float, metadata={"description": "Rate of the inverse of the mortality."})
    temperature_recruitment_rate: float = field(
        converter=float, metadata={"description": "Rate of the recruitment time."}
    )
    temperature_recruitment_max: float = field(
        validator=[validators.gt(0)],
        converter=float,
        metadata={"description": "Maximum value of the recruitment time (temperature is 0°C).", "units": "day"},
    )
    cohorts_timesteps: list[int] | None = field(
        metadata={"description": "The number of timesteps in the cohort. Useful for cohorts aggregation."},
    )

    @temperature_recruitment_rate.validator
    def _temperature_recruitment_rate_positive(
        self: FunctionalGroupUnitRelationParameters, attribute: Attribute, value: float
    ) -> None:
        if value > 0:
            message = (
                f"Parameter {attribute.name} : {value} has a positive value. It means that the recruitment time "
                "is increasing with temperature. Do you mean to use a negative value?"
            )
            logger.warning(message)
        if value == 0:
            message = (
                f"Parameter {attribute.name} : {value} has a null value. It means that the recruitment time "
                "is not affected by temperature. Do you really mean to use this value?"
            )
            logger.warning(message)

    @inv_lambda_rate.validator
    def _inv_lambda_rate_rate_positive(
        self: FunctionalGroupUnitRelationParameters, attribute: Attribute, value: float
    ) -> None:
        if value > 0:
            message = (
                f"Parameter {attribute.name} : {value} has a positive value. It means that the mortality is decreasing "
                "when temperature increase. Do you mean to use a negative value?"
            )
            logger.warning(message)
        if value == 0:
            message = (
                f"Parameter {attribute.name} : {value} has a null value. It means that the mortality is not affected "
                "by temperature. Do you really mean to use this value?"
            )
            logger.warning(message)

    @cohorts_timesteps.validator
    def _cohorts_timesteps_equal_tr_max(
        self: FunctionalGroupUnitRelationParameters, attribute: Attribute, value: Iterable[int]
    ) -> None:
        if not np.all(np.asarray(value) % 1 == 0):
            raise TimestepInDaysError(value)
        if np.sum(value) != np.ceil(self.temperature_recruitment_max):
            message = (
                f"Parameter {attribute.name} : {value} does not sum (= {np.sum(value)}) to the maximum recruitment "
                f"time {np.ceil(self.temperature_recruitment_max)} (ceiled value)."
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


# TODO : We should be able to fix some parameters easily
# Utiliser une matrice 2D avec des NONE pour les paramètres à opti et des valeurs pour les paramètres fixé.
# ensuite on rempli la matrice avec les valeurs de args en déroulant la matrice.


@frozen(kw_only=True)
class FunctionalGroupUnit:
    """Represent a functional group."""

    name: str = field(metadata={"description": "The name of the functional group."})

    energy_transfert: float = field(
        validator=[validators.ge(0), validators.le(1)],
        converter=float,
        metadata={"description": "Energy transfert coefficient between primary production and functional group."},
    )

    functional_type: FunctionalGroupUnitRelationParameters = field(
        validator=validators.instance_of(FunctionalGroupUnitRelationParameters),
        metadata={"description": "Parameters linked to the relation between temperature and the functional group."},
    )

    migratory_type: FunctionalGroupUnitMigratoryParameters = field(
        validator=validators.instance_of(FunctionalGroupUnitMigratoryParameters),
        metadata={"description": "Parameters linked to the migratory behavior of the functional group."},
    )
