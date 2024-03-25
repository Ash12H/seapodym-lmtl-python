"""Store all labels used in the No Transport model."""
from __future__ import annotations

from enum import Enum, StrEnum
from typing import Literal


class CoordinatesLabels(StrEnum):
    """A single place to store all labels as declared in coordinates module. It follow the cf_xarray convention."""

    functional_group = "functional_group"
    time = "T"
    Y = "Y"
    X = "X"
    Z = "Z"
    cohort = "cohort"

    @classmethod
    def ordered(cls: CoordinatesLabels) -> list[CoordinatesLabels]:
        """Return all labels in the order they should be used in a dataset. It follow the CF convention."""
        return [cls.functional_group, cls.time, cls.Y, cls.X, cls.Z, cls.cohort]


class SeaLayers(Enum):
    """Enumerate the sea layers."""

    # NOTE(Jules): The following order of the layers declaration is important.
    ## Since python 3.4 this order is preserved.
    EPI = ("epipelagic", 1)
    UPMESO = ("upper-mesopelagic", 2)
    LOWMESO = ("lower-mesopelagic", 3)

    @property
    def standard_name(
        self: SeaLayers,
    ) -> Literal["epipelagic", "upper-mesopelagic", "lower-mesopelagic"]:
        """Return the standard_name of the sea layer."""
        return self.value[0]

    @property
    def depth(self: SeaLayers) -> Literal[1, 2, 3]:
        """Return the depth of the sea layer."""
        return self.value[1]


class ConfigurationLabels(StrEnum):
    """A single place to store all labels as declared in parameters module."""

    # Functional group
    fgroup_name = "name"
    energy_transfert = "energy_transfert"
    inv_lambda_max = "inv_lambda_max"
    inv_lambda_rate = "inv_lambda_rate"
    temperature_recruitment_max = "temperature_recruitment_max"
    temperature_recruitment_rate = "temperature_recruitment_rate"
    day_layer = "day_layer"
    night_layer = "night_layer"
    # Cohorts
    cohort = "cohort"  # New axis
    timesteps_number = "timesteps_number"
    min_timestep = "min_timestep"
    max_timestep = "max_timestep"
    mean_timestep = "mean_timestep"
    # Forcing
    timestep = "timestep"
    resolution_latitude = "resolution_latitude"
    resolution_longitude = "resolution_longitude"
    initial_condition_production = "initial_condition_production"
    initial_condition_biomass = "initial_condition_biomass"
    temperature = "temperature"
    primary_production = "primary_production"


class PreproductionLabels(StrEnum):
    """A single place to store all labels as declared in pre-production module."""

    mask_global = "mask"
    mask_by_fgroup = "mask_fgroup"
    day_length = "day_length"
    avg_temperature_by_fgroup = "average_temperature_by_fgroup"
    primary_production_by_fgroup = "primary_production_by_fgroup"
    min_temperature_by_cohort = "min_temperature_by_cohort"
    mask_temperature = "mask_temperature"
    cell_area = "cell_area"
    mortality_field = "mortality_field"


class ProductionLabels(StrEnum):
    """A single place to store all labels as declared in production module."""

    recruited = "recruited"
    preproduction = "preproduction"


class PostproductionLabels(StrEnum):
    """A single place to store all labels as declared in post-production module."""

    biomass = "biomass"
