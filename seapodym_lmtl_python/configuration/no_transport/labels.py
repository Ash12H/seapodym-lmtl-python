"""Store all labels used in the No Transport model."""

from enum import StrEnum


class ConfigurationLabels(StrEnum):
    """A single place to store all labels as declared in parameters module."""

    # Functional group
    fgroup = "functional_group"  # Equivalent to name
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
    timestep = "timestep"
    resolution = "resolution"


class PreproductionLabels(StrEnum):
    """A single place to store all labels as declared in pre-production module."""

    # Pre-production
    mask_global = "mask"
    mask_by_fgroup = "mask_fgroup"
    day_length = "day_length"
    avg_temperature_by_fgroup = "average_temperature_by_fgroup"
    primary_production_by_fgroup = "primary_production_by_fgroup"
    min_temperature_by_cohort = "min_temperature_by_cohort"
    mask_temperature = "mask_temperature"
    cell_area = "cell_area"
    mortality_field = "mortality_field"
    # Parameters
    temperature = "temperature"
    primary_production = "primary_production"


class ProductionLabels(StrEnum):
    """A single place to store all labels as declared in production module."""

    recruited = "recruited"
    preproduction = "preproduction"


class PostproductionLabels(StrEnum):
    """A single place to store all labels as declared in post-production module."""

    biomass = "biomass"
