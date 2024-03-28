"""Store all default attributs for the seapopym xarray.DataArray."""
from __future__ import annotations

from typing import Iterable

from seapopym.standard.labels import CoordinatesLabels
from seapopym.standard.units import StandardUnitsLabels


# TODO(Jules): Do the same for cohort axis
def functional_group_desc(f_group_coord_data: Iterable, groups_name: list[str]) -> dict:
    """Functional group attributs. Standard name is used as an accessor by cf_xarray."""
    return {
        "flag_values": f_group_coord_data,
        "flag_meanings": " ".join(groups_name),
        "standard_name": CoordinatesLabels.functional_group,
        "long_name": "functional group",
    }


"""dict: Functional group attributs."""

global_mask_desc = {
    "standard_name": "mask",
    "long_name": "mask",
    "flag_values": [0, 1],
    "flag_meanings": "0:land, 1:ocean",
}
"""dict: Global mask attributs."""

mask_by_fgroup_desc = global_mask_desc.copy()
"""dict: Mask by fgroup attributs."""


def day_length_desc(angle_horizon_sun: int = 0) -> dict[str, str]:
    """Day length attributs."""
    return {
        "long_name": "Day length",
        "standard_name": "day_length",
        "description": f"Day length at the surface using Forsythe's method with p={angle_horizon_sun}",
        "units": "day",
    }


average_temperature_by_fgroup_desc = {
    "long_name": "average sea temperature by fonctional group",
    "standard_name": "sea water temperature",
    "description": ("Average temperature by functional group according to their layer position during day and night."),
    "units": str(StandardUnitsLabels.temperature.units),
}

apply_coefficient_to_primary_production_desc = {
    "standard_name": "primary production",
    "long_name": "primary production by functional group",
    "description": "Primary production by functional group according to their energy transfert coefficient.",
    "units": str(StandardUnitsLabels.production.units),
}
"""dict: Apply coefficient to primary production attributs."""

min_temperature_by_cohort_desc = {
    "standard_name": "minimum temperature",
    "long_name": "minimum temperature by cohort",
    "description": "Minimum temperature to recruit a cohort according to its age.",
    "units": str(StandardUnitsLabels.temperature.units),
}
"""dict: Minimum temperature by cohort attributs."""

mask_temperature_desc = {
    "standard_name": "mask",
    "long_name": "cohort recruitment mask by functional group",
    "description": "Mask to recruit a cohort according to the temperature.",
    "flag_values": [0, 1],
    "flag_meanings": "0:not recruited, 1:recruited",
}
"""dict: Mask temperature attributs."""

compute_cell_area_desc = {
    "standard_name": "cell_area",
    "long_name": "cell area",
    "description": "Cell area computed from the latitude and longitude centroid.",
    "units": str(StandardUnitsLabels.height.units**2),
}
"""dict: Compute cell area attributs."""

mortality_field_desc = {
    "standard_name": "mortality",
    "long_name": "mortality coefficient",
    "description": "Mortality coefficient according to the temperature.",
}
"""dict: Mortality field attributs."""

recruited_desc = {
    "standard_name": "production",
    "long_name": "production",
    "units": str(StandardUnitsLabels.production.units),
}

preproduction_desc = {
    "standard_name": "preproduction",
    "long_name": "pre-production",
    "description": "The entire population before recruitment, divided into cohorts.",
    "units": str(StandardUnitsLabels.production.units),
}

biomass_desc = {
    "long_name": "biomass",
    "units": str(StandardUnitsLabels.biomass.units),
    "description": "The biomass of the recruited individuals.",
}
