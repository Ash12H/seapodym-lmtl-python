"""A module for handling units in the forcing data following the CF conventions."""
from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

# ------------------------------------------------------------------------------------------------------- #
# NOTE(Jules): cf_xarray.units import is necessary to work with cf_xarray complient Datasets. DO NOT REMOVE
import cf_xarray.units  # noqa: F401

# ------------------------------------------------------------------------------------------------------- #
import pint
import pint_xarray  # noqa: F401

from seapopym.logging.custom_logger import logger

if TYPE_CHECKING:
    import xarray as xr


class StandardUnitsLabels(StrEnum):
    """Unit of measurement as used in the model."""

    height = "meter"
    weight = "kilogram"
    temperature = "celsius"
    time = "day"
    biomass = "kilogram / meter**2"
    production = "kilogram / meter**2 / day"

    def __init__(self: StandardUnitsLabels, unit_as_str: str) -> None:
        """Prevent the instantiation of this class."""
        self._units = pint.application_registry(unit_as_str).units

    @property
    def units(self: StandardUnitsLabels) -> pint.Unit:
        """Convert the string unit to the equivalent pint unit."""
        return self._units


def check_units(forcing: xr.DataArray, units: pint.Unit | pint.Quantity | str) -> xr.DataArray:
    """Check the units of the forcing and convert it to the target units if necessary."""
    if isinstance(units, str):
        units = pint.Unit(units)
    elif isinstance(units, pint.Quantity):
        units = units.units
    elif not isinstance(units, pint.Unit):
        msg = f"units must be a pint.Unit, pint.Quantity or str, not {type(units)}."
        raise TypeError(msg)

    forcing = forcing.pint.quantify()
    if forcing.pint.units != units:
        logger.warning(f"{forcing.name} unit is {forcing.pint.units}, it will be converted to {units}.")
    try:
        forcing = forcing.pint.to(units)
    except Exception as e:
        logger.error(f"Failed to convert forcing to {units}. forcing is in {forcing.pint.units}.")
        raise type(e) from e
    return forcing.pint.dequantify()
