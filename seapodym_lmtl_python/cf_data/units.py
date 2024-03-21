"""A module for handling units in the forcing data following the CF conventions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import pint
import pint_xarray  # noqa: F401

from seapodym_lmtl_python.logging.custom_logger import logger

if TYPE_CHECKING:
    import xarray as xr


def check_units(forcing: xr.DataArray, units: pint.Unit | pint.Quantity) -> xr.DataArray:
    """Check the units of the forcing and convert it to the target units if necessary."""
    if isinstance(units, pint.Quantity):
        units = units.units
    forcing = forcing.pint.quantify()
    if forcing.pint.units != units:
        logger.warning(f"{forcing.name} unit is {forcing.pint.units}, it will be converted to {units}.")
    try:
        forcing = forcing.pint.to(units)
    except Exception as e:
        logger.error(f"Failed to convert forcing to {units}. forcing is in {forcing.pint.units}.")
        raise type(e) from e
    return forcing.pint.dequantify()
