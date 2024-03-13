"""Define the ForcingUnit data class used to store access paths to a forcing field."""

from __future__ import annotations

from pathlib import Path

import cf_xarray  # noqa: F401
import numpy as np
import pandas as pd
import xarray as xr
from attrs import Attribute, field, frozen

from seapodym_lmtl_python.logging.custom_logger import logger

DECIMALS = 5  # ie. 1e-5 degrees which is equivalent to ~1m at the equator


def _check_single_forcing_resolution(latitude: xr.DataArray, longitude: xr.DataArray) -> float | tuple[float, float]:
    """Helper function used by ForcingUnit to check the resolution of a single forcing."""
    decimals = 5
    lat_resolution = np.unique(np.round(latitude[1:].data - latitude[:-1].data, decimals=decimals))
    lon_resolution = np.unique(np.round(longitude[1:].data - longitude[:-1].data, decimals=decimals))
    if lat_resolution.size > 1 or lon_resolution.size > 1:
        msg = (
            f"The resolution of the forcing is not constant (rounded to E-{decimals})."
            f"\n\tLatitude resolution: {lat_resolution}\n\tLongitude resolution: {lon_resolution}"
            "\nConsider setting the resolution manually in the configuration."
        )
        raise ValueError(msg)

    lat_resolution = lat_resolution.item()
    lon_resolution = lon_resolution.item()

    if lat_resolution != lon_resolution:
        msg = f"The latitude resolution ({lat_resolution}) of the forcing is not the same as longitude ({lon_resolution})."
        logger.info(msg)
    return (lat_resolution, lon_resolution)


def _check_single_forcing_timestep(timeseries: pd.DatetimeIndex) -> float | tuple[float, float]:
    """Helper function used by ForcingUnit to check the resolution of a single forcing."""
    timedelta = pd.TimedeltaIndex(timeseries[1:] - timeseries[:-1]).days.unique()
    if len(timedelta) != 1:
        msg = (
            f"The time axis is not regular. Differents values of timedelta are found: {timedelta}."
            "\nConsider to use the cftime library to handle special calendar or xarray extrapolation."
        )
        raise ValueError(msg)
    return timedelta[0]


def path_validation(path: Path) -> None:
    """Check if the path exists and if the Dataset contains the specfied DataArray."""
    if not path.exists():
        msg = f"The path '{path}' does not exist."
        raise FileExistsError(msg)


def name_isin_forcing(forcing: xr.Dataset, name: str) -> None:
    """Check if the name exists in the forcing Dataset."""
    if name not in forcing:
        message = f"DataArray {name} is not in the Dataset.\nAccepted values are : {", ".join(list(forcing))}"
        raise ValueError(message)


# TODO(Jules): Allow to directly pass a xarray.Dataset


@frozen(kw_only=True)
class ForcingUnit:
    """
    TODO(Jules): Refactor.
    This data class is used to store access paths to a forcing field (read with xarray.open_dataset).

    Parameters
    ----------
    forcing : Path
        Path to the forcing.
    name : str
        Name of the field in the forcing file.
    resolution : float | tuple[float, float]
        Space resolution of the field as (lat, lon) or both if equals.
    timestep : int
        Timestep of the field in day(s).


    Notes
    -----
    - This class is used to store access paths to a forcing field (read with xarray.open_dataset). It also stores the
    resolution and timestep of the field. If not provided, the resolution and timestep are automatically computed from
    the forcing file. However, they can be set manually.
    - Be sure to follow the CF conventions for the forcing file. To do so you can use the `cf_xarray` package.

    """

    forcing: xr.DataArray = field(
        converter=xr.DataArray,
        metadata={"description": "Forcing field."},
    )

    # name: str = field(
    #     converter=str,
    #     validator=name_isin_forcing,
    #     metadata={"description": "Name of the field in the forcing file."},
    # )

    resolution: tuple[float, float] = field(
        default=None,
        metadata={"description": "Space resolution of the field as (lat, lon)."},
    )

    timestep: int = field(default=None, metadata={"description": "Timestep of the field in day(s)."})

    # NOTE(Jules):  For resolution and timestep, `default=None` because these attributes are automatically computed from
    #               the forcing file. However, they can be set manually.

    @classmethod
    def from_dataset(
        cls: ForcingUnit,
        forcing: xr.Dataset,
        name: str,
        resolution: tuple[float, float] | float | None,
        timestep: int | None,
    ) -> ForcingUnit:
        """Create a ForcingUnit from a path and a name."""
        name_isin_forcing(forcing, name)
        forcing = forcing[name]

        if resolution is not None:
            resolution = float(resolution)
        if isinstance(resolution, float):
            resolution = (resolution, resolution)
        if timestep is not None:
            timestep = int(timestep)
        return cls(forcing=forcing, resolution=resolution, timestep=timestep)

    @classmethod
    def from_path(
        cls: ForcingUnit,
        forcing: Path | str,
        name: str,
        resolution: tuple[float, float] | float | None,
        timestep: int | None,
    ) -> ForcingUnit:
        """Create a ForcingUnit from a path and a name."""
        forcing = Path(forcing)
        path_validation(forcing)
        return cls.from_dataset(xr.open_dataset(forcing), name, resolution, timestep)

    def __attrs_post_init__(self: ForcingUnit) -> None:
        """Setup the space and time resolutions."""

        if "X" in self.forcing.cf and "Y" in self.forcing.cf:
            resolution = _check_single_forcing_resolution(latitude=self.forcing.cf["Y"], longitude=self.forcing.cf["X"])
            object.__setattr__(self, "resolution", resolution)

        if "T" in self.forcing.cf:
            timestep = _check_single_forcing_timestep(timeseries=self.forcing.cf.indexes["T"])
            object.__setattr__(self, "timestep", timestep)
