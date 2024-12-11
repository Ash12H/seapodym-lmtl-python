"""Define the ForcingUnit data class used to store access paths to a forcing field."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, ParamSpecArgs, ParamSpecKwargs

import cf_xarray  # noqa: F401
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from attrs import converters, field, frozen

from seapopym.logging.custom_logger import logger
from seapopym.standard.units import check_units

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
        msg = (
            f"The latitude resolution ({lat_resolution}) of the forcing is not the same as longitude "
            f"({lon_resolution})."
        )
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


def path_validation(path: str | Path) -> str | Path:
    """Check if the path exists."""
    with fsspec.open(str(path)) as file:
        if "file" not in file.fs.protocol:
            logger.info(f"Remote file : {file.fs.protocol}")
            return str(path)
        if "file" in file.fs.protocol and Path(path).exists():
            # logger.debug(f"Local file : ({file.fs.protocol})")
            return Path(path)
    msg = f"Cannot reach '{path}'."
    raise FileNotFoundError(msg)


def name_isin_forcing(forcing: xr.Dataset, name: str) -> None:
    """Check if the name exists in the forcing Dataset."""
    if name not in forcing:
        message = f"DataArray {name} is not in the Dataset.\nAccepted values are : {", ".join(list(forcing))}"
        raise ValueError(message)


@frozen(kw_only=True)
class ForcingUnit:
    """
    This data class is used to store a forcing field and its resolution and timestep.

    Parameters
    ----------
    forcing: xr.DataArray
        Forcing field.
    resolution: tuple[float, float] | None
        Space resolution of the field as (lat, lon).
    timestep: int | None
        Timestep of the field in day(s).


    Notes
    -----
    - This class is used to store a forcing field. It also stores the resolution and timestep of the field. If not
    provided, the resolution and timestep are automatically computed from the forcing file. However, they can be set
    manually.
    - Be sure to follow the CF conventions for the forcing file. To do so you can use the `cf_xarray` package.

    """

    forcing: xr.DataArray = field(
        converter=xr.DataArray,
        metadata={"description": "Forcing field."},
    )

    resolution: Iterable[float, float] | float | None = field(
        default=None,
        converter=converters.optional(
            lambda x: tuple(float(item) for item in x) if isinstance(x, Iterable) else float(x)
        ),
        metadata={"description": "Space resolution of the field as (lat, lon)."},
    )

    timestep: int | None = field(
        default=None,
        converter=converters.optional(int),
        metadata={"description": "Timestep of the field in day(s)."},
    )

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
            if isinstance(resolution, Iterable):
                resolution = tuple(float(item) for item in resolution)
            else:
                resolution = (float(resolution), float(resolution))

        if timestep is not None:
            timestep = int(timestep)
        return cls(forcing=forcing, resolution=resolution, timestep=timestep)

    @classmethod
    def from_path(
        cls: ForcingUnit,
        forcing: Path | str,
        name: str,
        resolution: tuple[float, float] | float | None = None,
        timestep: int | None = None,
        engine: Literal["zarr", "netcdf"] = "zarr",
        *args: ParamSpecArgs,
        **kwargs: ParamSpecKwargs,
    ) -> ForcingUnit:
        """Create a ForcingUnit from a path and a name."""
        path_validation(forcing)
        data = xr.open_dataset(forcing, *args, engine=engine, **kwargs)
        # data = xr.open_zarr(forcing) if engine == "zarr" else xr.open_dataset(forcing)
        return cls.from_dataset(data, name, resolution, timestep)

    def __attrs_post_init__(self: ForcingUnit) -> None:
        """Setup the space and time resolutions."""
        if self.resolution is not None:
            if not isinstance(self.resolution, Iterable):
                object.__setattr__(self, "resolution", (float(self.resolution), float(self.resolution)))
        elif ("X" in self.forcing.cf.indexes) and ("Y" in self.forcing.cf.indexes):
            resolution = _check_single_forcing_resolution(latitude=self.forcing.cf["Y"], longitude=self.forcing.cf["X"])
            object.__setattr__(self, "resolution", resolution)

        if (self.timestep is None) and ("T" in self.forcing.cf.indexes):
            data = self.forcing.cf.dropna("T")
            timestep = _check_single_forcing_timestep(timeseries=data.cf.indexes["T"])
            object.__setattr__(self, "timestep", timestep)

    def with_units(self: ForcingUnit, units: str, *, in_place: bool = False) -> ForcingUnit:
        """Ensure that the forcing has the correct units. It is both a validator and a converter."""
        if in_place:
            object.__setattr__(self, "forcing", check_units(self.forcing, units))
            return self
        return ForcingUnit(forcing=check_units(self.forcing, units), resolution=self.resolution, timestep=self.timestep)
