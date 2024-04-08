from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from seapopym.configuration.parameters.parameter_forcing import ForcingUnit
from seapopym.standard import coordinates


def _right_time_and_space_forcing() -> xr.DataArray:
    time = coordinates.new_time(xr.cftime_range(start="2000-01-01", periods=3, freq="D"))
    latitude = coordinates.new_latitude([0, 1, 2])
    longitude = coordinates.new_longitude([0, 1, 2])
    name = "right_time_and_space_forcing"
    return xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=1.0,
        name=name,
    )


@pytest.fixture()
def right_time_and_space_forcing() -> xr.DataArray:
    return _right_time_and_space_forcing()


@pytest.fixture()
def right_time_and_space_forcing_dataset() -> xr.Dataset:
    forcing = _right_time_and_space_forcing()
    return (forcing.name, xr.Dataset({forcing.name: forcing}))


@pytest.fixture()
def right_time_and_space_forcing_file(tmp_path):
    forcing = _right_time_and_space_forcing()
    tmp_file = tmp_path / "forcing_file.nc"
    forcing.to_zarr(tmp_file)
    return (tmp_file, forcing.name)


@pytest.fixture()
def wrong_time_right_space_forcing() -> xr.DataArray:
    time = coordinates.new_time(
        [np.datetime64("2000-01-01", "ns"), np.datetime64("2000-01-02", "ns"), np.datetime64("2000-01-04", "ns")]
    )
    latitude = coordinates.new_latitude([0, 1, 2])
    longitude = coordinates.new_longitude([0, 1, 2])
    name = "wrong_time_right_space_forcing"

    return xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=1.0,
        name=name,
    )


@pytest.fixture()
def wrong_space_right_time_forcing() -> xr.DataArray:
    time = coordinates.new_time(xr.cftime_range(start="2000-01-01", periods=3, freq="D"))
    latitude = coordinates.new_latitude([0, 1, 3, 4])
    longitude = coordinates.new_longitude([0, 1, 2, 4])
    name = "wrong_space_right_time_forcing"

    return xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=1.0,
        name=name,
    )


class TestForcingUnit:
    def test_right_time_and_space_forcing(self, right_time_and_space_forcing: xr.DataArray):
        forcing_unit = ForcingUnit(forcing=right_time_and_space_forcing, resolution=1, timestep=1)
        assert forcing_unit.resolution == (1, 1)
        assert forcing_unit.timestep == 1

    def test_right_time_and_space_forcing_dataset(self, right_time_and_space_forcing_dataset: tuple[str, xr.Dataset]):
        name, forcing = right_time_and_space_forcing_dataset
        forcing_unit = ForcingUnit.from_dataset(forcing=forcing, name=name, resolution=1, timestep=1)
        assert forcing_unit.resolution == (1, 1)
        assert forcing_unit.timestep == 1

    def test_right_time_and_space_forcing_file(self, right_time_and_space_forcing_file: tuple[str, str]):
        forcing_path, forcing_name = right_time_and_space_forcing_file
        forcing_unit = ForcingUnit.from_path(forcing=forcing_path, name=forcing_name, resolution=1, timestep=1)
        assert forcing_unit.resolution == (1, 1)
        assert forcing_unit.timestep == 1

        forcing_unit = ForcingUnit.from_path(forcing=Path(forcing_path), name=forcing_name, resolution=1, timestep=1)
        assert forcing_unit.resolution == (1, 1)
        assert forcing_unit.timestep == 1

    def test_wrong_time_right_space_forcing(self, wrong_time_right_space_forcing: xr.DataArray):
        with pytest.raises(ValueError):
            ForcingUnit(forcing=wrong_time_right_space_forcing)

    def test_wrong_space_right_time_forcing(self, wrong_space_right_time_forcing: xr.DataArray):
        with pytest.raises(ValueError):
            ForcingUnit(forcing=wrong_space_right_time_forcing)
