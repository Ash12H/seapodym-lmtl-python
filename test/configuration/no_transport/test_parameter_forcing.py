from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit


@pytest.fixture()
def right_time_and_space_forcing_file(tmp_path):
    time = coordinates.new_time(xr.cftime_range(start="2000-01-01", periods=3, freq="D"))
    latitude = coordinates.new_latitude([0, 1, 2])
    longitude = coordinates.new_longitude([0, 1, 2])
    name = "right_time_and_space_forcing"
    tmp_file = tmp_path / "forcing_file.nc"

    xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=1.0,
        name=name,
    ).to_netcdf(tmp_file)

    return (tmp_file, name)


@pytest.fixture()
def wrong_time_right_space_forcing_file(tmp_path):
    time = coordinates.new_time(
        [np.datetime64("2000-01-01", "ns"), np.datetime64("2000-01-02", "ns"), np.datetime64("2000-01-04", "ns")]
    )
    latitude = coordinates.new_latitude([0, 1, 2])
    longitude = coordinates.new_longitude([0, 1, 2])
    name = "wrong_time_right_space_forcing"
    tmp_file = tmp_path / "forcing_file.nc"

    xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=1.0,
        name=name,
    ).to_netcdf(tmp_file)

    return (tmp_file, name)


@pytest.fixture()
def wrong_space_right_time_forcing_file(tmp_path):
    time = coordinates.new_time(xr.cftime_range(start="2000-01-01", periods=3, freq="D"))
    latitude = coordinates.new_latitude([0, 1, 3, 4])
    longitude = coordinates.new_longitude([0, 1, 2, 4])
    name = "wrong_space_right_time_forcing"
    tmp_file = tmp_path / "forcing_file.nc"

    xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=1.0,
        name=name,
    ).to_netcdf(tmp_file)

    return (tmp_file, name)


class TestForcingUnit:
    def test_right_time_and_space_forcing(self, right_time_and_space_forcing_file: tuple[str, str]):
        forcing_path, forcing_name = right_time_and_space_forcing_file
        forcing_unit = ForcingUnit(forcing_path=forcing_path, name=forcing_name)
        assert forcing_unit.resolution == (1, 1)
        assert forcing_unit.timestep == 1

    def test_wrong_time_right_space_forcing(self, wrong_time_right_space_forcing_file: tuple[str, str]):
        forcing_path, forcing_name = wrong_time_right_space_forcing_file
        with pytest.raises(ValueError):
            ForcingUnit(forcing_path=forcing_path, name=forcing_name)

    def test_wrong_space_right_time_forcing(self, wrong_space_right_time_forcing_file: tuple[str, str]):
        forcing_path, forcing_name = wrong_space_right_time_forcing_file
        with pytest.raises(ValueError):
            ForcingUnit(forcing_path=forcing_path, name=forcing_name)
