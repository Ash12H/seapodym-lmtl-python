import numpy as np
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit
from seapodym_lmtl_python.configuration.no_transport.parameters import ForcingParameters
from seapodym_lmtl_python.exception.parameter_exception import DifferentForcingTimestepError

time_1 = coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=2))
time_2 = coordinates.new_time(xr.cftime_range(start="2020", freq="2D", periods=2))
latitude_1 = coordinates.new_latitude(np.array([0, 1, 2]))
longitude_1 = coordinates.new_longitude(np.array([0, 1, 2]))
latitude_2 = coordinates.new_latitude(np.array([0, 2, 4]))
longitude_2 = coordinates.new_longitude(np.array([0, 2, 4]))
layer = coordinates.new_layer()


@pytest.fixture()
def forcing_time_1_space_1():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_1, "latitude": latitude_1, "longitude": longitude_1},
            data=np.full((time_1.size, latitude_1.size, longitude_1.size), 0, dtype=float),
        )
    )


@pytest.fixture()
def forcing_time_2_space_1():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_2, "latitude": latitude_1, "longitude": longitude_1},
            data=np.full((time_2.size, latitude_1.size, longitude_1.size), 0, dtype=float),
        )
    )


@pytest.fixture()
def forcing_time_1_space_2():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time_1, "latitude": latitude_2, "longitude": longitude_2},
            data=np.full((time_1.size, latitude_2.size, longitude_2.size), 0, dtype=float),
        )
    )


class TestForcingParameters:
    def test_forcing_parameters_initialization(self, forcing_time_1_space_1, forcing_time_1_space_2):
        forcing_param = ForcingParameters(
            temperature=forcing_time_1_space_1,
            primary_production=forcing_time_1_space_1,
        )
        assert forcing_param.timestep == 1
        assert forcing_param.resolution == (1, 1)

        forcing_param = ForcingParameters(
            temperature=forcing_time_1_space_2,
            primary_production=forcing_time_1_space_2,
        )
        assert forcing_param.timestep == 1
        assert forcing_param.resolution == (2, 2)

    def test_forcing_parameters_initialization_with_different_space(
        self, forcing_time_1_space_1, forcing_time_1_space_2
    ):
        forcing_param = ForcingParameters(
            temperature=forcing_time_1_space_1,
            primary_production=forcing_time_1_space_2,
        )
        assert forcing_param.resolution == (1, 1)

    def test_forcing_parameters_initialization_with_different_time(
        self, forcing_time_1_space_1, forcing_time_2_space_1
    ):
        with pytest.raises(DifferentForcingTimestepError):
            ForcingParameters(
                temperature=forcing_time_1_space_1,
                primary_production=forcing_time_2_space_1,
            )
