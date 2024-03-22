import numpy as np
import pytest
import xarray as xr

from seapodym_lmtl_python.configuration.no_transport import parameter_functional_group
from seapodym_lmtl_python.configuration.no_transport.configuration import NoTransportConfiguration
from seapodym_lmtl_python.configuration.no_transport.parameter import (
    ForcingParameters,
    FunctionalGroups,
    NoTransportParameters,
)
from seapodym_lmtl_python.configuration.no_transport.parameter_environment import EnvironmentParameter
from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit
from seapodym_lmtl_python.standard import coordinates
from seapodym_lmtl_python.standard.units import StandardUnitsLabels


@pytest.fixture()
def forcing_param():
    time = coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=2))
    latitude = coordinates.new_latitude(np.array([0, 1, 2]))
    longitude = coordinates.new_longitude(np.array([0, 1, 2]))

    temperature = xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=np.full((time.size, latitude.size, longitude.size), 0, dtype=float),
        attrs={"units": str(StandardUnitsLabels.temperature.units)},
        name="temperature",
    )
    primary_production = temperature.copy()
    primary_production.attrs["units"] = str(StandardUnitsLabels.production.units)
    primary_production.name = "primary_production"

    temperature = ForcingUnit(forcing=temperature)
    primary_production = ForcingUnit(forcing=primary_production)
    return ForcingParameters(temperature=temperature, primary_production=primary_production)


@pytest.fixture()
def fgroup_param():
    migratory_param = parameter_functional_group.FunctionalGroupUnitMigratoryParameters(day_layer=1, night_layer=1)
    functional_param = parameter_functional_group.FunctionalGroupUnitRelationParameters(
        cohorts_timesteps=[1, 2, 3, 3, 1],
        inv_lambda_max=10,
        inv_lambda_rate=0.5,
        temperature_recruitment_max=10,
        temperature_recruitment_rate=-0.5,
    )
    fgroup = parameter_functional_group.FunctionalGroupUnit(
        name="phytoplankton", energy_transfert=0.5, functional_type=functional_param, migratory_type=migratory_param
    )
    return FunctionalGroups(functional_groups=[fgroup])


class TestNoTransportConfiguration:
    def test_parameters(self, forcing_param, fgroup_param):
        param = NoTransportParameters(
            forcing_parameters=forcing_param,
            functional_groups_parameters=fgroup_param,
            environment_parameters=EnvironmentParameter(),
        )
        configuration = NoTransportConfiguration(parameters=param)

        m_param = configuration.model_parameters
        assert m_param is not None
        assert isinstance(m_param, xr.Dataset)

        e_param = configuration.environment_parameters
        assert e_param is not None
        assert isinstance(e_param, EnvironmentParameter)
