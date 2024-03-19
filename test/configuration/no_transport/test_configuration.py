import numpy as np
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.configuration.no_transport import parameter_functional_group
from seapodym_lmtl_python.configuration.no_transport.configuration import NoTransportConfiguration
from seapodym_lmtl_python.configuration.no_transport.labels import ConfigurationLabels
from seapodym_lmtl_python.configuration.no_transport.parameter_environment import EnvironmentParameter
from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit
from seapodym_lmtl_python.configuration.no_transport.parameters import (
    ForcingParameters,
    FunctionalGroups,
    NoTransportParameters,
)


@pytest.fixture()
def forcing_param():
    time = coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=2))
    latitude = coordinates.new_latitude(np.array([0, 1, 2]))
    longitude = coordinates.new_longitude(np.array([0, 1, 2]))

    forcing = ForcingUnit(
        forcing=xr.DataArray(
            dims=("time", "latitude", "longitude"),
            coords={"time": time, "latitude": latitude, "longitude": longitude},
            data=np.full((time.size, latitude.size, longitude.size), 0, dtype=float),
        )
    )
    return ForcingParameters(temperature=forcing, primary_production=forcing)


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
        param = NoTransportParameters(functional_groups_parameters=fgroup_param, forcing_parameters=forcing_param)
        configuration = NoTransportConfiguration(parameters=param)
        assert configuration.parameters is not None
        assert isinstance(configuration.parameters, NoTransportParameters)

    def test_load_forcings(self, forcing_param, fgroup_param):
        param = NoTransportParameters(functional_groups_parameters=fgroup_param, forcing_parameters=forcing_param)
        configuration = NoTransportConfiguration(parameters=param)
        forcings = configuration._load_forcings()
        assert isinstance(forcings, xr.Dataset)
        assert ConfigurationLabels.timestep in forcings
        assert ConfigurationLabels.resolution_latitude in forcings
        assert ConfigurationLabels.resolution_longitude in forcings

    def test_build_fgroup_dataset(self, forcing_param, fgroup_param):
        param = NoTransportParameters(functional_groups_parameters=fgroup_param, forcing_parameters=forcing_param)
        configuration = NoTransportConfiguration(parameters=param)
        fgroup_dataset = configuration._build_fgroup_dataset()
        assert isinstance(fgroup_dataset, xr.Dataset)
        assert ConfigurationLabels.fgroup_name in fgroup_dataset
        assert ConfigurationLabels.energy_transfert in fgroup_dataset
        assert ConfigurationLabels.inv_lambda_max in fgroup_dataset
        assert ConfigurationLabels.inv_lambda_rate in fgroup_dataset
        assert ConfigurationLabels.temperature_recruitment_max in fgroup_dataset
        assert ConfigurationLabels.temperature_recruitment_rate in fgroup_dataset
        assert ConfigurationLabels.day_layer in fgroup_dataset
        assert ConfigurationLabels.night_layer in fgroup_dataset

    def test_build_cohort_dataset(self, forcing_param, fgroup_param):
        param = NoTransportParameters(functional_groups_parameters=fgroup_param, forcing_parameters=forcing_param)
        configuration = NoTransportConfiguration(parameters=param)
        names = xr.DataArray(
            dims=ConfigurationLabels.fgroup, coords={ConfigurationLabels.fgroup: [0]}, data=["phytoplankton"]
        )
        cohort_dataset = configuration._build_cohort_dataset(names)
        assert isinstance(cohort_dataset, xr.Dataset)
        assert ConfigurationLabels.timesteps_number in cohort_dataset
        assert ConfigurationLabels.min_timestep in cohort_dataset
        assert ConfigurationLabels.max_timestep in cohort_dataset
        assert ConfigurationLabels.mean_timestep in cohort_dataset

        assert np.all(cohort_dataset[ConfigurationLabels.timesteps_number] == [1, 2, 3, 3, 1])
        assert np.all(cohort_dataset[ConfigurationLabels.min_timestep] == [1, 2, 4, 7, 10])
        assert np.all(cohort_dataset[ConfigurationLabels.max_timestep] == [1, 3, 6, 9, 10])
        assert np.all(cohort_dataset[ConfigurationLabels.mean_timestep] == [1, 2.5, 5, 8, 10])

    def test_as_dataset(self, forcing_param, fgroup_param):
        param = NoTransportParameters(functional_groups_parameters=fgroup_param, forcing_parameters=forcing_param)
        configuration = NoTransportConfiguration(parameters=param)

        dataset = configuration.as_dataset()
        assert np.all([label in dataset for label in ConfigurationLabels])
