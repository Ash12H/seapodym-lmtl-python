import numpy as np
import pytest
import xarray as xr

from seapodym_lmtl_python.cf_data import coordinates
from seapodym_lmtl_python.configuration.no_transport import parameter_functional_group
from seapodym_lmtl_python.configuration.no_transport.configuration_to_dataset import (
    _as_dataset__build_cohort_dataset,
    _as_dataset__build_fgroup_dataset,
    _as_dataset__load_forcings,
    as_dataset,
)
from seapodym_lmtl_python.configuration.no_transport.labels import ConfigurationLabels, StandardUnitsLabels
from seapodym_lmtl_python.configuration.no_transport.parameter_forcing import ForcingUnit
from seapodym_lmtl_python.configuration.no_transport.parameters import (
    ForcingParameters,
    FunctionalGroups,
)

time = coordinates.new_time(xr.cftime_range(start="2020", freq="D", periods=2))
latitude = coordinates.new_latitude(np.array([0, 1, 2]))
longitude = coordinates.new_longitude(np.array([0, 1, 2]))
fgroup = np.array([0])
cohort = np.array([0, 1, 2])


@pytest.fixture()
def forcing_param():
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
def forcing_param_with_init():
    init_cond_biomass = ForcingUnit(
        forcing=xr.DataArray(
            dims=("functional_group", "latitude", "longitude"),
            coords={"functional_group": fgroup, "latitude": latitude, "longitude": longitude},
            data=np.full((fgroup.size, latitude.size, longitude.size), 0, dtype=float),
            attrs={"units": str(StandardUnitsLabels.biomass.units)},
            name="initial_condition_biomass",
        )
    )
    init_cond_production = ForcingUnit(
        forcing=xr.DataArray(
            dims=("functional_group", "latitude", "longitude", "cohort"),
            coords={"functional_group": fgroup, "latitude": latitude, "longitude": longitude, "cohort": cohort},
            data=np.full((fgroup.size, latitude.size, longitude.size, cohort.size), 0, dtype=float),
            attrs={"units": str(StandardUnitsLabels.production.units)},
            name="initial_condition_production",
        )
    )
    temperature = xr.DataArray(
        dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": latitude, "longitude": longitude},
        data=np.full((time.size, latitude.size, longitude.size), 0, dtype=float),
        attrs={"units": str(StandardUnitsLabels.temperature.units)},
        name="temperature",
    )
    primary_production = temperature.copy()
    primary_production.attrs["units"] = StandardUnitsLabels.production.units
    primary_production.name = "primary_production"
    temperature = ForcingUnit(forcing=temperature)
    primary_production = ForcingUnit(forcing=primary_production)
    return ForcingParameters(
        temperature=temperature,
        primary_production=primary_production,
        initial_condition_biomass=init_cond_biomass,
        initial_condition_production=init_cond_production,
    )


@pytest.fixture()
def init_cond_biomass():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("functional_group", "latitude", "longitude"),
            coords={"functional_group": fgroup, "latitude": latitude, "longitude": longitude},
            data=np.full((fgroup.size, latitude.size, longitude.size), 0, dtype=float),
        )
    )


@pytest.fixture()
def init_cond_production():
    return ForcingUnit(
        forcing=xr.DataArray(
            dims=("functional_group", "latitude", "longitude", "cohort"),
            coords={"functional_group": fgroup, "latitude": latitude, "longitude": longitude, "cohort": cohort},
            data=np.full((fgroup.size, latitude.size, longitude.size, cohort.size), 0, dtype=float),
        )
    )


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
    return FunctionalGroups(functional_groups=[fgroup]).functional_groups


class TestConfigurationToDataset:
    def test_load_forcings(self, forcing_param):
        forcings = _as_dataset__load_forcings(forcing_parameters=forcing_param)
        assert isinstance(forcings, xr.Dataset)
        assert ConfigurationLabels.timestep in forcings
        assert ConfigurationLabels.resolution_latitude in forcings
        assert ConfigurationLabels.resolution_longitude in forcings

    def test_build_fgroup_dataset(self, fgroup_param):
        fgroup_dataset = _as_dataset__build_fgroup_dataset(functional_groups=fgroup_param)
        assert isinstance(fgroup_dataset, xr.Dataset)
        assert ConfigurationLabels.fgroup_name in fgroup_dataset
        assert ConfigurationLabels.energy_transfert in fgroup_dataset
        assert ConfigurationLabels.inv_lambda_max in fgroup_dataset
        assert ConfigurationLabels.inv_lambda_rate in fgroup_dataset
        assert ConfigurationLabels.temperature_recruitment_max in fgroup_dataset
        assert ConfigurationLabels.temperature_recruitment_rate in fgroup_dataset
        assert ConfigurationLabels.day_layer in fgroup_dataset
        assert ConfigurationLabels.night_layer in fgroup_dataset

    def test_build_cohort_dataset(self, fgroup_param):
        names = xr.DataArray(
            dims=ConfigurationLabels.fgroup, coords={ConfigurationLabels.fgroup: [0]}, data=["phytoplankton"]
        )
        cohort_dataset = _as_dataset__build_cohort_dataset(
            functional_groups=fgroup_param,
            names=names,
        )
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
        dataset = as_dataset(forcing_parameters=forcing_param, functional_groups=fgroup_param)

        expected = set(ConfigurationLabels) - {
            ConfigurationLabels.initial_condition_biomass,
            ConfigurationLabels.initial_condition_production,
        }

        assert np.all([label in dataset for label in expected])

    def test_as_dataset_with_initial_conditions(self, forcing_param_with_init, fgroup_param):
        dataset = as_dataset(forcing_parameters=forcing_param_with_init, functional_groups=fgroup_param)

        expected = set(ConfigurationLabels)

        assert np.all([label in dataset for label in expected])
