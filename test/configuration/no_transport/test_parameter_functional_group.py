import numpy as np
import pytest

from seapopym.configuration.no_transport.parameter_functional_group import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
    FunctionalGroupUnitRelationParameters,
)


class TestFunctionalGroupUnitMigratoryParameters:
    def test_functional_group_unit_migratory_parameters(self):
        migratory_parameters = FunctionalGroupUnitMigratoryParameters(day_layer=0, night_layer=1)
        assert migratory_parameters.day_layer == 0
        assert migratory_parameters.night_layer == 1

        migratory_parameters = FunctionalGroupUnitMigratoryParameters(day_layer="0", night_layer=0)
        assert migratory_parameters.day_layer == 0
        assert migratory_parameters.night_layer == 0

    def test_functional_group_unit_migratory_parameters_layer_validation(self):
        with pytest.raises(ValueError):
            FunctionalGroupUnitMigratoryParameters(day_layer="test", night_layer=0)
        with pytest.raises(ValueError):
            FunctionalGroupUnitMigratoryParameters(day_layer=0, night_layer="test")


class TestFunctionalGroupUnitRelationParameters:
    def test_functional_group_unit_relation_parameters(self):
        relation_parameters = FunctionalGroupUnitRelationParameters(
            inv_lambda_max=10,
            inv_lambda_rate=0.2,
            temperature_recruitment_rate=-0.2,
            temperature_recruitment_max=10,
            cohorts_timesteps=[1, 2, 3, 3, 1],
        )
        assert relation_parameters.inv_lambda_max == 10
        assert relation_parameters.inv_lambda_rate == 0.2
        assert relation_parameters.temperature_recruitment_rate == -0.2
        assert relation_parameters.temperature_recruitment_max == 10
        assert np.array_equal(relation_parameters.cohorts_timesteps, [1, 2, 3, 3, 1])

    def test_functional_group_unit_relation_parameters_last_cohort(self):
        relation_parameters = FunctionalGroupUnitRelationParameters(
            inv_lambda_max=10,
            inv_lambda_rate=0.2,
            temperature_recruitment_rate=-0.2,
            temperature_recruitment_max=10,
            cohorts_timesteps=[1, 2, 3, 4],
        )
        assert relation_parameters.inv_lambda_max == 10
        assert relation_parameters.inv_lambda_rate == 0.2
        assert relation_parameters.temperature_recruitment_rate == -0.2
        assert relation_parameters.temperature_recruitment_max == 10
        assert np.array_equal(relation_parameters.cohorts_timesteps, [1, 2, 3, 3, 1])

    def test_functional_group_unit_relation_parameters_validation(self):
        with pytest.raises(ValueError):
            FunctionalGroupUnitRelationParameters(
                inv_lambda_max=-10,
                inv_lambda_rate=0.2,
                temperature_recruitment_rate=-0.2,
                temperature_recruitment_max=10,
                cohorts_timesteps=[1, 2, 3, 4],
            )

        with pytest.raises(ValueError):
            FunctionalGroupUnitRelationParameters(
                inv_lambda_max=10,
                inv_lambda_rate=0.2,
                temperature_recruitment_rate=-0.2,
                temperature_recruitment_max=-10,
                cohorts_timesteps=[1, 2, 3, 4],
            )

        with pytest.raises(ValueError):
            FunctionalGroupUnitRelationParameters(
                inv_lambda_max=10,
                inv_lambda_rate=0.2,
                temperature_recruitment_rate=-0.2,
                temperature_recruitment_max=10,
                cohorts_timesteps=[1, 2, 3, 3],
            )


@pytest.fixture()
def functional_type() -> FunctionalGroupUnitRelationParameters:
    return FunctionalGroupUnitRelationParameters(
        inv_lambda_max=10,
        inv_lambda_rate=0.2,
        temperature_recruitment_max=10,
        temperature_recruitment_rate=-0.2,
        cohorts_timesteps=[1, 2, 3, 3, 1],
    )


@pytest.fixture()
def migratory_type() -> FunctionalGroupUnitMigratoryParameters:
    return FunctionalGroupUnitMigratoryParameters(day_layer=0, night_layer=1)


class TestFunctionalGroupUnit:
    def test_functional_group_unit_creation(self, functional_type, migratory_type):
        functional_group = FunctionalGroupUnit(
            name="Test Group", energy_transfert=0.5, functional_type=functional_type, migratory_type=migratory_type
        )
        assert functional_group.name == "Test Group"
        assert functional_group.energy_transfert == 0.5
        assert functional_group.functional_type == functional_type
        assert functional_group.migratory_type == migratory_type

    def test_functional_group_unit_energy_transfert_validation(self, functional_type, migratory_type):
        with pytest.raises(ValueError):
            FunctionalGroupUnit(
                name="Invalid Group",
                energy_transfert=-0.5,  # Negative
                functional_type=functional_type,
                migratory_type=migratory_type,
            )

        with pytest.raises(ValueError):
            FunctionalGroupUnit(
                name="Invalid Group",
                energy_transfert=1.5,  # Greater than 1
                functional_type=functional_type,
                migratory_type=migratory_type,
            )

    def test_functional_group_unit_functional_type_validation(self, migratory_type):
        with pytest.raises(TypeError):
            FunctionalGroupUnit(
                name="Test Group",
                energy_transfert=0.5,
                functional_type=None,  # wrong type
                migratory_type=migratory_type,
            )

    def test_functional_group_unit_migratory_type_validation(self, functional_type):
        with pytest.raises(TypeError):
            FunctionalGroupUnit(
                name="Test Group",
                energy_transfert=0.5,
                functional_type=functional_type,
                migratory_type=None,  # wrong type
            )
