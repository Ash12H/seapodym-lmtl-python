import pytest

from seapodym_lmtl_python.configuration.no_transport.parameter_functional_group import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
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


class TestFunctionalGroupUnit:
    def test_functional_group_unit_creation(self):
        functional_group: FunctionalGroupUnit = FunctionalGroupUnit(
            name="Test Group", energy_transfert=0.5, functional_type=None, migratory_type=None
        )
        assert functional_group.name == "Test Group"
        assert functional_group.energy_transfert == 0.5
        assert functional_group.functional_type is None
        assert functional_group.migratory_type is None

    def test_functional_group_unit_energy_transfert_validation(self):
        with pytest.raises(ValueError):
            FunctionalGroupUnit(name="Invalid Group", energy_transfert=-0.5, functional_type=None, migratory_type=None)

        with pytest.raises(ValueError):
            FunctionalGroupUnit(name="Invalid Group", energy_transfert=1.5, functional_type=None, migratory_type=None)

    # def test_functional_group_unit_functional_type(self):
    #     functional_group = FunctionalGroupUnit(
    #         name="Test Group", energy_transfert=0.5, functional_type="Functional Type", migratory_type=None
    #     )
    #     assert functional_group.functional_type == "Functional Type"

    # def test_functional_group_unit_migratory_type(self):
    #     functional_group = FunctionalGroupUnit(
    #         name="Test Group", energy_transfert=0.5, functional_type=None, migratory_type="Migratory Type"
    #     )
    #     assert functional_group.migratory_type == "Migratory Type"
