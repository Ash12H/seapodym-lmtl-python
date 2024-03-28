import numpy as np
import pytest

from seapopym.function.generator.biomass.biomass import biomass_sequence


@pytest.fixture()
def ones():  # No mortality
    return np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=float)


@pytest.fixture()
def zeros():  # No survival
    return np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)


class TestBiomass:
    def test_biomass_sequence(self, ones, zeros):
        assert np.array_equal(biomass_sequence(recruited=ones, mortality=zeros, initial_conditions=None), ones)

        expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float)  # + 1 each timestep
        assert np.array_equal(biomass_sequence(recruited=ones, mortality=ones, initial_conditions=None), expected)

    def test_with_initial_conditions(self, ones):
        initial_conditions = np.array([1, 1], dtype=float)
        expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=float) + 1  # + 1 initial condition
        assert np.array_equal(
            biomass_sequence(recruited=ones, mortality=ones, initial_conditions=initial_conditions), expected
        )
