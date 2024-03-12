import numpy as np

from seapodym_lmtl_python.production.production import ageing


class TestAgeing:
    def test_ageing_1d_array_without_aggregation(self):
        production = np.array([1, 2, 3, 4])
        nb_timestep_by_cohort = np.array([1, 1, 1, 1])
        aged_production = ageing(production, nb_timestep_by_cohort)
        expected_output = np.array([0, 1, 2, 7])
        assert np.array_equal(aged_production, expected_output)

        production = np.array([1, 2, 3, 0])
        aged_production = ageing(production, nb_timestep_by_cohort)
        expected_output = np.array([0, 1, 2, 3])
        assert np.array_equal(aged_production, expected_output)

        production = np.array([0, 0, 0, 0])
        aged_production = ageing(production, nb_timestep_by_cohort)
        assert np.array_equal(aged_production, production)

        # NOTE(Jules): This should never happen, but we have to know what is happening when we have negative values
        production = np.array([-1, -1, -1, -1])
        aged_production = ageing(production, nb_timestep_by_cohort)
        expected_output = np.array([0, -1, -1, -2])
        assert np.array_equal(aged_production, expected_output)

    def test_ageing_1d_array_with_aggregation(self):
        production = np.array([1, 2, 3, 4])
        nb_timestep_by_cohort = np.array([2, 2, 2, 2])
        aged_production = ageing(production, nb_timestep_by_cohort)
        expected_output = np.array([0.5, 1.5, 2.5, 5.5])
        assert np.array_equal(aged_production, expected_output)

        production = np.array([1, 2, 3, 0])
        aged_production = ageing(production, nb_timestep_by_cohort)
        expected_output = np.array([0.5, 1.5, 2.5, 1.5])
        assert np.array_equal(aged_production, expected_output)

        production = np.array([0, 0, 0, 0])
        aged_production = ageing(production, nb_timestep_by_cohort)
        assert np.array_equal(aged_production, production)

        # NOTE(Jules): This should never happen, but we have to know what is happening when we have negative values
        production = np.array([-1, -1, -1, -1])
        aged_production = ageing(production, nb_timestep_by_cohort)
        expected_output = np.array([-0.5, -1, -1, -1.5])
        assert np.array_equal(aged_production, expected_output)

    def test_ageing_nan_values(self):
        production = np.array([1, 2, 3, 4, np.nan])
        nb_timestep_by_cohort = np.array([1, 1, 1, 1, 1])
        aged_production = ageing(production, nb_timestep_by_cohort)
        # NOTE(Jules):
        # -----------
        #   This should be the expected output, but the ageing function is developed to handle non nan
        #   values only. That was a choice to simplify the implementation and speed up the computation.
        #   Numba.jit function are simple and we use xarray to ensure data consistency.
        #   -------------------------------------------------------------------------------------------
        #   Expected output : expected_output = np.array([0, 1, 2, 7, np.nan])
        #   -------------------------------------------------------------------------------------------
        expected_output = np.array([0, 1, 2, 3, np.nan])
        assert np.array_equal(aged_production, expected_output, equal_nan=True)
