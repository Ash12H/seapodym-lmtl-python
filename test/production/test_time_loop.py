import numpy as np
import pytest
from numba.core.errors import TypingError

from seapopym.production.production import time_loop


class TestTimeLoop:
    def test_time_loop_without_preproduction(self):
        primary_production = np.tile(np.array([1]), (5, 1, 1))  # T, X, Y
        mask_temperature = np.tile(np.array([False, False, False, False]), (5, 1, 1, 1))  # T, X, Y, C
        timestep_number = np.array([1, 1, 1, 1])  # C
        output_recruited, _ = time_loop(
            primary_production=primary_production,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
            initial_production=None,
            export_preproduction=None,
        )
        assert isinstance(output_recruited, np.ndarray)

    def test_time_loop_with_preproduction(self):
        primary_production = np.tile(np.array([1]), (5, 1, 1))  # T, X, Y
        mask_temperature = np.tile(np.array([False, False, False, False]), (5, 1, 1, 1))  # T, X, Y, C
        timestep_number = np.array([1, 1, 1, 1])  # C
        output_recruited, _ = time_loop(
            primary_production=primary_production,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
            initial_production=None,
            export_preproduction=np.array([0]),
        )
        assert isinstance(output_recruited, np.ndarray)

    def test_time_loop_without_recruitment(self):
        time_len = 5
        export_preproduction = np.arange(time_len, dtype=int)
        primary_production = np.tile(np.array([1]), (time_len, 1, 1))  # T, X, Y
        mask_temperature = np.tile(np.array([False, False, False, False]), (time_len, 1, 1, 1))  # T, X, Y, C
        timestep_number = np.array([1, 1, 1, 1])  # C
        output_recruited, output_preproduction = time_loop(
            primary_production=primary_production,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
            initial_production=None,
            export_preproduction=export_preproduction,
        )
        assert isinstance(output_recruited, np.ndarray)
        assert isinstance(output_preproduction, np.ndarray)

        # Mask is false so no production is recruited
        expected_output_recruited = np.array(
            [
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 0.0, 0.0, 0.0]]],
            ]
        )
        # Mask is false so all cohorts are ageing
        expected_output_preproduction = np.array(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 1.0, 0.0]]],
                [[[1.0, 1.0, 1.0, 1.0]]],
                [[[1.0, 1.0, 1.0, 2.0]]],
            ]
        )

        assert np.allclose(output_recruited, expected_output_recruited)
        assert np.allclose(output_preproduction, expected_output_preproduction)

    def test_time_loop_with_instant_recruitment(self):
        time_len = 5
        export_preproduction = np.arange(time_len, dtype=int)
        primary_production = np.tile(np.array([1]), (time_len, 1, 1))  # T, X, Y
        mask_temperature = np.tile(np.array([True, True, True, True]), (time_len, 1, 1, 1))  # T, X, Y, C
        timestep_number = np.array([1, 1, 1, 1])  # C
        output_recruited, output_preproduction = time_loop(
            primary_production=primary_production,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
            initial_production=None,
            export_preproduction=export_preproduction,
        )
        assert isinstance(output_recruited, np.ndarray)
        assert isinstance(output_preproduction, np.ndarray)

        # Mask is false so no production is recruited
        expected_output_recruited = np.array(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
            ]
        )
        # Mask is false so all cohorts are ageing
        expected_output_preproduction = np.array(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0, 0.0]]],
            ]
        )

        assert np.allclose(output_recruited, expected_output_recruited)
        assert np.allclose(output_preproduction, expected_output_preproduction)

    def test_time_loop_with_delayed_recruitment(self):
        time_len = 5
        export_preproduction = np.arange(time_len, dtype=int)
        primary_production = np.tile(np.array([1]), (time_len, 1, 1))  # T, X, Y
        mask_temperature = np.tile(np.array([False, True, True, True]), (time_len, 1, 1, 1))  # T, X, Y, C
        timestep_number = np.array([1, 1, 1, 1])  # C
        output_recruited, output_preproduction = time_loop(
            primary_production=primary_production,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
            initial_production=None,
            export_preproduction=export_preproduction,
        )
        assert isinstance(output_recruited, np.ndarray)
        assert isinstance(output_preproduction, np.ndarray)

        # Mask is false so no production is recruited
        expected_output_recruited = np.array(
            [
                [[[0.0, 0.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
                [[[0.0, 1.0, 0.0, 0.0]]],
            ]
        )
        # Mask is false so all cohorts are ageing
        expected_output_preproduction = np.array(
            [
                [[[1.0, 0.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 0.0, 0.0]]],
                [[[1.0, 1.0, 0.0, 0.0]]],
            ]
        )

        assert np.allclose(output_recruited, expected_output_recruited)
        assert np.allclose(output_preproduction, expected_output_preproduction)

    def test_time_loop_different_input_types(self):
        primary_production = np.array([1, 2, 3, 4])
        cohorts = np.array([1, 2, 3, 4])
        mask_temperature = np.array([True, False, True, False])
        timestep_number = np.array([1, 1, 1, 1])
        with pytest.raises(TypingError):
            time_loop(list(primary_production), cohorts, mask_temperature, timestep_number, None)
        with pytest.raises(TypingError):
            time_loop(primary_production, list(cohorts), mask_temperature, timestep_number, None)
        with pytest.raises(TypingError):
            time_loop(primary_production, cohorts, list(mask_temperature), timestep_number, None)
        with pytest.raises(TypingError):
            time_loop(primary_production, cohorts, mask_temperature, list(timestep_number), None)

    def test_time_loop_with_initial_conditions(self):
        time_len = 5
        export_preproduction = np.arange(time_len, dtype=int)
        primary_production = np.tile(np.array([1]), (time_len, 1, 1))  # T, X, Y
        mask_temperature = np.tile(np.array([False, False, False, False]), (time_len, 1, 1, 1))  # T, X, Y, C
        timestep_number = np.array([1, 1, 1, 1])  # C
        initial_production = np.array([[[1.0, 0.0, 0.0, 0.0]]])
        _, output_preproduction = time_loop(
            primary_production=primary_production,
            mask_temperature=mask_temperature,
            timestep_number=timestep_number,
            initial_production=initial_production,
            export_preproduction=export_preproduction,
        )
        expected = [
            [[[2.0, 0.0, 0.0, 0.0]]],
            [[[1.0, 2.0, 0.0, 0.0]]],
            [[[1.0, 1.0, 2.0, 0.0]]],
            [[[1.0, 1.0, 1.0, 2.0]]],
            [[[1.0, 1.0, 1.0, 3.0]]],
        ]
        assert np.all(output_preproduction == expected)
