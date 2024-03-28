"""This module contains the compiled (JIT) functions used by the biomass generator."""
from __future__ import annotations

import numpy as np
from numba import jit


@jit()
def biomass_sequence(recruited: np.ndarray, mortality: np.ndarray, initial_conditions: np.ndarray | None) -> np.ndarray:
    """
    Compute the biomass of the recruited individuals.

    Formula :
        B(t) = R(t) + (M(t) * B(t-1))

    Parameters
    ----------
    recruited : np.ndarray
        The number of recruited individuals. Minimum dims : (F, T, <Y, X, ...>). Type : np.float64
    mortality : np.ndarray
        The mortality rate of the recruited individuals. Minimum dims : (F, T, <Y, X, ...>). Type : np.float64
    initial_conditions : np.ndarray
        The initial biomass. Minimum dims : (F, <Y, X, ...>). Type : np.float64
        If None, the initial biomass is set to `np.zeros(recruited[:, 0, ...].shape)`.

    Warning
    -------
    - Be sure to use floats for the input arrays.

    """
    initial_conditions = (
        np.zeros(recruited[:, 0, ...].shape, dtype=np.float64) if initial_conditions is None else initial_conditions
    )
    biomass = np.zeros(recruited.shape)
    biomass[:, 0, ...] = recruited[:, 0, ...] + (mortality[:, 0, ...] * initial_conditions)

    for timestep in range(1, recruited.shape[1]):
        # -> For now, we consider that the mortality is the same for all cohorts because mortality is implicite to
        # pre-production. We can use the sum of all cohorts to compute the production.
        biomass[:, timestep, ...] = recruited[:, timestep, ...] + (
            mortality[:, timestep, ...] * biomass[:, timestep - 1, ...]
        )
    return biomass
