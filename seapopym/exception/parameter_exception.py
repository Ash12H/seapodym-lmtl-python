"""List of all exceptions used in the package."""

from __future__ import annotations

from typing import Iterable


class CohortTimestepConsistencyError(Exception):
    """Raised when the cohorts timesteps are not multiple of the global timestep."""

    def __init__(
        self: CohortTimestepConsistencyError, cohort_name: str, cohort_timesteps: list[float], global_timestep: float
    ) -> None:
        """Constructor."""
        msg = (
            f"The cohorts timesteps of the functional group {cohort_name} are not multiple of the global timestep."
            f"\nGlobal timestep : {global_timestep}"
            f"\nCohorts timesteps : {cohort_timesteps}"
        )
        super().__init__(msg)
        self.cohort_name = cohort_name
        self.cohort_timesteps = cohort_timesteps
        self.global_timestep = global_timestep


class DifferentForcingTimestepError(Exception):
    """Raised when the forcings have different timesteps."""

    def __init__(self: DifferentForcingTimestepError, timesteps: list[float] | dict[str, float]) -> None:
        """Constructor."""
        if isinstance(timesteps, dict):
            timestep_print = "\n".join([f"{key} : {value}" for key, value in timesteps.items()])
            msg = f"The forcings have different timesteps :\n{timestep_print}."
        if isinstance(timesteps, Iterable):
            msg = f"The forcings have different timesteps : {timesteps}."
        else:
            msg = f"timesteps must be a dict or an Iterable. Got {type(timesteps)}"
            raise TypeError(msg)
        super().__init__(msg)
        self.timesteps = timesteps


class TimestepInDaysError(Exception):
    """Raised when the forcing timestep is not in days."""

    def __init__(self: TimestepInDaysError, timestep: float | Iterable[float]) -> None:
        """Constructor."""
        msg = f"Timestep(s) must be expreced in days (int > 0). Got {timestep}."
        super().__init__(msg)
        self.timestep = timestep
