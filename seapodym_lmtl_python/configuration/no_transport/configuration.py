"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from pathlib import Path
from typing import IO, TYPE_CHECKING

import attrs
import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapodym_lmtl_python.configuration.base_configuration import BaseConfiguration
from seapodym_lmtl_python.configuration.no_transport.labels import ConfigurationLabels
from seapodym_lmtl_python.configuration.no_transport.parameter_functional_group import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
    FunctionalGroupUnitRelationParameters,
)

if TYPE_CHECKING:
    from seapodym_lmtl_python.configuration.no_transport.parameters import NoTransportParameters


class NoTransportConfiguration(BaseConfiguration):
    """Configuration for the NoTransportModel."""

    def __init__(self: NoTransportConfiguration, parameters: NoTransportParameters) -> None:
        """Create a NoTransportConfiguration object."""
        self._parameters = parameters

    @property
    def parameters(self: NoTransportConfiguration) -> NoTransportParameters:
        """The attrs dataclass that stores all the model parameters."""
        return self._parameters

    @classmethod
    def parse(cls: NoTransportConfiguration, configuration_file: str | Path | IO) -> NoTransportConfiguration:
        """Parse the configuration file and create a NoTransportConfiguration object."""
        msg = "This method is not implemented yet."
        raise NotImplementedError(msg)

    def _load_forcings(self: NoTransportConfiguration, **kargs: dict) -> xr.Dataset:
        """Return the forcings as a xarray.Dataset."""
        if kargs is None:
            kargs = {}

        all_forcing = {
            ConfigurationLabels.timestep: self.parameters.timestep,
            ConfigurationLabels.resolution: self.parameters.resolution,
        }

        for forcing_name, forcing_value in attrs.asdict(self.parameters.path_parameters).items():
            if forcing_value is not None:
                data = xr.open_dataset(
                    forcing_value["forcing_path"],
                    **kargs,
                )[forcing_value["name"]]
                all_forcing[forcing_name] = data

        return xr.Dataset(all_forcing)

    def _build_fgroup_dataset__generate_param_coords(self: NoTransportConfiguration) -> tuple[dict, xr.DataArray]:
        """
        Helper method used by _build_fgroup_dataset to generate both the functional groups parameters as dictionary and
        the functional groups coordinates as xarray.DataArray.
        """

        def _rec_parameters(data: list) -> dict:
            """Recursivaly list all parameters."""
            all_param = {}
            for key in data[0]:
                if not isinstance(data[0][key], dict):
                    all_param[key] = [dic[key] for dic in data]
                else:
                    all_param.update(_rec_parameters([dic[key] for dic in data]))
            return all_param

        # 1. Generate a dictionary with all parameters as list
        grps: list[FunctionalGroupUnit] = self.parameters.functional_groups_parameters.functional_groups
        grps_param = _rec_parameters([attrs.asdict(grp) for grp in grps])

        # 2. Generate the coordinates (i.e. functional groups)
        f_group_coord_data = list(range(len(grps_param["name"])))
        f_group_coord = xr.DataArray(
            coords=(f_group_coord_data,),
            dims=(ConfigurationLabels.fgroup,),
            name=ConfigurationLabels.fgroup,
            attrs={  # cf_xarray convention
                "flag_values": f_group_coord_data,
                "flag_meanings": " ".join(grps_param["name"]),
                "standard_name": "functional group",
            },
            data=f_group_coord_data,
        )

        return grps_param, f_group_coord

    def _build_fgroup_dataset__generate_variables(
        self: NoTransportConfiguration, params: dict, classes_and_names: list[tuple[attrs.Attribute, str]]
    ) -> dict[str, tuple]:
        """
        Generate a dictionary where each key is a variable name and each value is a tuple of parameters. It can be used
        to create a xr.Dataset.
        """

        def _sel_attrs_meta(param_class: attrs.Attribute, attribut: str) -> dict:
            """Extract metadata from a specific attribut in an Attrs dataclass."""
            return next(filter(lambda x: x.name == attribut, attrs.fields(param_class))).metadata

        def _generate_tuple(param_class: attrs.Attribute, name: str) -> tuple:
            return ((ConfigurationLabels.fgroup,), params[name], _sel_attrs_meta(param_class, name))

        return {name: _generate_tuple(cls, name) for cls, name in classes_and_names}

    def _build_fgroup_dataset(self: NoTransportConfiguration) -> xr.Dataset:
        """
        Return the functional groups parameters as a xarray.Dataset.
        This function is used by the _build_model_configuration function.
        """
        (param_as_dict, f_group_coord) = self._build_fgroup_dataset__generate_param_coords()
        names_classes = [
            (FunctionalGroupUnit, ConfigurationLabels.fgroup_name),
            (FunctionalGroupUnit, ConfigurationLabels.energy_transfert),
            (FunctionalGroupUnitRelationParameters, ConfigurationLabels.inv_lambda_max),
            (FunctionalGroupUnitRelationParameters, ConfigurationLabels.inv_lambda_rate),
            (FunctionalGroupUnitRelationParameters, ConfigurationLabels.temperature_recruitment_max),
            (FunctionalGroupUnitRelationParameters, ConfigurationLabels.temperature_recruitment_rate),
            (FunctionalGroupUnitMigratoryParameters, ConfigurationLabels.day_layer),
            (FunctionalGroupUnitMigratoryParameters, ConfigurationLabels.night_layer),
        ]
        param_variables = self._build_fgroup_dataset__generate_variables(param_as_dict, names_classes)
        return xr.Dataset(param_variables, coords={ConfigurationLabels.fgroup: f_group_coord})

    def _build_cohort_dataset(self: NoTransportConfiguration, names: xr.DataArray) -> xr.Dataset:
        """Return the cohort parameters as a xarray.Dataset."""

        def _cohort_by_fgroup(fgroup: int, timesteps_number: list[int]) -> xr.Dataset:
            """
            Build the cohort axis for a specific functional group using the `timesteps_number` parameter given by the
            user.
            """
            cohort_index = np.arange(0, len(timesteps_number), 1, dtype=int)
            max_timestep = np.cumsum(timesteps_number)
            min_timestep = max_timestep - (np.asarray(timesteps_number) - 1)
            mean_timestep = (max_timestep + min_timestep) / 2

            data_vars = {
                ConfigurationLabels.timesteps_number: (
                    (ConfigurationLabels.fgroup, ConfigurationLabels.cohort),
                    [timesteps_number],
                    {
                        "description": (
                            "The number of timesteps represented in the cohort. If there is no aggregation, all values are "
                            "equal to 1."
                        )
                    },
                ),
                ConfigurationLabels.min_timestep: (
                    (ConfigurationLabels.fgroup, ConfigurationLabels.cohort),
                    [min_timestep],
                    {"description": "The minimum timestep index."},
                ),
                ConfigurationLabels.max_timestep: (
                    (ConfigurationLabels.fgroup, ConfigurationLabels.cohort),
                    [max_timestep],
                    {"description": "The maximum timestep index."},
                ),
                ConfigurationLabels.mean_timestep: (
                    (ConfigurationLabels.fgroup, ConfigurationLabels.cohort),
                    [mean_timestep],
                    {"description": "The mean timestep index."},
                ),
            }

            return xr.Dataset(
                coords={ConfigurationLabels.fgroup: fgroup, ConfigurationLabels.cohort: cohort_index},
                data_vars=data_vars,
            )

        all_fgroups = self.parameters.functional_groups_parameters.functional_groups
        all_cohorts_timesteps = [fgroup.functional_type.cohorts_timesteps for fgroup in all_fgroups]
        all_index = [names[ConfigurationLabels.fgroup][names == fgroup.name] for fgroup in all_fgroups]
        return xr.merge(
            [_cohort_by_fgroup(grp_index, timesteps) for grp_index, timesteps in zip(all_index, all_cohorts_timesteps)]
        )

    # TODO(Jules): Add the validation process
    # https://github.com/users/Ash12H/projects/3?pane=issue&itemId=54978078

    def as_dataset(self: NoTransportConfiguration) -> xr.Dataset:
        """Return the configuration as a xarray.Dataset."""
        fgroup = self._build_fgroup_dataset()
        forcings = self._load_forcings()
        cohorts = self._build_cohort_dataset(fgroup[ConfigurationLabels.fgroup_name])
        return xr.merge([fgroup, forcings, cohorts])
