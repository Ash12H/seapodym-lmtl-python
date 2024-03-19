"""A module that provide tools to convert the configuration of the model to a xarray.Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import xarray as xr

from seapodym_lmtl_python.configuration.no_transport.labels import ConfigurationLabels
from seapodym_lmtl_python.configuration.no_transport.parameter_functional_group import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
    FunctionalGroupUnitRelationParameters,
)

if TYPE_CHECKING:
    from seapodym_lmtl_python.configuration.no_transport.parameters import ForcingParameters


def _as_dataset__load_forcings(forcing_parameters: ForcingParameters) -> xr.Dataset:
    """Return the forcings as a xarray.Dataset."""
    all_forcing = {
        forcing_key: forcing_value["forcing"]
        for forcing_key, forcing_value in attrs.asdict(forcing_parameters).items()
        if forcing_value is not None and forcing_key not in ["timestep", "resolution"]
    }
    all_forcing[ConfigurationLabels.timestep] = forcing_parameters.timestep
    all_forcing[ConfigurationLabels.resolution_latitude] = forcing_parameters.resolution[0]
    all_forcing[ConfigurationLabels.resolution_longitude] = forcing_parameters.resolution[1]

    return xr.Dataset(all_forcing)


def _as_dataset__build_fgroup_dataset__generate_param_coords(
    functional_groups: list[FunctionalGroupUnit],
) -> tuple[dict, xr.DataArray]:
    """
    Generate both the functional groups parameters as dictionary and the functional groups coordinates as
    xarray.DataArray.
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
    grps_param = _rec_parameters([attrs.asdict(grp) for grp in functional_groups])

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


def _as_dataset__build_fgroup_dataset__generate_variables(
    params: dict, classes_and_names: list[tuple[attrs.Attribute, str]]
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


def _as_dataset__build_fgroup_dataset(functional_groups: list[FunctionalGroupUnit]) -> xr.Dataset:
    """
    Return the functional groups parameters as a xarray.Dataset.

    Warning:
    -------
    - The `names_classes` list must be updated if the `FunctionalGroupUnit` class is updated. All variables in the list
    are stored in the dataset.

    """
    (param_as_dict, f_group_coord) = _as_dataset__build_fgroup_dataset__generate_param_coords(functional_groups)
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
    param_variables = _as_dataset__build_fgroup_dataset__generate_variables(param_as_dict, names_classes)
    return xr.Dataset(param_variables, coords={ConfigurationLabels.fgroup: f_group_coord})


def _as_dataset__build_cohort_dataset___cohort_by_fgroup(fgroup: int, timesteps_number: list[int]) -> xr.Dataset:
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


def _as_dataset__build_cohort_dataset(functional_groups: list[FunctionalGroupUnit], names: xr.DataArray) -> xr.Dataset:
    """Return the cohort parameters as a xarray.Dataset."""
    all_fgroups = functional_groups
    all_cohorts_timesteps = [fgroup.functional_type.cohorts_timesteps for fgroup in all_fgroups]
    all_index = [names[ConfigurationLabels.fgroup][names == fgroup.name] for fgroup in all_fgroups]
    return xr.merge(
        [
            _as_dataset__build_cohort_dataset___cohort_by_fgroup(grp_index, timesteps)
            for grp_index, timesteps in zip(all_index, all_cohorts_timesteps)
        ]
    )


def as_dataset(functional_groups: list[FunctionalGroupUnit], forcing_parameters: ForcingParameters) -> xr.Dataset:
    """Return the configuration as a xarray.Dataset."""
    fgroup = _as_dataset__build_fgroup_dataset(functional_groups=functional_groups)
    forcings = _as_dataset__load_forcings(forcing_parameters)
    cohorts = _as_dataset__build_cohort_dataset(functional_groups, fgroup[ConfigurationLabels.fgroup_name])
    return xr.merge([fgroup, forcings, cohorts])
