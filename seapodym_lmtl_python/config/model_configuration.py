"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from typing import Callable

import attrs
import cf_xarray  # noqa: F401
import xarray as xr

from seapodym_lmtl_python.config.parameters import Parameters, PathParameters


def _load_forcings(path_parameters: PathParameters, **kargs: dict) -> xr.Dataset:
    """Return the forcings as a xarray.Dataset."""
    if kargs is None:
        kargs = {}

    all_forcing = {}

    for forcing_name, forcing_value in attrs.asdict(path_parameters).items():
        if forcing_value is not None:
            all_forcing[forcing_name] = xr.open_dataset(
                forcing_value["forcing_path"],
                **kargs,
            )[forcing_value["name"]]

    return xr.Dataset(all_forcing)


def _validate_forcings(parameters: Parameters, forcings: xr.Dataset) -> None:
    """Validate the forcings against the parameters."""
    # TODO(Jules): Check that all day/night_layer values are in the depth coordinates
    return forcings


def _build_fgroup_dataset(parameters: Parameters) -> xr.Dataset:
    """
    Return the functional groups parameters as a xarray.Dataset.
    This function is used by the _build_model_configuration function.
    """
    # 1. Parse attrs.dataclass
    grps = parameters.functional_groups_parameters.functional_groups
    grp_keys = [i.name for i in attrs.fields(type(grps[0]))]
    grp_values = [list(t) for t in zip(*[attrs.astuple(grp) for grp in grps])]
    grps_param = dict(zip(grp_keys, grp_values))

    # 2. Generate the coordinates (i.e. functional groups)
    f_group_coord_data = list(range(len(grps_param["name"])))
    f_group_coord = xr.DataArray(
        coords=(f_group_coord_data,),
        dims=("functional_group",),
        name="functional_group",
        attrs={
            "flag_values": f_group_coord_data,
            "flag_meanings": " ".join(grps_param["name"]),
            "standard_name": "functional group",
        },
        data=f_group_coord_data,
    )
    # 3. Generate all the variables (i.e. parameters)
    day_position = (
        ("functional_group",),
        grps_param["day_layer"],
        {
            "description": "Layer in which the functional group is located during the day."
        },
    )
    night_position = (
        ("functional_group",),
        grps_param["night_layer"],
        {
            "description": "Layer in which the functional group is located during the night."
        },
    )
    functional_group_energy_coefficient = (
        ("functional_group",),
        grps_param["energy_coefficient"],
        {"description": "Energy coefficient of the functional group (named E')."},
    )
    # 4. Gather in a ready to merge dataset
    return xr.Dataset(
        data_vars={
            "day_position": day_position,
            "night_position": night_position,
            "functional_group_energy_coefficient": functional_group_energy_coefficient,
        },
        coords={"functional_group": f_group_coord},
    )


def _build_model_configuration(
    parameters: Parameters, forcings: xr.Dataset
) -> xr.Dataset:
    """Return the model configuration as a xarray.Dataset."""
    # 1. Simply add global model parameters
    data_as_dict = attrs.asdict(parameters.function_parameters)
    for parameter_unit in attrs.fields(type(parameters.function_parameters)):
        forcings[parameter_unit.name] = xr.DataArray(
            data_as_dict[parameter_unit.name], attrs=parameter_unit.metadata
        )
    # 2. Add functional groups parameters
    results = xr.merge(
        (forcings, _build_fgroup_dataset(parameters)),
        combine_attrs="no_conflicts",
    )
    return results.transpose(
        "functional_group",
        results.cf["T"].name,
        results.cf["Y"].name,
        results.cf["X"].name,
        results.cf["Z"].name,
    )


def process(
    param: Parameters,
    optional_validation: None | Callable[[Parameters, xr.Dataset], None] = None,
) -> xr.Dataset:
    """
    Return the model configuration as a xarray.Dataset. Also run some test to ensure the parameters are consistant
    with the forcings.

    The returned Dataset is a ready to use configuration of the model that can be reused.
    In the case of a more advanced use of the model, you can use the `optional_validation` argument to run additional
    tests on the parameters and the forcings.

    Parameters
    ----------
    param : Parameters
        The parameters of the model.
    optional_validation : None | Callable[[Parameters, xr.Dataset], None]
        A function that takes the parameters and the forcings (loaded from PathParameters or childs) as arguments and
        return None. This function is used to run additional tests on the parameters and the forcings.

    NOTE(Jules): It could be great to chunk the forcings according to functional_group axis to allow parallel computing
    during the dependent process.

    """
    forcings = _load_forcings(param.path_parameters)
    _validate_forcings(param, forcings)
    if optional_validation is not None:
        optional_validation(param, forcings)
    return _build_model_configuration(param, forcings)
