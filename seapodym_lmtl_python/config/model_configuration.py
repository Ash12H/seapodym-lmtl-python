"""
This module contains the configuration of the model as a xarray.Dataset. It allows to store the model parameters and
the forcings (lazily).
"""

from __future__ import annotations

from typing import Callable

import xarray as xr

from seapodym_lmtl_python.config import Parameters, PathParameters


def _load_forcings(path_parameters: PathParameters) -> xr.Dataset:
    """Return the forcings as a xarray.Dataset."""
    pass


def _validate_forcings(parameters: Parameters, forcings: xr.Dataset) -> None:
    """Validate the forcings against the parameters."""
    pass


def _build_model_configuration(
    parameters: Parameters, forcings: xr.Dataset
) -> xr.Dataset:
    """Return the model configuration as a xarray.Dataset."""
    pass


def model_configuration(
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
