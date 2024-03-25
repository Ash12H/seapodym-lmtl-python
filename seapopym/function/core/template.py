"""Contains a unique function to generate a template for a new variable based on the state of the model."""
from __future__ import annotations

from typing import Iterable

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.standard import coordinates


def generate_template(
    state: xr.Dataset,
    dims: Iterable[str],
    attributs: dict | None = None,
    chunk: dict[str, int] | None = None,
    **kargs: dict[str, Iterable],
) -> xr.Dataset:
    """
    Generate a template for a new variable based on the state of the model.

    Parameters
    ----------
    state: xr.Dataset
        The state of the model.
    dims: Iterable[str]
        The dimensions of the variable.
    attributs: None | dict
        The attributes of the variable.
    chunk: dict[str, int]
        The chunk size of the variable.
    **kargs: dict[str, Iterable]
        Additional dimensions that are not already defined in the state of the model.

    Warning:
    -------
    - For additional dimensions, use coordinates module from seapopym.standard. It implement the standard definition of
    the coordinates as defined in the CF conventions.

    Returns
    -------
    xr.Dataset
        A template for a new variable that can be used in a xarray.map_blocks function.

    """
    coords = {state.cf[dim].name: state.cf[dim] for dim in dims if dim in state.cf}
    coords = {**coords, **kargs}
    return coordinates.reorder_dims(xr.DataArray(dims=coords.keys(), coords=coords, attrs=attributs).cf.chunk(chunk))
