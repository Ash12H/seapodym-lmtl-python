"""
Functions used to generalize the usage of map_blocks function in the model.

Notes
-----
This module is used to generate a template for a new variable that can be used in a xarray.map_blocks function. The
template is based on the state of the model and the dimensions of the new variable. The template can be chunked if
needed.

### xarray documentation:

> If none of the variables in obj is backed by dask arrays, calling this function is equivalent to calling
> `func(obj, *args, **kwargs)`.

"""
from __future__ import annotations

from typing import Callable, Iterable, ParamSpecArgs, ParamSpecKwargs

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.standard import coordinates


# TODO(Jules): Add name integration to template
def generate_template(
    state: xr.Dataset,
    dims: Iterable[str],
    attributs: dict | None = None,
    name: str | None = None,
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
        The chunk size of the variable. If None the template is not chunked. If you want to automatically chunk the
        variable, set the chunk parameter to `{}`.
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
    # dims      <- cf_xarray style naming, ex: "Y"
    # coords    <- data names, ex: "latitude"
    coords = {state.cf[dim].name: state.cf[dim] for dim in dims if dim in state.cf.coords}
    coords = {**coords, **kargs}
    data = xr.DataArray(dims=coords.keys(), coords=coords, name=name, attrs=attributs)
    if chunk is not None:
        chunk = {dim: chunk[dim] for dim in coords if dim in chunk}
        data = data.cf.chunk(chunk)
    return coordinates.reorder_dims(data)


def apply_map_block(
    function: Callable[[xr.Dataset, ParamSpecArgs, ParamSpecKwargs], xr.DataArray],
    state: xr.Dataset,
    template: Iterable[str] | xr.DataArray | xr.Dataset,
    attributs: dict | None = None,
    chunk: dict | None = None,
    *args: list,
    **kwargs: dict,
) -> xr.DataArray:
    """
    Wrap the function computation with a map_block function. If the state is not chunked, the function is directly
    applied to the state.

    Parameters
    ----------
    function: Callable[[xr.Dataset, ParamSpecArgs, ParamSpecKwargs], xr.DataArray]
        The function to apply to the state.
    state: xr.Dataset
        The state of the model.
    template: Iterable[str] | xr.DataArray | xr.Dataset
        The template of the new variable.
        You can pass a list of dimensions and the function will generate the template for you.
        You can also pass a DataArray or a Dataset as a template to override the default template generation.
    attributs: None | dict
        The attributes of the variable.
    chunk: None | dict
        The chunk size of the variable. If None the template is not chunked. If you want to automatically chunk the
        variable, set the chunk parameter to `{}`.
    *args: list
        Additional arguments to pass to the `function`.
    **kwargs: dict
        Additional keyword arguments to pass to the `function`.

    """
    if attributs is None:
        attributs = {}
    if len(state.chunks) == 0:  # Dataset chunks == FrozenDict({}) when not chunked
        return function(state, *args, **kwargs).assign_attrs(attributs)
    if not isinstance(template, (xr.DataArray, xr.Dataset)):
        template = generate_template(state=state, dims=template, attributs=attributs, chunk=chunk)
    return xr.map_blocks(function, state, template=template, kwargs=kwargs, args=args)
