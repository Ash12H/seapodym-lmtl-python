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

from typing import TYPE_CHECKING, Callable, Iterable, Literal, ParamSpecArgs, ParamSpecKwargs

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.logging.custom_logger import logger
from seapopym.standard import coordinates
from seapopym.standard.types import ForcingName

if TYPE_CHECKING:
    from seapopym.standard.types import ForcingAttrs, SeapopymDims, SeapopymForcing, SeapopymState


def generate_template(
    state: SeapopymState,
    dims: Iterable[SeapopymDims],
    attributs: ForcingAttrs | None = None,
    name: ForcingName | None = None,
    chunk: dict[str, int | Literal["auto"]] | None = None,
    **kargs: dict[str, Iterable],
) -> SeapopymForcing:
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
    function: Callable[[SeapopymState, ParamSpecArgs, ParamSpecKwargs], SeapopymForcing | xr.Dataset[SeapopymForcing]],
    state: SeapopymState,
    name: Iterable[ForcingName] | ForcingName,
    dims: dict[ForcingName, Iterable[SeapopymDims]] | Iterable[SeapopymDims],
    dims_kwargs: dict[ForcingName, dict[str, Iterable]] | dict[str, Iterable] | None = None,
    attributs: dict[ForcingName, ForcingAttrs] | ForcingAttrs | None = None,
    chunk: dict[str, int | Literal["auto"]] | None = None,
    *args: list,
    **kwargs: dict,
) -> SeapopymForcing | xr.Dataset[SeapopymForcing]:
    """
    Wrap the function computation with a map_block function. If the state is not chunked, the function is directly
    applied to the state.

    Parameters
    ----------
    function: Callable[[xr.Dataset, ParamSpecArgs, ParamSpecKwargs], xr.DataArray]
        The function to apply to the state.
    state: xr.Dataset
        The state of the model.
    dims: Iterable[str]
        The dims of the new variable.
    dims_kwargs: None | dict
        Additional dims that are not already defined in the state of the model. See `generate_template` for more
        information.
    name: None | str
        The name of the variable.
    attributs: None | dict
        The attributes of the variable.
    chunk: None | dict
        The chunk size of the variable. If None the template is not chunked. If you want to automatically chunk the
        variable, set the chunk parameter to `{}`.
    *args: list
        Additional arguments to pass to the `function`.
    **kwargs: dict
        Additional keyword arguments to pass to the `function`.

    Notes
    -----
    - Templating is less efficient than direct computation for small datasets. If the dataset is small, you should
    provide the state as a in memory dataset.

    """

    def _without_dask() -> SeapopymForcing | xr.Dataset[SeapopymForcing]:
        logger.debug(f"Direct computation for {name}.")
        results = function(state, *args, **kwargs)

        if isinstance(name, ForcingName):
            results.name = name
            return results.assign_attrs(attributs)

        for var_name, results_var in zip(name, results):
            results[results_var].name = var_name
            results[results_var] = results[results_var].assign_attrs(attributs.get(var_name, {}))
        return results

    def _using_dask() -> SeapopymForcing | xr.Dataset[SeapopymForcing]:
        logger.debug(f"Creating template for {name}.")

        if isinstance(name, str):
            template = generate_template(
                state=state, dims=dims, name=name, attributs=attributs, chunk=chunk, **dims_kwargs
            )
        else:
            template = {
                var_name: generate_template(
                    state=state,
                    dims=dims[var_name],
                    name=var_name,
                    attributs=attributs.get(var_name, {}),
                    chunk=chunk,
                    **dims_kwargs.get(var_name, {}),
                )
                for var_name in name
            }
            template = xr.Dataset(template)

        logger.debug(f"Applying map_blocks to {name}.")
        return xr.map_blocks(function, state, template=template, kwargs=kwargs, args=args)

    if attributs is None:
        attributs = {}
    if dims_kwargs is None:
        dims_kwargs = {}

    if len(state.chunks) == 0:  # Dataset chunks == FrozenDict({}) when not chunked
        return _without_dask()

    return _using_dask()
