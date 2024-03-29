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
from attr import define, field

from seapopym.logging.custom_logger import logger
from seapopym.standard import coordinates
from seapopym.standard.types import ForcingName, SeapopymDims

if TYPE_CHECKING:
    from seapopym.standard.types import ForcingAttrs, SeapopymForcing, SeapopymState


@define
class Template:
    """Template for a new variable that can be used in a xarray.map_blocks function."""

    name: ForcingName
    """The name of the variable."""
    dims: Iterable[SeapopymDims | tuple[SeapopymDims, Iterable[float] | xr.DataArray]]
    """The dimensions of the variable."""
    attributs: ForcingAttrs = field(factory=dict)
    """The attributes of the variable."""
    chunk: dict[str, int | Literal["auto"]] | None = field(default=None)
    """The chunk size of the variable. If None the template is not chunked. If you want to automatically chunk the
    variable, set the chunk parameter to `{}`."""

    def generate(self: Template, state: SeapopymState | None = None) -> SeapopymForcing:
        """
        Generate a template for a new variable based on the state of the model.

        Parameters
        ----------
        state: xr.Dataset
            The state of the model. Needed to generate the template if the dimensions are not defined as a tuple.

        Returns
        -------
        xr.Dataset
            A template for a new variable that can be used in a xarray.map_blocks function.

        """
        template_coords = {}
        for dim in self.dims:
            if isinstance(dim, SeapopymDims) and state is None:
                msg = "You need to provide the state of the model to generate the template."
                raise ValueError(msg)
            if isinstance(dim, SeapopymDims) and dim not in state.cf.coords:
                msg = f"Dimension {dim} is not defined in the state of the model."
                raise ValueError(msg)

            if isinstance(dim, tuple):
                template_coords[state.cf[dim[0]].name if dim[0] in state.cf.coords else dim[0]] = dim[1]
            elif isinstance(dim, SeapopymDims):
                template_coords[state.cf[dim].name] = state.cf[dim]

            else:
                msg = f"Dimension {dim} is not valid."
                raise TypeError(msg)

        template = xr.DataArray(
            dims=template_coords.keys(), coords=template_coords, name=self.name, attrs=self.attributs
        )
        if self.chunk is not None:
            chunk = {dim: self.chunk[dim] for dim in template_coords if dim in self.chunk}
            template = template.cf.chunk(chunk)
        return coordinates.reorder_dims(template)


def _map_block_without_dask(
    function: Callable[[SeapopymState, ParamSpecArgs, ParamSpecKwargs], SeapopymForcing | xr.Dataset[SeapopymForcing]],
    state: SeapopymState,
    template: Template | Iterable[Template],
    *args: list,
    **kwargs: dict,
) -> SeapopymForcing:
    logger.debug(f"Direct computation for {function.__name__}.")
    results = function(state, *args, **kwargs)

    if isinstance(results, xr.Dataset) and not isinstance(template, Iterable):
        msg = "When the function returns a xarray.Dataset, the template attribut should be an Iterable of Template."
        raise TypeError(msg)

    if isinstance(results, xr.DataArray) and isinstance(template, Iterable):
        msg = "When the function returns a xarray.DataArray, the template attribut should be a Template."
        raise TypeError(msg)

    if isinstance(template, Template):
        results.name = template.name
        return results.assign_attrs(template.attributs)

    for tmp in template:
        if tmp.name not in results:
            msg = f"Variable {tmp.name} is not in the results."
            raise ValueError(msg)
        results[tmp.name] = results[tmp.name].assign_attrs(tmp.attributs)
    return results


def _map_block_with_dask(
    function: Callable[[SeapopymState, ParamSpecArgs, ParamSpecKwargs], SeapopymForcing | xr.Dataset[SeapopymForcing]],
    state: SeapopymState,
    template: Template | Iterable[Template],
    *args: list,
    **kwargs: dict,
) -> SeapopymForcing:
    logger.debug(f"Creating template for {function.__name__}.")

    if isinstance(template, Template):
        result_template = template.generate(state)
    elif isinstance(template, Iterable):
        result_template = xr.Dataset({tmp.name: tmp.generate(state) for tmp in template})
    else:
        msg = "The template attribut should be a Template or an Iterable of Template."
        raise TypeError(msg)

    logger.debug(f"Applying map_blocks to {function.__name__}.")
    return xr.map_blocks(function, state, template=result_template, kwargs=kwargs, args=args)


def apply_map_block(
    function: Callable[[SeapopymState, ParamSpecArgs, ParamSpecKwargs], SeapopymForcing | xr.Dataset[SeapopymForcing]],
    state: SeapopymState,
    template: Template | Iterable[Template],
    *args: list,
    **kwargs: dict,
) -> SeapopymForcing | xr.Dataset[SeapopymForcing]:
    """
    Wrap the function computation with a map_block function. If the state is not chunked, the function is directly
    applied to the state.

    Parameters
    ----------
    function: Callable[[xr.Dataset, ParamSpecArgs, ParamSpecKwargs], xr.DataArray | xr.Dataset]
        The function to apply to the state.
    state: xr.Dataset
        The state of the model.
    template: Template | Iterable[Template]
        The template for the new variable(s).
    *args: list
        Additional arguments to pass to the `function`.
    **kwargs: dict
        Additional keyword arguments to pass to the `function`.

    Notes
    -----
    - Templating is less efficient than direct computation for small datasets. If the dataset is small, you should
    provide the state as a in memory dataset.

    """
    if len(state.chunks) == 0:  # Dataset chunks == FrozenDict({}) when not chunked
        return _map_block_without_dask(function, state, template, *args, **kwargs)

    return _map_block_with_dask(function, state, template, *args, **kwargs)
