"""Define the Kernels used in the model."""

from __future__ import annotations

from typing import Callable, Iterable, ParamSpecArgs, ParamSpecKwargs

import xarray as xr
from attrs import define, field

from seapopym.function.core.template import BaseTemplate, ForcingTemplate, StateTemplate
from seapopym.logging.custom_logger import logger
from seapopym.standard import types


@define
class KernelUnits:
    name: str
    template: BaseTemplate
    function: Callable[
        [types.SeapopymState, ParamSpecArgs, ParamSpecKwargs], types.SeapopymState | types.SeapopymForcing
    ]
    args: ParamSpecArgs = field(factory=tuple)
    kwargs: ParamSpecKwargs = field(factory=dict)

    def _map_block_without_dask(self: KernelUnits, state: types.SeapopymState):
        logger.debug(f"Direct computation for {self.function.__name__}.")
        results = self.function(state, *self.args, **self.kwargs)

        if isinstance(self.template, ForcingTemplate):
            if isinstance(results, types.SeapopymState):
                msg = "When the function returns a xarray.Dataset, the template attribut should be a ForcingTemplate."
                raise TypeError(msg)
            results.name = self.template.name
            return results.assign_attrs(self.template.attrs)

        if isinstance(self.template, StateTemplate):
            if isinstance(results, types.SeapopymForcing):
                msg = "When the function returns a xarray.Dataset, the template attribut should be a StateTemplate."
                raise TypeError(msg)
            for template in self.template.template:
                if template.name not in results:
                    msg = f"Variable {template.name} is not in the results."
                    raise ValueError(msg)
                results[template.name] = results[template.name].assign_attrs(template.attrs)
            return results

        msg = "The template attribut should be a ForcingTemplate or a StateTemplate."
        raise TypeError(msg)

    def _map_block_with_dask(self: KernelUnits, state: types.SeapopymState):
        logger.debug(f"Creating template for {self.function.__name__}.")

        result_template = self.template.generate(state)

        logger.debug(f"Applying map_blocks to {self.function.__name__}.")
        return xr.map_blocks(self.function, state, template=result_template, args=self.args, kwargs=self.kwargs)

    def run(self: KernelUnits, state: types.SeapopymState) -> types.SeapopymState | types.SeapopymForcing:
        if len(state.chunks) == 0:
            return self._map_block_without_dask(state)

        return self._map_block_with_dask(state)


class Kernel:
    def __init__(self: Kernel, functions: Iterable[KernelUnits]) -> None:
        self._kernels = functions

    def run(self: Kernel, state: types.SeapopymState) -> types.SeapopymState:
        for kernel in self._kernels:
            results = kernel.run(state)
            state = state.merge(results)
        return state
