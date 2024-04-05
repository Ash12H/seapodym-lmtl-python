from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, ParamSpecArgs, ParamSpecKwargs

import cf_xarray
import dask.array as da
import numpy as np
import xarray as xr
from attrs import define, field, fields, validators

from seapopym.function.core.template import BaseTemplate, ForcingTemplate, StateTemplate
from seapopym.logging.custom_logger import logger
from seapopym.standard import coordinates, types


@define
class KernelUnits:
    name: str
    template: BaseTemplate
    function: Callable[
        [types.SeapopymState, ParamSpecArgs, ParamSpecKwargs], types.SeapopymState | types.SeapopymForcing
    ]

    def _map_block_without_dask(self, state: types.SeapopymState, *args, **kwargs):
        logger.debug(f"Direct computation for {self.function.__name__}.")
        results = self.function(state, *args, **kwargs)

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
                results[template.name] = results[template.name].assign_attrs(template.attributs)
            return results

        msg = "The template attribut should be a ForcingTemplate or a StateTemplate."
        raise TypeError(msg)

    def _map_block_with_dask(self, state: types.SeapopymState, *args, **kwargs):
        logger.debug(f"Creating template for {self.function.__name__}.")

        result_template = self.template.generate(state)

        logger.debug(f"Applying map_blocks to {self.function.__name__}.")
        return xr.map_blocks(self.function, state, template=result_template, kwargs=kwargs, args=args)

    def run(
        self: KernelUnits, state: types.SeapopymState, *args: ParamSpecArgs, **kwargs: ParamSpecKwargs
    ) -> types.SeapopymState | types.SeapopymForcing:
        if state.chunks is None:
            self._map_block_without_dask(state, *args, **kwargs)

        return self._map_block_with_dask(state, *args, **kwargs)
