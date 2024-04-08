"""A landmask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401

from seapopym.function.core.day_length import mesh_day_length
from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import day_length_desc
from seapopym.standard.labels import CoordinatesLabels, PreproductionLabels

if TYPE_CHECKING:
    import xarray as xr

    from seapopym.standard.types import SeapopymForcing

# def day_length(
#     state: xr.Dataset, chunk: dict | None = None, angle_horizon_sun: int = 0, lazy: ForcingName | None = None
# ) -> SeapopymForcing:
#     """Wrap the day length computation with a map_block function."""

#     class_type = Template if lazy is None else TemplateLazy
#     template_attributs = {
#         "name": PreproductionLabels.day_length,
#         "dims": [CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
#         "attributs": day_length_desc(angle_horizon_sun=angle_horizon_sun),
#         "chunk": chunk,
#     }
#     if lazy is not None:
#         template_attributs["model_name"] = lazy
#     template = class_type(**template_attributs)

#     return apply_map_block(function=_wrapper_mesh_day_lengths, state=state, template=template)


def day_length_template(chunk: dict | None = None, angle_horizon_sun: float = 0) -> ForcingTemplate:
    return ForcingTemplate(
        name=PreproductionLabels.day_length,
        dims=[CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=day_length_desc(angle_horizon_sun=angle_horizon_sun),
        chunks=chunk,
    )


def _wrapper_mesh_day_lengths(state: xr.DataArray, angle_horizon_sun: float = 0) -> xr.DataArray:
    return mesh_day_length(
        state.cf[CoordinatesLabels.time],
        state.cf[CoordinatesLabels.Y],
        state.cf[CoordinatesLabels.X],
        angle_horizon_sun,
    )


def day_length_kernel(
    *, chunk: dict | None = None, template: ForcingTemplate | None = None, angle_horizon_sun: float = 0
) -> SeapopymForcing:
    if template is None:
        template = day_length_template(chunk=chunk, angle_horizon_sun=angle_horizon_sun)
    return KernelUnits(
        name=PreproductionLabels.day_length,
        template=template,
        function=_wrapper_mesh_day_lengths,
        kwargs={"angle_horizon_sun": angle_horizon_sun},
    )
