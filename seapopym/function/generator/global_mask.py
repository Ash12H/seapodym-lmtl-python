"""A landmask computation wrapper. Use xarray.map_block."""
from __future__ import annotations

from typing import TYPE_CHECKING

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.mask import landmask_from_nan
from seapopym.function.core.template import ForcingTemplate
from seapopym.standard.attributs import global_mask_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels

if TYPE_CHECKING:
    import xarray as xr

    from seapopym.standard.types import SeapopymForcing


# def global_mask(state: xr.Dataset, chunk: dict | None = None, lazy: ForcingName | None = None) -> SeapopymForcing:
#     """Wrap the landmask computation with a map_block function."""

#     class_type = Template if lazy is None else TemplateLazy
#     template_attributs = {
#         "name": PreproductionLabels.global_mask,
#         "dims": [],
#         "attributs": global_mask_desc,
#         "chunk": chunk,
#     }
#     if lazy is not None:
#         template_attributs["model_name"] = lazy
#     template = class_type(**template_attributs)

#     return apply_map_block(function=_global_mask_from_nan, state=state, template=template)


def global_mask_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=PreproductionLabels.global_mask,
        dims=[CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z],
        attrs=global_mask_desc,
        chunks=chunk,
    )


def _global_mask_from_nan(state: xr.Dataset) -> xr.DataArray:
    return landmask_from_nan(state[ConfigurationLabels.temperature])


def global_mask_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> SeapopymForcing:
    if template is None:
        template = global_mask_template(chunk=chunk)
    return KernelUnits(
        name=PreproductionLabels.global_mask,
        template=template,
        function=_global_mask_from_nan,
    )
