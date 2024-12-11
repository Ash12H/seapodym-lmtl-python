"""Funcitons used to generate a landmask from any forcing data."""
from __future__ import annotations

import cf_xarray  # noqa: F401  # noqa: F401
import xarray as xr

from seapopym.function.core.kernel import KernelUnits
from seapopym.function.core.template import ForcingTemplate
from seapopym.logging.custom_logger import logger
from seapopym.standard.attributs import global_mask_desc, mask_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels


def landmask_from_nan(forcing: xr.DataArray) -> xr.DataArray:
    """Create a landmask from a forcing data array."""
    mask = forcing.cf.isel(T=0).notnull().cf.reset_coords("T", drop=True)
    mask.name = "mask"
    mask.attrs = global_mask_desc
    return mask


def mask_by_fgroup(state: xr.Dataset) -> xr.DataArray:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """
    day_layers = state[ConfigurationLabels.day_layer]
    night_layers = state[ConfigurationLabels.night_layer]
    global_mask = state[ForcingLabels.global_mask]

    masks = []
    for i in day_layers[CoordinatesLabels.functional_group]:
        day_pos = day_layers.sel(functional_group=i)
        night_pos = night_layers.sel(functional_group=i)

        day_mask = global_mask.cf.sel(Z=day_pos)
        night_mask = global_mask.cf.sel(Z=night_pos)
        masks.append(day_mask & night_mask)

    return xr.DataArray(
        dims=(CoordinatesLabels.functional_group, global_mask.cf["Y"].name, global_mask.cf["X"].name),
        coords={
            CoordinatesLabels.functional_group: day_layers[CoordinatesLabels.functional_group],
            global_mask.cf["Y"].name: global_mask.cf["Y"],
            global_mask.cf["X"].name: global_mask.cf["X"],
        },
        data=masks,
    )


# NOTE(Jules):  Other functions can be implemented here. For example, a function that creates a landmask from a user
#               text file.


def apply_mask_to_state(state: xr.Dataset) -> xr.Dataset:
    """Apply a mask to a state dataset."""
    if ForcingLabels.global_mask in state:
        # logger.debug("Applying the global mask to the state.")
        return state.where(state[ForcingLabels.global_mask])
    return state


def global_mask_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.global_mask,
        dims=[CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z],
        attrs=global_mask_desc,
        chunks=chunk,
    )


def mask_by_fgroup_template(chunk: dict | None = None) -> ForcingTemplate:
    return ForcingTemplate(
        name=ForcingLabels.mask_by_fgroup,
        dims=[CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X],
        attrs=mask_by_fgroup_desc,
        chunks=chunk,
    )


def global_mask_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    def global_mask_from_nan(state: xr.Dataset) -> xr.DataArray:
        return landmask_from_nan(state[ForcingLabels.temperature])

    if template is None:
        template = global_mask_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.global_mask,
        template=template,
        function=global_mask_from_nan,
    )


def mask_by_fgroup_kernel(*, chunk: dict | None = None, template: ForcingTemplate | None = None) -> KernelUnits:
    if template is None:
        template = mask_by_fgroup_template(chunk=chunk)
    return KernelUnits(
        name=ForcingLabels.mask_by_fgroup,
        template=template,
        function=mask_by_fgroup,
    )
