"""An average temperature by fgroup computation wrapper. Use xarray.map_block."""

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.template import generate_template
from seapopym.standard.attributs import mask_by_fgroup_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels


def _mask_by_fgroup_helper(state: xr.Dataset) -> xr.DataArray:
    """
    The `mask_by_fgroup` has at least 3 dimensions (lat, lon, layer) and is a boolean array.

    Output
    ------
    - mask_by_fgroup  [functional_group, latitude, longitude] -> boolean
    """
    day_layers = state[ConfigurationLabels.day_layer]
    night_layers = state[ConfigurationLabels.night_layer]
    global_mask = state[PreproductionLabels.global_mask]

    masks = []
    for i in day_layers[CoordinatesLabels.functional_group]:
        day_pos = day_layers.sel(functional_group=i)
        night_pos = night_layers.sel(functional_group=i)

        day_mask = global_mask.cf.sel(Z=day_pos)
        night_mask = global_mask.cf.sel(Z=night_pos)
        masks.append(day_mask & night_mask)

    return xr.DataArray(
        coords={
            CoordinatesLabels.functional_group: day_layers[CoordinatesLabels.functional_group],
            global_mask.cf["Y"].name: global_mask.cf["Y"],
            global_mask.cf["X"].name: global_mask.cf["X"],
        },
        dims=(CoordinatesLabels.functional_group, global_mask.cf["Y"].name, global_mask.cf["X"].name),
        data=masks,
        name=PreproductionLabels.mask_by_fgroup,
    )


def mask_by_fgroup(state: xr.Dataset, chunk: dict) -> xr.DataArray:
    """Wrap the mask by fgroup computation with a map_block function."""
    max_dims = [CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X]
    template_mask = generate_template(state=state, dims=max_dims, attributs=mask_by_fgroup_desc, chunk=chunk)
    return xr.map_blocks(_mask_by_fgroup_helper, state, template=template_mask)
