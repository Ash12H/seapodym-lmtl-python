"""Wrapper for the application of the transfert cooeficient to primary production. Use xarray.map_block."""

import cf_xarray  # noqa: F401
import xarray as xr

from seapopym.function.core.template import generate_template
from seapopym.standard.attributs import apply_coefficient_to_primary_production_desc
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, PreproductionLabels
from seapopym.standard.units import StandardUnitsLabels, check_units


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


def _apply_coefficient_to_primary_production_helper(state: xr.Dataset) -> xr.DataArray:
    """
    It is equivalent to generate the fisrt cohort of pre-production.

    Input
    -----
    - primary_production [time, latitude, longitude]
    - functional_group_coefficient [functional_group]

    Output
    ------
    - primary_production [functional_group, time, latitude, longitude]
    """
    primary_production = check_units(
        state[ConfigurationLabels.primary_production], StandardUnitsLabels.production.units
    )
    pp_by_fgroup_gen = (i * primary_production for i in state[ConfigurationLabels.energy_transfert])
    pp_by_fgroup = xr.concat(pp_by_fgroup_gen, dim=CoordinatesLabels.functional_group, combine_attrs="drop")
    pp_by_fgroup.name = "primary_production_by_fgroup"
    return pp_by_fgroup


def apply_coefficient_to_primary_production(state: xr.Dataset, chunk: dict) -> xr.DataArray:
    """Wrap the application of the transfert cooeficient to primary production with a map_block function."""
    max_dims = [CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X]
    template_coef_to_pp = generate_template(
        state=state, dims=max_dims, attributs=apply_coefficient_to_primary_production_desc, chunk=chunk
    )
    return xr.map_blocks(_apply_coefficient_to_primary_production_helper, state, template=template_coef_to_pp)
