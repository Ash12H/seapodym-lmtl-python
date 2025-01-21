"""Compute the layers depth. Based on the O. Titaud work from 2018."""
from __future__ import annotations

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr


def compute_limits_from_zeu_lat_sst(
    zeu: xr.DataArray,
    sst: xr.DataArray,
    layer_bounds: dict | None = None,
) -> xr.Dataset:
    """
    Compute the layers limits from zeu and definition. Apply a correction based on latitude and sst.

    Based on 2020 report evolution for cmems.

    Parameters
    ----------
    zeu : xarray.DataArray
        3D DataArray containing euphotic depth data.
    sst : xarray.DataArray
        3D DataArray countaining sea surface temperature data.
    layer_bounds : dict
        layer bounds definition w.r.t. zeu
        Default: {"epipelagic": 1.5, "upper_mesopelagic": 4.5, "lower_mesopelagic": 10.5, "treshold": 1000.0}

    Returns
    -------
    xarray.Dataset
        A Dataset containing limits of layers wrt to euphotic depth.

    """
    # Constant values
    if layer_bounds is None:
        layer_bounds = {"epipelagic": 1.5, "upper_mesopelagic": 4.5, "lower_mesopelagic": 10.5, "treshold": 1000.0}
    alpha = 10e-12
    beta = 0.5
    l2max = 400
    l3max = 800
    limit_40 = 40
    limit_60 = 60

    coef_a = alpha / ((1 - alpha) * np.exp(-beta * np.abs(zeu.cf["Y"])) + alpha)
    coef_a = np.tile(coef_a, (zeu.cf["X"].size, 1)).T
    coef_a = xr.DataArray(dims=(zeu.cf["Y"].name, zeu.cf["X"].name), coords=(zeu.cf["Y"], zeu.cf["X"]), data=coef_a)
    coef_a_40_60 = coef_a.where((np.abs(zeu.cf["Y"]) > limit_40) & (np.abs(zeu.cf["Y"]) <= limit_60), 0)

    zeu_0_40 = zeu.where(np.abs(zeu.cf["Y"]) <= limit_40, 0)
    sst_40_60 = sst.where((np.abs(zeu.cf["Y"]) > limit_40) & (np.abs(zeu.cf["Y"]) <= limit_60), 0)
    zeu_40_60 = zeu.where((np.abs(zeu.cf["Y"]) > limit_40) & (np.abs(zeu.cf["Y"]) <= limit_60), 0)

    # EPIPELAGIC
    epipelagic_0_40 = layer_bounds["epipelagic"] * zeu_0_40
    epipelagic_40_60 = coef_a_40_60 * (108 + 1.2 * sst_40_60) + (1 - coef_a_40_60) * (
        layer_bounds["epipelagic"] * zeu_40_60
    )
    epipelagic_60_90 = 108 + 1.2 * sst.where(np.abs(zeu.cf["Y"]) > limit_60, 0)
    epipelagic = epipelagic_0_40 + epipelagic_40_60 + epipelagic_60_90

    # UPPER MESOPELAGIC
    upper_mesopelagic_0_40 = layer_bounds["upper_mesopelagic"] * zeu_0_40
    upper_mesopelagic_40_60 = coef_a_40_60 * l2max + (1 - coef_a_40_60) * (
        layer_bounds["upper_mesopelagic"] * zeu_40_60
    )
    upper_mesopelagic_60_90 = xr.where(np.abs(zeu.cf["Y"]) > limit_60, l2max, 0)
    upper_mesopelagic = upper_mesopelagic_0_40 + upper_mesopelagic_40_60 + upper_mesopelagic_60_90

    # LOWER MESOPELAGIC
    lower_mesopelagic_0_40 = layer_bounds["lower_mesopelagic"] * zeu_0_40
    lower_mesopelagic_40_60 = coef_a_40_60 * l3max + (1 - coef_a_40_60) * (
        layer_bounds["lower_mesopelagic"] * zeu_40_60
    )
    lower_mesopelagic_60_90 = xr.where(np.abs(zeu.cf["Y"]) > limit_60, l3max, 0)
    lower_mesopelagic = lower_mesopelagic_0_40 + lower_mesopelagic_40_60 + lower_mesopelagic_60_90
    lower_mesopelagic = xr.where(
        lower_mesopelagic > layer_bounds["treshold"], layer_bounds["treshold"], lower_mesopelagic
    )

    return xr.Dataset(
        {
            "limit_surface": xr.zeros_like(zeu),
            "limit_epipelagic": xr.where(zeu.notnull(), epipelagic, np.nan),
            "limit_upper_mesopelagic": xr.where(zeu.notnull(), upper_mesopelagic, np.nan),
            "limit_lower_mesopelagic": xr.where(zeu.notnull(), lower_mesopelagic, np.nan),
        }
    )
