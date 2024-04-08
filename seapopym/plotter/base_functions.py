"""All the functions to plot the data of the model."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import xarray as xr

from seapopym.standard.labels import CoordinatesLabels, ForcingLabels
from seapopym.writer.base_functions import _helper_check_state

if TYPE_CHECKING:
    from seapopym.model.no_transport_model import NoTransportModel


def plot_biomass(model: NoTransportModel, method: Literal["sum", "mean"] = "mean", **kargs: dict) -> None:
    """Plot the biomass of the model."""
    _helper_check_state(model)

    if ForcingLabels.biomass not in model.state:
        msg = "The model does not have biomass to plot."
        raise ValueError(msg)

    with xr.set_options(keep_attrs=True):
        if method == "sum":
            data = model.state[ForcingLabels.biomass].cf.sum(["Y", "X"])
        elif method == "mean":
            data = model.state[ForcingLabels.biomass].cf.mean(["Y", "X"])
        else:
            msg = "The method must be either 'sum' or 'mean'."
            raise ValueError(msg)

    return data.cf.plot.line(x="T", hue=CoordinatesLabels.functional_group, **kargs)
