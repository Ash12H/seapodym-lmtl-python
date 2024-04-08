"""Defines type aliases for Seapopym standard types."""

from typing import TypeAlias

import xarray as xr

from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

SeapopymState: TypeAlias = xr.Dataset
"""SeapopymState is the model state that stores all forcings (IN/OUT) and parameters."""

SeapopymForcing: TypeAlias = xr.DataArray
"""SeapopymForcing is the forcing used as a xarray.DataArray in the SeapopymState."""

ForcingName: TypeAlias = ConfigurationLabels | ForcingLabels | str
"""ForcingName is the name of the forcing used as key in the SeapopymState."""

SeapopymDims: TypeAlias = CoordinatesLabels | str
"""SeapopymDims is the name of the dimensions used in the SeapopymState."""

ForcingAttrs: TypeAlias = dict[str, object]
"""ForcingAttrs is the attributes of the forcing used as xarray.DataArray.attrs in the SeapopymState."""
