import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.average_temperature import average_temperature
from seapopym.standard.labels import CoordinatesLabels


class TestMaskByFgroup:
    def test_mask_by_fgroup(self, state_preprod_fg4_t4d_y1_x1):
        fgroup_mask = average_temperature(state_preprod_fg4_t4d_y1_x1, chunk={CoordinatesLabels.functional_group: 1})

        assert isinstance(fgroup_mask, xr.DataArray)
        for dim in (
            CoordinatesLabels.functional_group,
            CoordinatesLabels.time,
            CoordinatesLabels.Y,
            CoordinatesLabels.X,
        ):
            assert dim in fgroup_mask.cf.coords

        assert fgroup_mask.shape == (
            state_preprod_fg4_t4d_y1_x1.cf[CoordinatesLabels.functional_group].size,
            state_preprod_fg4_t4d_y1_x1.cf[CoordinatesLabels.time].size,
            state_preprod_fg4_t4d_y1_x1.cf[CoordinatesLabels.Y].size,
            state_preprod_fg4_t4d_y1_x1.cf[CoordinatesLabels.X].size,
        )

        assert fgroup_mask.dtype == float

        assert np.array_equal(fgroup_mask.cf.isel(T=0, Y=0, X=0), [0, 0.5, 1.0, 0.5])
