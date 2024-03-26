import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.mask_by_fgroup import mask_by_fgroup
from seapopym.standard.labels import CoordinatesLabels


class TestMaskByFgroup:
    def test_mask_by_fgroup(self, state_preprod_fg4_t4d_y1_x1):
        results = mask_by_fgroup(state_preprod_fg4_t4d_y1_x1, {CoordinatesLabels.functional_group: 1})

        assert isinstance(results, xr.DataArray)
        for dim in (CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X):
            assert dim in results.cf.coords

        assert results.shape == (
            state_preprod_fg4_t4d_y1_x1[CoordinatesLabels.functional_group].size,
            state_preprod_fg4_t4d_y1_x1.cf[CoordinatesLabels.Y].size,
            state_preprod_fg4_t4d_y1_x1.cf[CoordinatesLabels.X].size,
        )

        assert results.dtype == bool
