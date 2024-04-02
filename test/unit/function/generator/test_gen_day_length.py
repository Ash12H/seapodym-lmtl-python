import cf_xarray  # noqa: F401  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.generator.day_length import day_length
from seapopym.standard.labels import CoordinatesLabels


class TestMaskByFgroup:
    def test_simple_working(self, state_preprod_fg4_t4d_y1_x1_z3):
        results = day_length(state_preprod_fg4_t4d_y1_x1_z3)
        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf.coords
        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape

        assert results.dtype == float

        assert np.all(results > 0)
        assert np.all(results <= 24)

    def test_simple_working_with_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {CoordinatesLabels.functional_group: 1}
        data = state_preprod_fg4_t4d_y1_x1_z3.chunk(chunk)
        results = day_length(data, chunk=chunk).compute()
        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf.coords
        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape

        assert results.dtype == float

        assert np.all(results > 0)
        assert np.all(results <= 24)
