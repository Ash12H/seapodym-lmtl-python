import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.generator.average_temperature import average_temperature_kernel
from seapopym.standard.labels import CoordinatesLabels


class TestAverageTemperature:
    def test_average_temperature_no_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        kernel = average_temperature_kernel()
        results = kernel.run(state_preprod_fg4_t4d_y1_x1_z3)
        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf

        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape
        assert results.dtype == float
        assert np.array_equal(results.cf.isel(T=0, Y=0, X=0), [0, 0.5, 1.0, 0.5])
        assert len(results.attrs) > 0

    def test_average_temperature_chunked(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {CoordinatesLabels.Y: 1, CoordinatesLabels.X: 1}
        kernel = average_temperature_kernel(chunk=chunk)
        results = kernel.run(state_preprod_fg4_t4d_y1_x1_z3.cf.chunk(chunk))

        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf
        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape
        assert results.dtype == float
        assert np.array_equal(results.cf.isel(T=0, Y=0, X=0), [0, 0.5, 1.0, 0.5])
        assert len(results.attrs) > 0
