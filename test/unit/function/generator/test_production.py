import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr

from seapopym.function.generator.production.production import production_kernel
from seapopym.standard.labels import CoordinatesLabels, ProductionLabels


class TestProduction:
    def test_mask_by_fgroup_without_init_export_chunk(self, state_production_fg4_t4d_y1_x1_c4):
        state_production_fg4_t4d_y1_x1_c4 = state_production_fg4_t4d_y1_x1_c4.drop_vars("initial_condition_production")
        kernel = production_kernel()
        results = kernel.run(state_production_fg4_t4d_y1_x1_c4)
        results = results[ProductionLabels.recruited]
        assert isinstance(results, xr.DataArray)
        dims = (
            CoordinatesLabels.functional_group,
            CoordinatesLabels.time,
            CoordinatesLabels.Y,
            CoordinatesLabels.X,
            CoordinatesLabels.cohort,
        )
        for dim in dims:
            assert dim in results.cf

        shape = tuple(state_production_fg4_t4d_y1_x1_c4.cf[dim].size for dim in dims)
        assert results.shape == shape
        assert results.dtype == float
        assert np.all(results.cf.isel({"Y": 0, "X": 0, CoordinatesLabels.cohort: 0}) == 1)
        assert np.all(results.cf.isel({"Y": 0, "X": 0, CoordinatesLabels.cohort: slice(1, None)}) == 0)
        assert len(results.attrs) > 0

    def test_mask_by_fgroup_with_chunk_and_without_init_export(self, state_production_fg4_t4d_y1_x1_c4):
        chunk = {"Y": 1, "X": 1}
        state_production_fg4_t4d_y1_x1_c4 = state_production_fg4_t4d_y1_x1_c4.drop_vars("initial_condition_production")
        state_production_fg4_t4d_y1_x1_c4 = state_production_fg4_t4d_y1_x1_c4.cf.chunk(chunk)
        kernel = production_kernel(chunk=chunk)
        results = kernel.run(state_production_fg4_t4d_y1_x1_c4).compute()
        results = results[ProductionLabels.recruited]
        assert isinstance(results, xr.DataArray)
        dims = (
            CoordinatesLabels.functional_group,
            CoordinatesLabels.time,
            CoordinatesLabels.Y,
            CoordinatesLabels.X,
            CoordinatesLabels.cohort,
        )
        for dim in dims:
            assert dim in results.cf

        shape = tuple(state_production_fg4_t4d_y1_x1_c4.cf[dim].size for dim in dims)
        assert results.shape == shape
        assert results.dtype == float
        assert np.all(results.cf.isel({"Y": 0, "X": 0, CoordinatesLabels.cohort: 0}) == 1)
        assert np.all(results.cf.isel({"Y": 0, "X": 0, CoordinatesLabels.cohort: slice(1, None)}) == 0)
        assert len(results.attrs) > 0
