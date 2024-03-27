import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.production.production import production
from seapopym.logging.custom_logger import logger
from seapopym.standard.labels import CoordinatesLabels, ProductionLabels


class TestProduction:
    def test_mask_by_fgroup_without_init_export_chunk(self, state_production_fg4_t4d_y1_x1_c4):
        state_production_fg4_t4d_y1_x1_c4 = state_production_fg4_t4d_y1_x1_c4.drop_vars("initial_condition_production")
        results = production(state_production_fg4_t4d_y1_x1_c4, export_preproduction=None)
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
        state_production_fg4_t4d_y1_x1_c4 = state_production_fg4_t4d_y1_x1_c4.drop_vars("initial_condition_production")
        chunk = {"Y": 1, "X": 1}

        results = production(
            state_production_fg4_t4d_y1_x1_c4.cf.chunk(chunk), chunk=chunk, export_preproduction=None
        ).compute()
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
