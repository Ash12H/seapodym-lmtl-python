from time import time

import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.apply_coefficient_to_primary_production import apply_coefficient_to_primary_production
from seapopym.logging.custom_logger import logger
from seapopym.standard.labels import CoordinatesLabels


class TestApplyCoefficientToPrimaryProduction:
    def test_apply_coefficient_to_primary_production_no_chunk(self, state_preprod_fg4_t4d_y1_x1_z3):
        start = time()
        results = apply_coefficient_to_primary_production(state_preprod_fg4_t4d_y1_x1_z3)
        stop = time()
        logger.debug(f"Execution time no chunk: {stop - start}")

        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf

        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape
        assert results.dtype == float
        assert np.array_equal(results, np.ones_like(results))
        assert len(results.attrs) > 0

    def test_apply_coefficient_to_primary_production_chunked(self, state_preprod_fg4_t4d_y1_x1_z3):
        chunk = {"Y": 1, "X": 1}
        data = state_preprod_fg4_t4d_y1_x1_z3.cf.chunk(chunk)
        start = time()
        results = apply_coefficient_to_primary_production(data, chunk=chunk).compute()
        stop = time()
        logger.debug(f"Execution time no chunk: {stop - start}")

        assert isinstance(results, xr.DataArray)
        dims = (CoordinatesLabels.functional_group, CoordinatesLabels.time, CoordinatesLabels.Y, CoordinatesLabels.X)
        for dim in dims:
            assert dim in results.cf
        shape = tuple(state_preprod_fg4_t4d_y1_x1_z3.cf[dim].size for dim in dims)
        assert results.shape == shape

        assert results.dtype == float

        assert np.array_equal(results, np.ones_like(results))
        assert len(results.attrs) > 0
