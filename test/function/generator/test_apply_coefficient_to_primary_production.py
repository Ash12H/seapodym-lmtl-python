import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.apply_coefficient_to_primary_production import apply_coefficient_to_primary_production
from seapopym.standard.labels import CoordinatesLabels


class TestApplyCoefficientToPrimaryProduction:
    def test_apply_coefficient_to_primary_production(self, state_preprod_fg4_t4d_y1_x1_z3):
        results = apply_coefficient_to_primary_production(state_preprod_fg4_t4d_y1_x1_z3)
        assert isinstance(results, xr.DataArray)
        for dim in (
            CoordinatesLabels.functional_group,
            CoordinatesLabels.time,
            CoordinatesLabels.Y,
            CoordinatesLabels.X,
        ):
            assert dim in results.cf.coords

        assert results.shape == (
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.functional_group].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.time].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.Y].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.X].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.Z].size,
        )

        assert results.dtype == float

        assert np.array_equal(results, np.ones_like(results))
