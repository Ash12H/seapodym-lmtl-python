import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.mask import (
    global_mask_kernel,
    landmask_from_nan,
    mask_by_fgroup,
    mask_by_fgroup_kernel,
)
from seapopym.standard import coordinates
from seapopym.standard.labels import CoordinatesLabels
from seapopym.standard.types import SeapopymForcing


@pytest.fixture()
def simple_forcing() -> xr.DataArray:
    """Simple data with shape (10, 10, 10, 3)."""
    coords_layer = coordinates.new_layer()
    coords_latitude = coordinates.new_latitude(np.arange(0, 10, 1))
    coords_longitude = coordinates.new_longitude(np.arange(0, 10, 1))
    coords_time = coordinates.new_time(xr.cftime_range(start="2000-01-01", end="2000-01-10", freq="D"))
    forcing = xr.DataArray(
        coords={
            "time": coords_time,
            "latitude": coords_latitude,
            "longitude": coords_longitude,
            "layer": coords_layer,
        },
        dims=["time", "latitude", "longitude", "layer"],
        data=np.full(
            (
                len(coords_time),
                len(coords_latitude),
                len(coords_longitude),
                len(coords_layer),
            ),
            1,
            dtype=float,
        ),
    )
    forcing.data[:, :5, :5, :] = np.nan
    return forcing


class TestLandmask:
    def test_landmask(self, simple_forcing):
        res = landmask_from_nan(simple_forcing)
        # Keep spatial dimensions only
        assert res.shape == simple_forcing.isel(time=0).shape
        # Only boolean values
        assert res.dtype == bool

    def test_landmask_kernel(self, state_preprod_fg4_t4d_y1_x1_z3):
        kernel = global_mask_kernel()
        results = kernel.run(state_preprod_fg4_t4d_y1_x1_z3)

        assert isinstance(results, SeapopymForcing)
        for dim in (CoordinatesLabels.Y, CoordinatesLabels.X, CoordinatesLabels.Z):
            assert dim in results.cf.coords

        assert results.shape == (
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.Y].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.X].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.Z].size,
        )

        assert results.dtype == bool


class TestMaskByFgroup:
    def test_mask_by_fgroup(self, state_preprod_fg4_t4d_y1_x1_z3):
        results = mask_by_fgroup(state_preprod_fg4_t4d_y1_x1_z3)

        assert isinstance(results, SeapopymForcing)
        for dim in (CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X):
            assert dim in results.cf.coords

        assert results.shape == (
            state_preprod_fg4_t4d_y1_x1_z3[CoordinatesLabels.functional_group].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.Y].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.X].size,
        )

        assert results.dtype == bool

    def test_mask_by_fgroup_kernel(self, state_preprod_fg4_t4d_y1_x1_z3):
        kernel = mask_by_fgroup_kernel()
        results = kernel.run(state_preprod_fg4_t4d_y1_x1_z3)

        assert isinstance(results, SeapopymForcing)
        for dim in (CoordinatesLabels.functional_group, CoordinatesLabels.Y, CoordinatesLabels.X):
            assert dim in results.cf.coords

        assert results.shape == (
            state_preprod_fg4_t4d_y1_x1_z3[CoordinatesLabels.functional_group].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.Y].size,
            state_preprod_fg4_t4d_y1_x1_z3.cf[CoordinatesLabels.X].size,
        )

        assert results.dtype == bool
