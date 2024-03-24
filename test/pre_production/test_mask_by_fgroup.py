import cf_xarray  # noqa: F401
import numpy as np
import pytest
import xarray as xr

from seapopym.function.generator.pre_production import mask_by_fgroup
from seapopym.standard.labels import ConfigurationLabels


@pytest.fixture()
def day_layers():
    return xr.DataArray(
        dims=(ConfigurationLabels.fgroup,),
        coords={ConfigurationLabels.fgroup: np.arange(4)},
        data=np.array([1, 1, 2, 2], dtype=int),
    )


@pytest.fixture()
def night_layers():
    return xr.DataArray(
        dims=(ConfigurationLabels.fgroup,),
        coords={ConfigurationLabels.fgroup: np.arange(4)},
        data=np.array([1, 2, 2, 1], dtype=int),
    )


@pytest.fixture()
def mask():
    return xr.DataArray(
        dims=("Z", "Y", "X"),
        coords={"Y": np.arange(3), "X": np.arange(3), "Z": np.array([1, 2])},
        data=np.array(
            [
                [
                    [0, 1, 1],
                    [0, 1, 1],
                    [0, 1, 1],
                ],
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ],
            ],
            dtype=bool,
        ),
    )


class TestMaskByFgroup:
    def test_mask_by_fgroup(self, day_layers, night_layers, mask):
        fgroup_mask = mask_by_fgroup(
            day_layers=day_layers,
            night_layers=night_layers,
            mask=mask,
        )

        assert isinstance(fgroup_mask, xr.DataArray)
        for dim in (ConfigurationLabels.fgroup, "Y", "X"):
            assert dim in fgroup_mask.cf.coords

        assert fgroup_mask.shape == (day_layers.size, mask.cf["Y"].size, mask.cf["X"].size)

        assert fgroup_mask.dtype == bool

        assert np.array_equal(
            fgroup_mask.sel({ConfigurationLabels.fgroup: 0}),
            mask.cf.sel(Z=1),
        )
        assert np.array_equal(
            fgroup_mask.sel({ConfigurationLabels.fgroup: 1}),
            mask.cf.sel(Z=2),
        )
        assert np.array_equal(
            fgroup_mask.sel({ConfigurationLabels.fgroup: 2}),
            mask.cf.sel(Z=2),
        )
        assert np.array_equal(
            fgroup_mask.sel({ConfigurationLabels.fgroup: 3}),
            mask.cf.sel(Z=2),
        )
