"""Tests for fxr.evaluate_with_reindex."""

import numpy as np
import pytest
import xarray as xr

import funcexpr_xr as fxr


@pytest.fixture
def da_with_nan():
    """DataArray with NaN at (time=100, x=30) and (time=300, x=30)."""
    return xr.DataArray(
        np.array(
            [[10.0, 20.0, np.nan, np.nan],
             [40.0, 50.0, 60.0,  70.0 ],
             [80.0, 90.0, np.nan, np.nan]],
            dtype=np.float64,
        ),
        dims=["time", "x"],
        coords={"time": [100.0, 200.0, 300.0], "x": [10.0, 20.0, 30.0, 40.0]},
    )


@pytest.fixture
def ref_grid():
    """Reference grid: time=[100,150,200], x=[10,15,20,30]."""
    return xr.DataArray(
        np.zeros((3, 4), dtype=np.float64),
        dims=["time", "x"],
        coords={"time": [100.0, 150.0, 200.0], "x": [10.0, 15.0, 20.0, 30.0]},
    )


class TestEvaluateWithReindexReturnType:
    def test_returns_dataarray(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid}, reindex_ref="ref"
        )
        assert isinstance(result, xr.DataArray)

    def test_result_dims_match_ref(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid}, reindex_ref="ref"
        )
        assert result.dims == ref_grid.dims

    def test_result_coords_match_ref(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid}, reindex_ref="ref"
        )
        np.testing.assert_array_equal(
            result.coords["time"].values, [100.0, 150.0, 200.0]
        )
        np.testing.assert_array_equal(
            result.coords["x"].values, [10.0, 15.0, 20.0, 30.0]
        )


class TestEvaluateWithReindexNoInterp:
    def test_matching_points_taken_as_is(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid}, reindex_ref="ref"
        )
        # (time=200, x=30) exists in da_with_nan -> 60.0
        assert result.sel(time=200.0, x=30.0).values == pytest.approx(60.0)

    def test_nonmatching_points_are_nan(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid}, reindex_ref="ref"
        )
        # (time=150, x=15) has no match in da_with_nan -> NaN
        assert np.isnan(result.sel(time=150.0, x=15.0).values)

    def test_source_nan_preserved(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid}, reindex_ref="ref"
        )
        # (time=100, x=30) is NaN in source -> stays NaN
        assert np.isnan(result.sel(time=100.0, x=30.0).values)


class TestEvaluateWithReindexWithInterp:
    def test_matching_points_taken_as_is(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid},
            reindex_ref="ref", interp=True,
        )
        # (time=200, x=30) exists in da_with_nan -> must be 60.0, not interpolated
        assert result.sel(time=200.0, x=30.0).values == pytest.approx(60.0)

    def test_interpolable_points_filled(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid},
            reindex_ref="ref", interp=True,
        )
        # (time=150, x=10): da_with_nan has time=[100,200,300] x=10 -> [10,40,80]
        # linear interp at time=150 -> 25.0
        assert result.sel(time=150.0, x=10.0).values == pytest.approx(25.0)

    def test_source_nan_not_propagated_where_interp_succeeds(self, da_with_nan, ref_grid):
        result = fxr.evaluate_with_reindex(
            "a", ctx={"a": da_with_nan, "ref": ref_grid},
            reindex_ref="ref", interp=True,
        )
        # (time=150, x=20): da_with_nan has time=[100,200,300] x=20 -> [20,50,90]
        # linear interp at time=150 -> 35.0
        assert result.sel(time=150.0, x=20.0).values == pytest.approx(35.0)


class TestEvaluateWithReindexValidation:
    def test_no_dataarray_raises(self):
        with pytest.raises(ValueError, match="funcexpr"):
            fxr.evaluate_with_reindex("a", ctx={"a": 1.0}, reindex_ref="a")

    def test_reindex_ref_not_in_ctx_raises(self, da_with_nan):
        with pytest.raises(ValueError, match="reindex_ref"):
            fxr.evaluate_with_reindex(
                "a", ctx={"a": da_with_nan}, reindex_ref="z"
            )

    def test_error_message_lists_available_keys(self, da_with_nan):
        with pytest.raises(ValueError, match="'a'"):
            fxr.evaluate_with_reindex(
                "a", ctx={"a": da_with_nan}, reindex_ref="z"
            )

    def test_reindex_ref_is_scalar_raises(self, da_with_nan):
        with pytest.raises(ValueError, match="reindex_ref"):
            fxr.evaluate_with_reindex(
                "a + s", ctx={"a": da_with_nan, "s": 1.0}, reindex_ref="s"
            )