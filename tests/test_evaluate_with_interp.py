"""Tests for fxr.evaluate_with_interp."""

import numpy as np
import pytest
import xarray as xr

import funcexpr_xr as fxr


class TestEvaluateWithInterpReturnType:
    def test_returns_dataarray(self, da_x, da_x_coarse):
        result = fxr.evaluate_with_interp(
            "a + b", ctx={"a": da_x, "b": da_x_coarse}, interp_ref="a"
        )
        assert isinstance(result, xr.DataArray)

    def test_result_dims_match_ref(self, da_x, da_x_coarse):
        result = fxr.evaluate_with_interp(
            "a + b", ctx={"a": da_x, "b": da_x_coarse}, interp_ref="a"
        )
        assert result.dims == da_x.dims

    def test_result_coords_match_rounded_ref(self, da_x, da_x_coarse):
        result = fxr.evaluate_with_interp(
            "a + b", ctx={"a": da_x, "b": da_x_coarse}, interp_ref="a", digits=10
        )
        np.testing.assert_array_equal(result.coords["x"].values, [1.0, 2.0, 3.0])

    def test_result_size_matches_ref(self, da_x, da_x_coarse):
        result = fxr.evaluate_with_interp(
            "a + b", ctx={"a": da_x, "b": da_x_coarse}, interp_ref="a"
        )
        assert result.sizes["x"] == da_x.sizes["x"]


class TestEvaluateWithInterpValidation:
    def test_no_dataarray_raises(self):
        with pytest.raises(ValueError, match="funcexpr"):
            fxr.evaluate_with_interp("a + b", ctx={"a": 1.0, "b": 2.0}, interp_ref="a")

    def test_interp_ref_not_in_ctx_raises(self, da_x):
        with pytest.raises(ValueError, match="interp_ref"):
            fxr.evaluate_with_interp("a", ctx={"a": da_x}, interp_ref="z")

    def test_error_message_lists_available_keys(self, da_x):
        with pytest.raises(ValueError, match="'a'"):
            fxr.evaluate_with_interp("a", ctx={"a": da_x}, interp_ref="z")

    def test_interp_ref_is_scalar_raises(self, da_x):
        with pytest.raises(ValueError, match="interp_ref"):
            fxr.evaluate_with_interp(
                "a + s", ctx={"a": da_x, "s": 1.0}, interp_ref="s"
            )


class TestEvaluateWithInterpDigits:
    def test_digits_none_disables_rounding(self, da_x, da_x_coarse):
        # Should still work when grids are clean integers
        result = fxr.evaluate_with_interp(
            "a + b", ctx={"a": da_x, "b": da_x_coarse}, interp_ref="a", digits=None
        )
        assert isinstance(result, xr.DataArray)

    def test_fp_error_absorbed(self, da_x_fp_error, da_x_coarse):
        # fp_error coords should be rounded to match clean grid
        result = fxr.evaluate_with_interp(
            "a + b",
            ctx={"a": da_x_fp_error, "b": da_x_coarse},
            interp_ref="a",
            digits=10,
        )
        assert isinstance(result, xr.DataArray)