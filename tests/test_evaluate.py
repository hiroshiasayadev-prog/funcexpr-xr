"""Tests for fxr.evaluate."""

import numpy as np
import pytest
import xarray as xr

import funcexpr_xr as fxr
from funcexpr_xr.alignment import register


class TestEvaluateReturnType:
    def test_returns_dataarray(self, da_x, da_x2):
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": da_x2})
        assert isinstance(result, xr.DataArray)

    def test_result_dims_match(self, da_x, da_x2):
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": da_x2})
        assert result.dims == ("x",)

    def test_result_coords_match_aligned(self, da_x, da_x2):
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": da_x2})
        np.testing.assert_array_equal(result.coords["x"].values, [1.0, 2.0, 3.0])


class TestEvaluateCtxValidation:
    def test_no_dataarray_raises(self):
        with pytest.raises(ValueError, match="funcexpr"):
            fxr.evaluate("a + b", ctx={"a": 1.0, "b": 2.0})

    def test_error_message_suggests_funcexpr(self):
        with pytest.raises(ValueError, match="funcexpr.evaluate"):
            fxr.evaluate("a + b", ctx={"a": np.array([1.0]), "b": np.array([2.0])})


class TestEvaluateAlignment:
    def test_unknown_alignment_raises(self, da_x):
        with pytest.raises(ValueError, match="unknown_strategy"):
            fxr.evaluate("a", ctx={"a": da_x}, alignment="unknown_strategy")

    def test_error_message_lists_registered_strategies(self, da_x):
        with pytest.raises(ValueError, match="exact"):
            fxr.evaluate("a", ctx={"a": da_x}, alignment="unknown_strategy")

    def test_custom_strategy_via_register(self, da_x, da_x2):
        def passthrough(arrays, digits=10):
            return arrays

        register("passthrough", passthrough)
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": da_x2}, alignment="passthrough")
        assert isinstance(result, xr.DataArray)

    def test_inner_alignment(self, da_x, da_x_shifted):
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": da_x_shifted}, alignment="inner")
        assert result.sizes["x"] == 2

    def test_outer_alignment(self, da_x, da_x_shifted):
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": da_x_shifted}, alignment="outer")
        assert result.sizes["x"] == 4


class TestEvaluateMixedCtx:
    def test_scalar_in_ctx(self, da_x):
        result = fxr.evaluate("a * 2.0", ctx={"a": da_x})
        assert isinstance(result, xr.DataArray)

    def test_ndarray_in_ctx(self, da_x):
        b = np.array([1.0, 2.0, 3.0])
        result = fxr.evaluate("a + b", ctx={"a": da_x, "b": b})
        assert isinstance(result, xr.DataArray)

class TestEvaluateCoordInjection:
    def test_coord_available_in_expr(self, da_x):
        # x coord [1.0, 2.0, 3.0] should be injected automatically
        result = fxr.evaluate("a * x", ctx={"a": da_x})
        np.testing.assert_array_almost_equal(result.values, [1.0, 4.0, 9.0])

    def test_coord_injection_with_digits(self, da_x):
        # digits=10 should round float coords before injection
        result = fxr.evaluate("a * x", ctx={"a": da_x}, digits=10)
        np.testing.assert_array_almost_equal(result.values, [1.0, 4.0, 9.0])

    def test_coord_injection_digits_none(self, da_x):
        result = fxr.evaluate("a * x", ctx={"a": da_x}, digits=None)
        np.testing.assert_array_almost_equal(result.values, [1.0, 4.0, 9.0])

    def test_explicit_ctx_takes_precedence(self, da_x):
        # user-supplied x should win over coord injection
        result = fxr.evaluate("a * x", ctx={"a": da_x, "x": np.array([10.0, 10.0, 10.0])})
        np.testing.assert_array_almost_equal(result.values, [10.0, 20.0, 30.0])

    def test_2d_coord_injection(self, da_xy):
        # time coord should broadcast correctly over (time, x) array
        result = fxr.evaluate("a * time", ctx={"a": da_xy})
        assert isinstance(result, xr.DataArray)
        # time=[0,1], x=[1,2,3] -> row0 * 0.0, row1 * 1.0
        np.testing.assert_array_almost_equal(result.values[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result.values[1], [1.0, 1.0, 1.0])

    def test_2d_coord_injection_x_axis(self, da_xy):
        result = fxr.evaluate("a * x", ctx={"a": da_xy})
        assert isinstance(result, xr.DataArray)
        # x=[1,2,3] broadcast over (time, x) -> each row multiplied by [1,2,3]
        np.testing.assert_array_almost_equal(result.values[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.values[1], [1.0, 2.0, 3.0])