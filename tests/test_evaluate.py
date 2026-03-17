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