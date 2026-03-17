"""Tests for funcexpr_xr.alignment._validators."""

import numpy as np
import pytest
import xarray as xr

from funcexpr_xr.alignment._validators import apply_rounding, validate_dims


class TestValidateDims:
    def test_matching_dims_returns_dims(self, da_x, da_x2):
        result = validate_dims({"a": da_x, "b": da_x2})
        assert result == ("x",)

    def test_single_array_passes(self, da_x):
        result = validate_dims({"a": da_x})
        assert result == ("x",)

    def test_mismatched_dim_names_raises(self, da_x, da_xy):
        with pytest.raises(ValueError, match="dims"):
            validate_dims({"a": da_x, "b": da_xy})

    def test_mismatched_dim_order_raises(self, da_xy, da_yx):
        with pytest.raises(ValueError, match="dims"):
            validate_dims({"a": da_xy, "b": da_yx})

    def test_error_message_contains_array_names(self, da_x, da_xy):
        with pytest.raises(ValueError, match="'a'"):
            validate_dims({"a": da_x, "b": da_xy})


class TestApplyRounding:
    def test_digits_none_returns_same_dict(self, da_x, da_x2):
        arrays = {"a": da_x, "b": da_x2}
        result = apply_rounding(arrays, digits=None)
        assert result is arrays

    def test_digits_rounds_coords(self):
        da = xr.DataArray(
            [1.0],
            dims=["x"],
            coords={"x": np.array([1.123456789])},
        )
        result = apply_rounding({"a": da}, digits=4)
        assert result["a"].coords["x"].values[0] == pytest.approx(1.1235)

    def test_all_arrays_rounded(self, da_x_fp_error, da_x):
        result = apply_rounding({"a": da_x_fp_error, "b": da_x}, digits=10)
        np.testing.assert_array_equal(
            result["a"].coords["x"].values,
            result["b"].coords["x"].values,
        )

    def test_keys_preserved(self, da_x, da_x2):
        arrays = {"a": da_x, "b": da_x2}
        result = apply_rounding(arrays, digits=10)
        assert set(result.keys()) == {"a", "b"}