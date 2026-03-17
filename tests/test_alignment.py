"""Tests for fxr.alignment strategies: exact, inner, outer."""

import numpy as np
import pytest
import xarray as xr

from funcexpr_xr.alignment.exact import exact_align
from funcexpr_xr.alignment.inner import inner_align
from funcexpr_xr.alignment.outer import outer_align


class TestExactAlign:
    def test_matching_coords_returns_unchanged(self, da_x, da_x2):
        result = exact_align({"a": da_x, "b": da_x2})
        np.testing.assert_array_equal(result["a"].values, da_x.values)
        np.testing.assert_array_equal(result["b"].values, da_x2.values)

    def test_coord_mismatch_raises(self, da_x, da_x_shifted):
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            exact_align({"a": da_x, "b": da_x_shifted})

    def test_error_message_contains_dim_name(self, da_x, da_x_shifted):
        with pytest.raises(ValueError, match="'x'"):
            exact_align({"a": da_x, "b": da_x_shifted})

    def test_error_message_contains_array_names(self, da_x, da_x_shifted):
        with pytest.raises(ValueError, match="'a'"):
            exact_align({"a": da_x, "b": da_x_shifted})

    def test_dims_mismatch_raises(self, da_x, da_xy):
        with pytest.raises(ValueError, match="dims"):
            exact_align({"a": da_x, "b": da_xy})

    def test_fp_error_absorbed_by_digits(self, da_x, da_x_fp_error):
        # With default digits=10, tiny fp errors should be absorbed
        result = exact_align({"a": da_x, "b": da_x_fp_error}, digits=10)
        np.testing.assert_array_equal(
            result["a"].coords["x"].values,
            result["b"].coords["x"].values,
        )

    def test_fp_error_raises_when_digits_none(self, da_x, da_x_fp_error):
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            exact_align({"a": da_x, "b": da_x_fp_error}, digits=None)

    def test_single_array_passes(self, da_x):
        result = exact_align({"a": da_x})
        assert "a" in result


class TestInnerAlign:
    def test_matching_coords_returns_unchanged(self, da_x, da_x2):
        result = inner_align({"a": da_x, "b": da_x2})
        assert result["a"].sizes["x"] == 3

    def test_partial_overlap_returns_intersection(self, da_x, da_x_shifted):
        # da_x: [1,2,3], da_x_shifted: [2,3,4] -> intersection: [2,3]
        result = inner_align({"a": da_x, "b": da_x_shifted})
        assert result["a"].sizes["x"] == 2
        np.testing.assert_array_equal(result["a"].coords["x"].values, [2.0, 3.0])

    def test_empty_intersection_raises(self, da_x, da_x_nooverlap):
        with pytest.raises(ValueError, match="empty"):
            inner_align({"a": da_x, "b": da_x_nooverlap})

    def test_dims_mismatch_raises(self, da_x, da_xy):
        with pytest.raises(ValueError, match="dims"):
            inner_align({"a": da_x, "b": da_xy})

    def test_keys_preserved(self, da_x, da_x_shifted):
        result = inner_align({"a": da_x, "b": da_x_shifted})
        assert set(result.keys()) == {"a", "b"}


class TestOuterAlign:
    def test_matching_coords_returns_unchanged(self, da_x, da_x2):
        result = outer_align({"a": da_x, "b": da_x2})
        assert result["a"].sizes["x"] == 3

    def test_partial_overlap_returns_union(self, da_x, da_x_shifted):
        # da_x: [1,2,3], da_x_shifted: [2,3,4] -> union: [1,2,3,4]
        result = outer_align({"a": da_x, "b": da_x_shifted})
        assert result["a"].sizes["x"] == 4

    def test_missing_values_filled_with_nan(self, da_x, da_x_shifted):
        result = outer_align({"a": da_x, "b": da_x_shifted})
        # da_x_shifted has no value at x=1.0
        assert np.isnan(result["b"].sel(x=1.0).values)

    def test_dims_mismatch_raises(self, da_x, da_xy):
        with pytest.raises(ValueError, match="dims"):
            outer_align({"a": da_x, "b": da_xy})

    def test_keys_preserved(self, da_x, da_x_shifted):
        result = outer_align({"a": da_x, "b": da_x_shifted})
        assert set(result.keys()) == {"a", "b"}