"""Tests for funcexpr_xr.rounding.coords.round_coords."""

import numpy as np
import pytest
import xarray as xr

from funcexpr_xr.rounding.coords import round_coords


class TestRoundCoords:
    def test_float64_coords_are_rounded(self):
        da = xr.DataArray(
            [1.0],
            dims=["x"],
            coords={"x": np.array([1.123456789012345])},
        )
        result = round_coords(da, digits=6)
        assert result.coords["x"].values[0] == pytest.approx(1.123457)

    def test_float32_is_cast_to_float64(self):
        da = xr.DataArray(
            [1.0],
            dims=["x"],
            coords={"x": np.array([1.5], dtype=np.float32)},
        )
        result = round_coords(da, digits=4)
        assert result.coords["x"].values.dtype == np.float64

    def test_fp_representation_error_absorbed(self):
        """0.1 + 0.2 != 0.3 in float64; rounding should canonicalize them."""
        coord_a = np.array([0.1 + 0.2])
        coord_b = np.array([0.3])
        assert not np.array_equal(coord_a, coord_b)

        da_a = xr.DataArray([1.0], dims=["x"], coords={"x": coord_a})
        da_b = xr.DataArray([1.0], dims=["x"], coords={"x": coord_b})

        rounded_a = round_coords(da_a, digits=10)
        rounded_b = round_coords(da_b, digits=10)

        assert np.array_equal(
            rounded_a.coords["x"].values,
            rounded_b.coords["x"].values,
        )

    def test_digits_zero_rounds_to_integer(self):
        da = xr.DataArray(
            [1.0],
            dims=["x"],
            coords={"x": np.array([1.6])},
        )
        result = round_coords(da, digits=0)
        assert result.coords["x"].values[0] == 2.0

    def test_data_values_unchanged(self, da_x):
        result = round_coords(da_x, digits=10)
        np.testing.assert_array_equal(result.values, da_x.values)

    def test_dims_unchanged(self, da_x):
        result = round_coords(da_x, digits=10)
        assert result.dims == da_x.dims

    def test_string_coord_raises(self):
        da = xr.DataArray(
            [1.0, 2.0],
            dims=["x"],
            coords={"x": ["a", "b"]},
        )
        with pytest.raises((ValueError, TypeError)):
            round_coords(da, digits=4)