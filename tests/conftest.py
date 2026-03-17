"""Shared fixtures for funcexpr_xr tests."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def da_x():
    """1D DataArray with x coords [1.0, 2.0, 3.0]."""
    return xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=["x"],
        coords={"x": [1.0, 2.0, 3.0]},
    )


@pytest.fixture
def da_x2():
    """1D DataArray with x coords [1.0, 2.0, 3.0], different values."""
    return xr.DataArray(
        [4.0, 5.0, 6.0],
        dims=["x"],
        coords={"x": [1.0, 2.0, 3.0]},
    )


@pytest.fixture
def da_x_shifted():
    """1D DataArray with x coords [2.0, 3.0, 4.0] — partial overlap with da_x."""
    return xr.DataArray(
        [10.0, 20.0, 30.0],
        dims=["x"],
        coords={"x": [2.0, 3.0, 4.0]},
    )


@pytest.fixture
def da_x_nooverlap():
    """1D DataArray with x coords [10.0, 20.0] — no overlap with da_x."""
    return xr.DataArray(
        [10.0, 20.0],
        dims=["x"],
        coords={"x": [10.0, 20.0]},
    )


@pytest.fixture
def da_xy():
    """2D DataArray with dims [time, x]."""
    return xr.DataArray(
        np.ones((2, 3)),
        dims=["time", "x"],
        coords={"time": [0.0, 1.0], "x": [1.0, 2.0, 3.0]},
    )


@pytest.fixture
def da_yx():
    """2D DataArray with dims [x, time] — reversed order vs da_xy."""
    return xr.DataArray(
        np.ones((3, 2)),
        dims=["x", "time"],
        coords={"x": [1.0, 2.0, 3.0], "time": [0.0, 1.0]},
    )


@pytest.fixture
def da_x_fp_error():
    """1D DataArray whose x coord has a tiny floating-point representation error."""
    coords = np.array([1.0, 2.0, 3.0]) + 1e-12
    return xr.DataArray(
        [1.0, 2.0, 3.0],
        dims=["x"],
        coords={"x": coords},
    )


@pytest.fixture
def da_x_coarse():
    """1D DataArray with coarser x grid [1.0, 3.0] for interp tests."""
    return xr.DataArray(
        [10.0, 30.0],
        dims=["x"],
        coords={"x": [1.0, 3.0]},
    )