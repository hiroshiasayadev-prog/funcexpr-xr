from collections.abc import Mapping
from typing import Callable

import numpy as np
import xarray as xr

import funcexpr
from .rounding import round_coords
from ._ref_utils import extract_da_ctx, validate_ref


def evaluate_with_interp(
    expr: str,
    ctx: Mapping[str, xr.DataArray | np.ndarray | int | float | complex],
    interp_ref: str,
    funcs: Mapping[str, Callable] | None = None,
    digits: int | None = 10,
) -> xr.DataArray:
    """Evaluate a Python expression string, interpolating all DataArrays onto
    a common grid defined by *interp_ref*.

    All DataArrays in *ctx* are interpolated onto the coordinate grid of the
    DataArray named *interp_ref*, after optionally rounding all coordinate
    values to *digits* decimal places. The result carries the (rounded)
    coordinates of *interp_ref*.

    This function is intended for cases where DataArrays share the same dims
    but have different coordinate values (e.g. different sampling grids), and
    interpolation onto a common grid is acceptable.

    Note:
        The returned DataArray carries the rounded coordinates of *interp_ref*,
        not the original pre-rounding values. This is by design: rounding
        defines the canonical grid, so returning the original coordinates would
        be inconsistent. See the ``digits`` parameter for details.

    Args:
        expr:
            A Python expression string (e.g. ``"a + b * c"``).
        ctx:
            A mapping from variable name to value. Accepts
            ``xr.DataArray``, ``np.ndarray``, ``int``, ``float``, or
            ``complex``. At least one DataArray must be present, and
            *interp_ref* must be a key in *ctx*.
        interp_ref:
            The name of the DataArray in *ctx* whose coordinate grid is
            used as the interpolation target. All other DataArrays are
            interpolated onto this grid. Required; there is no default to
            avoid ambiguity.
        funcs:
            Optional mapping from function name to callable, forwarded
            to ``funcexpr.evaluate``.
        digits:
            Number of decimal places to round coordinate values to before
            interpolation. Defaults to 10 to absorb floating-point
            representation errors. Pass None to disable rounding.

    Returns:
        The result as an ``xr.DataArray`` carrying the (rounded) dims and
        coordinates of *interp_ref*.

    Raises:
        ValueError: If *ctx* contains no DataArrays.
        ValueError: If *interp_ref* is not a key in *ctx*.
        ValueError: If *interp_ref* does not refer to a DataArray.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> import xeval
        >>> da1 = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})
        >>> da2 = xr.DataArray([10.0, 30.0], dims=["x"], coords={"x": [1.0, 3.0]})
        >>> xeval.evaluate_with_interp("a + b", ctx={"a": da1, "b": da2}, interp_ref="a")
        <xarray.DataArray (x: 3)>
        array([11., 22., 33.])
        Coordinates:
          * x  (x) float64 1.0 2.0 3.0
    """
    da_ctx, other_ctx = extract_da_ctx(ctx, "evaluate_with_interp")
    validate_ref(ctx, da_ctx, interp_ref, "interp_ref")

    # Round all DataArrays first, then use the rounded ref as the target grid.
    if digits is not None:
        da_ctx = {k: round_coords(v, digits) for k, v in da_ctx.items()}

    ref = da_ctx[interp_ref]

    # Interpolate all DataArrays (including ref itself, which is a no-op)
    # onto the ref grid.
    interpolated = {k: v.interp_like(ref) for k, v in da_ctx.items()}

    # Flatten to ndarrays for funcexpr.
    flat_ctx = {k: v.values for k, v in interpolated.items()} | other_ctx

    result_values = funcexpr.evaluate(expr, ctx=flat_ctx, funcs=funcs)

    return xr.DataArray(result_values, dims=ref.dims, coords=ref.coords)