from collections.abc import Mapping
from typing import Callable

import numpy as np
import xarray as xr

import funcexpr
from .rounding import round_coords
from ._ref_utils import extract_da_ctx, validate_ref, build_coord_ctx


def evaluate_with_reindex(
    expr: str,
    ctx: Mapping[str, xr.DataArray | np.ndarray | int | float | complex],
    reindex_ref: str,
    interp: bool = False,
    funcs: Mapping[str, Callable] | None = None,
    digits: int | None = 10,
) -> xr.DataArray:
    """Evaluate a Python expression string, reindexing all DataArrays onto
    a common grid defined by *reindex_ref*.

    All DataArrays in *ctx* are reindexed onto the coordinate grid of the
    DataArray named *reindex_ref*. Grid points that exist in both the source
    and the target are taken as-is; points that exist only in the target are
    filled with NaN.

    When *interp=True*, a ``interp_like`` pass is performed first, and the
    reindex result is used to fill any NaN values that the interpolation could
    not resolve (e.g. because the source DataArray contained NaN at grid-aligned
    points). This ensures that grid-aligned values are always preserved exactly,
    even when NaN values in the source would otherwise corrupt the interpolation.

    Args:
        expr:
            A Python expression string (e.g. ``"a + b * c"``).
        ctx:
            A mapping from variable name to value. Accepts
            ``xr.DataArray``, ``np.ndarray``, ``int``, ``float``, or
            ``complex``. At least one DataArray must be present, and
            *reindex_ref* must be a key in *ctx*.
            Coordinate arrays of the aligned DataArrays are 
            automatically injected into the expression namespace 
            under their coordinate names,
            unless a key of the same name already exists in ctx.
        reindex_ref:
            The name of the DataArray in *ctx* whose coordinate grid is
            used as the reindex target. Required; there is no default to
            avoid ambiguity.
        interp:
            If True, interpolate onto the target grid first, then fill
            remaining NaN values with the reindexed result. Grid-aligned
            points are always taken from the source as-is. Defaults to False.
        funcs:
            Optional mapping from function name to callable, forwarded
            to ``funcexpr.evaluate``.
        digits:
            Number of decimal places to round coordinate values to before
            reindexing. Defaults to 10. Pass None to disable rounding.
            Also applied to coordinate values injected into the expression namespace.

    Returns:
        The result as an ``xr.DataArray`` carrying the (rounded) dims and
        coordinates of *reindex_ref*.

    Raises:
        ValueError: If *ctx* contains no DataArrays.
        ValueError: If *reindex_ref* is not a key in *ctx*.
        ValueError: If *reindex_ref* does not refer to a DataArray.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> import funcexpr_xr as fxr
        >>> da1 = xr.DataArray(
        ...     [1.0, np.nan, 3.0],
        ...     dims=["x"], coords={"x": [1.0, 2.0, 3.0]}
        ... )
        >>> da2 = xr.DataArray(
        ...     [10.0, 30.0],
        ...     dims=["x"], coords={"x": [1.0, 3.0]}
        ... )
        >>> # reindex only: x=2.0 has no match in da2, so NaN
        >>> fxr.evaluate_with_reindex("a + b", ctx={"a": da1, "b": da2}, reindex_ref="a")
        >>> # interp=True: x=2.0 in da2 is interpolated to 20.0; da1's NaN at x=2.0 stays NaN
        >>> fxr.evaluate_with_reindex("a + b", ctx={"a": da1, "b": da2}, reindex_ref="a", interp=True)
    """
    da_ctx, other_ctx = extract_da_ctx(ctx, "evaluate_with_reindex")
    validate_ref(ctx, da_ctx, reindex_ref, "reindex_ref")

    if digits is not None:
        da_ctx = {k: round_coords(v, digits) for k, v in da_ctx.items()}

    ref = da_ctx[reindex_ref]

    if interp:
        # Interpolate first, then fill NaNs that interpolation couldn't resolve
        # (e.g. source NaNs at grid-aligned points) with the reindexed values.
        interpolated = {k: v.interp_like(ref) for k, v in da_ctx.items()}
        reindexed = {k: v.reindex_like(ref) for k, v in da_ctx.items()}
        aligned = {k: interpolated[k].fillna(reindexed[k]) for k in da_ctx}
    else:
        aligned = {k: v.reindex_like(ref) for k, v in da_ctx.items()}

    flat_ctx = {k: v.values for k, v in aligned.items()} | other_ctx
    coord_ctx = build_coord_ctx(ref, flat_ctx, digits)
    flat_ctx = coord_ctx | flat_ctx

    result_values = funcexpr.evaluate(expr, ctx=flat_ctx, funcs=funcs)

    return xr.DataArray(result_values, dims=ref.dims, coords=ref.coords)