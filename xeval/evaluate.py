from collections.abc import Mapping
from typing import Callable

import numpy as np
import xarray as xr

import funcexpr
from .alignment import ALIGNMENT_STRATEGIES, AlignmentStrategy


def evaluate(
    expr: str,
    ctx: Mapping[str, xr.DataArray | np.ndarray | int | float | complex],
    funcs: Mapping[str, Callable] | None = None,
    alignment: str = "exact",
    digits: int | None = 10,
) -> xr.DataArray:
    """Evaluate a Python expression string over xarray DataArrays.

    Extends ``funcexpr.evaluate`` by accepting ``xr.DataArray`` values in
    *ctx*. DataArrays are aligned according to *alignment* before being
    passed to funcexpr as raw ndarrays. The result is returned as a
    DataArray carrying the aligned coordinates.

    Args:
        expr:
            A Python expression string (e.g. ``"a + b * c"``).
        ctx:
            A mapping from variable name to value. Accepts
            ``xr.DataArray``, ``np.ndarray``, ``int``, ``float``, or
            ``complex``. At least one DataArray must be present.
        funcs:
            Optional mapping from function name to callable, forwarded
            to ``funcexpr.evaluate``.
        alignment:
            Strategy for aligning DataArray coordinates. One of
            ``"exact"`` (default), ``"inner"``, or ``"outer"``. Custom
            strategies can be registered via
            ``xeval.alignment.register()``.
        digits:
            Number of decimal places to round coordinate values to before
            alignment. Defaults to 10 to absorb floating-point
            representation errors. Pass None to disable rounding.

    Returns:
        The result as an ``xr.DataArray`` carrying the aligned dims and
        coordinates.

    Raises:
        ValueError: If *ctx* contains no DataArrays. Use
            ``funcexpr.evaluate()`` directly for ndarray-only expressions.
        ValueError: If *alignment* is not a registered strategy name.
        ValueError: If the DataArrays cannot be aligned under the chosen
            strategy (e.g. dim mismatch, empty intersection).
        TypeError: If a value in *ctx* or a callable return value cannot
            be normalized by funcexpr.

    Examples:
        >>> import numpy as np
        >>> import xarray as xr
        >>> import xeval
        >>> da1 = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [1, 2, 3]})
        >>> da2 = xr.DataArray([4.0, 5.0, 6.0], dims=["x"], coords={"x": [1, 2, 3]})
        >>> xeval.evaluate("a + b * 2", ctx={"a": da1, "b": da2})
        <xarray.DataArray (x: 3)>
        array([ 9., 12., 15.])
        Coordinates:
          * x  (x) int64 1 2 3
    """
    # Split ctx into DataArrays and everything else.
    da_ctx: dict[str, xr.DataArray] = {
        k: v for k, v in ctx.items() if isinstance(v, xr.DataArray)
    }
    other_ctx = {
        k: v for k, v in ctx.items() if not isinstance(v, xr.DataArray)
    }

    if not da_ctx:
        raise ValueError(
            "ctx contains no xr.DataArray values. "
            "xeval.evaluate() requires at least one DataArray. "
            "For ndarray-only expressions, use funcexpr.evaluate() directly."
        )

    if alignment not in ALIGNMENT_STRATEGIES:
        registered = list(ALIGNMENT_STRATEGIES.keys())
        raise ValueError(
            f"Unknown alignment strategy: {alignment!r}. "
            f"Registered strategies: {registered}. "
            "Use xeval.alignment.register() to add a custom strategy."
        )

    # Apply alignment to DataArrays only.
    strategy: AlignmentStrategy = ALIGNMENT_STRATEGIES[alignment]
    aligned = strategy(da_ctx, digits=digits)

    # Flatten DataArrays to ndarrays for funcexpr.
    flat_ctx = {k: v.values for k, v in aligned.items()} | other_ctx

    result_values = funcexpr.evaluate(expr, ctx=flat_ctx, funcs=funcs)

    # Reconstruct DataArray from aligned coords.
    # All aligned arrays share the same dims/coords by this point.
    ref = next(iter(aligned.values()))
    return xr.DataArray(result_values, dims=ref.dims, coords=ref.coords)