from collections.abc import Mapping

import xarray as xr


def extract_da_ctx(
    ctx: Mapping,
    func_name: str,
) -> tuple[dict[str, xr.DataArray], dict]:
    """Split ctx into DataArrays and everything else.

    Raises:
        ValueError: If ctx contains no DataArrays.
    """
    da_ctx = {k: v for k, v in ctx.items() if isinstance(v, xr.DataArray)}
    other_ctx = {k: v for k, v in ctx.items() if not isinstance(v, xr.DataArray)}

    if not da_ctx:
        raise ValueError(
            f"ctx contains no xr.DataArray values. "
            f"fxr.{func_name}() requires at least one DataArray. "
            "For ndarray-only expressions, use funcexpr.evaluate() directly."
        )

    return da_ctx, other_ctx


def validate_ref(
    ctx: Mapping,
    da_ctx: dict[str, xr.DataArray],
    ref_name: str,
    param_name: str,
) -> None:
    """Validate that ref_name exists in ctx and points to a DataArray.

    Raises:
        ValueError: If ref_name is not in ctx.
        ValueError: If ref_name is not a DataArray.
    """
    if ref_name not in ctx:
        raise ValueError(
            f"{param_name}={ref_name!r} is not a key in ctx. "
            f"Available keys: {list(ctx.keys())}."
        )
    if ref_name not in da_ctx:
        raise ValueError(
            f"{param_name}={ref_name!r} is not a DataArray. "
            f"{param_name} must point to an xr.DataArray in ctx."
        )