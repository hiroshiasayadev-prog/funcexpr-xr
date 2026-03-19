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


def build_coord_ctx(
    ref: xr.DataArray,
    flat_ctx: dict,
    digits: int | None,
) -> dict:
    """Build a coord variable mapping from a reference DataArray.

    Extracts coordinate arrays from *ref* and returns them as a dict
    suitable for merging into the flat ctx passed to ``funcexpr.evaluate``.
    Coordinates whose names are already present in *flat_ctx* are skipped,
    so explicit user-supplied values always take precedence.

    Floating-point coordinates are rounded to *digits* decimal places when
    *digits* is not ``None``, matching the rounding applied during alignment.
    Non-floating coordinates are passed through as-is.

    Args:
        ref:
            A reference DataArray whose coordinates are extracted. Typically
            the first aligned DataArray; all aligned arrays share the same
            coords by this point.
        flat_ctx:
            The already-flattened ctx dict (DataArrays resolved to ndarrays,
            scalars included). Used only to check for name conflicts.
        digits:
            Number of decimal places to round floating-point coordinate
            values. Pass ``None`` to disable rounding.

    Returns:
        A dict mapping coordinate name to ``np.ndarray``. Only coordinates
        not already present in *flat_ctx* are included.
    """
    import numpy as np

    coord_ctx = {}
    for coord_name, coord in ref.coords.items():
        if coord_name not in flat_ctx:
            values = coord.values
            if digits is not None and np.issubdtype(values.dtype, np.floating):
                values = np.round(values, digits)
            # reshape for broadcasting: (n,) -> proper shape for ref
            if coord_name in ref.dims:
                axis = ref.dims.index(coord_name)
                shape = [1] * ref.ndim
                shape[axis] = -1
                values = values.reshape(shape)
            coord_ctx[coord_name] = values
    return coord_ctx