import xarray as xr

from ._validators import apply_rounding, validate_dims


def outer_align(
    arrays: dict[str, xr.DataArray],
    digits: int | None = 10,
) -> dict[str, xr.DataArray]:
    """Align DataArrays by taking the union of coordinate values, filling gaps
    with NaN.

    This mirrors xarray's ``join='outer'`` behavior. The caller accepts
    responsibility for NaN values in the result.

    Args:
        arrays: Dict of DataArrays to align.
        digits: Number of decimal places to round coordinate values to before
            alignment. Defaults to 10. Pass None to disable rounding.

    Returns:
        A new dict with the same keys, where all DataArrays share the union
        of coordinate values. Missing values are filled with NaN.

    Raises:
        ValueError: If dims differ across any DataArrays.
    """
    arrays = apply_rounding(arrays, digits)
    validate_dims(arrays)

    keys = list(arrays.keys())
    values = list(arrays.values())

    aligned_values = xr.align(*values, join="outer")

    return dict(zip(keys, aligned_values))