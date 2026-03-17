import xarray as xr

from ._validators import apply_rounding, validate_dims


def inner_align(
    arrays: dict[str, xr.DataArray],
    digits: int | None = 10,
) -> dict[str, xr.DataArray]:
    """Align DataArrays by intersecting coordinate values across all arrays.

    Only coordinate values present in all DataArrays are retained. Unlike
    xarray's default behavior, an empty intersection (size 0 on any dim) is
    treated as an error rather than silently returning an empty array.

    Args:
        arrays: Dict of DataArrays to align.
        digits: Number of decimal places to round coordinate values to before
            alignment. Defaults to 10. Pass None to disable rounding.

    Returns:
        A new dict with the same keys, where all DataArrays share the
        intersected coordinate values.

    Raises:
        ValueError: If dims differ across any DataArrays.
        ValueError: If the intersection on any dim is empty.
    """
    arrays = apply_rounding(arrays, digits)
    validate_dims(arrays)

    keys = list(arrays.keys())
    values = list(arrays.values())

    aligned_values = xr.align(*values, join="inner")

    # Guard against silent empty intersection.
    ref_dims = values[0].dims
    for dim in ref_dims:
        size = aligned_values[0].sizes[dim]
        if size == 0:
            raise ValueError(
                f"Coordinate intersection on dim '{dim}' is empty. "
                "No overlapping coordinate values exist across all DataArrays. "
                "Use alignment='outer' to allow NaN-filled union instead."
            )

    return dict(zip(keys, aligned_values))