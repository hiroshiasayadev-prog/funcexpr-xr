import xarray as xr

from ._validators import validate_dims


def outer_align(arrays: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Align DataArrays by taking the union of coordinate values, filling gaps
    with NaN.

    This mirrors xarray's ``join='outer'`` behavior. The caller accepts
    responsibility for NaN values in the result.

    Args:
        arrays: Dict of DataArrays to align.

    Returns:
        A new dict with the same keys, where all DataArrays share the union
        of coordinate values. Missing values are filled with NaN.

    Raises:
        ValueError: If dims differ across any DataArrays.
    """
    validate_dims(arrays)

    keys = list(arrays.keys())
    values = list(arrays.values())

    aligned_values = xr.align(*values, join="outer")

    return dict(zip(keys, aligned_values))