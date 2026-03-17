from typing import Hashable

import xarray as xr

from ..rounding import round_coords


def apply_rounding(
    arrays: dict[str, xr.DataArray],
    digits: int | None,
) -> dict[str, xr.DataArray]:
    """Round coordinates of all DataArrays if *digits* is not None.

    Args:
        arrays: Dict of DataArrays to process.
        digits: Number of decimal places to round to, or None to skip.

    Returns:
        A new dict with rounded coordinates, or the original dict unchanged.
    """
    if digits is None:
        return arrays
    return {k: round_coords(v, digits) for k, v in arrays.items()}


def validate_dims(arrays: dict[str, xr.DataArray]) -> tuple[Hashable, ...]:
    """Validate that all DataArrays share the same dims (name and order).

    Args:
        arrays: Dict of DataArrays to validate.

    Returns:
        The common dims tuple.

    Raises:
        ValueError: If dims differ across any two DataArrays.
    """
    it = iter(arrays.items())
    ref_name, ref = next(it)
    ref_dims = ref.dims

    for name, da in it:
        if da.dims != ref_dims:
            raise ValueError(
                f"Dimension mismatch: '{ref_name}' has dims {ref_dims!r}, "
                f"but '{name}' has dims {da.dims!r}. "
                "All DataArrays must share the same dims in the same order."
            )

    return ref_dims