from typing import Hashable

import xarray as xr


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