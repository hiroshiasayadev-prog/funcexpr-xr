import numpy as np
import xarray as xr

from ._validators import apply_rounding, validate_dims


def exact_align(
    arrays: dict[str, xr.DataArray],
    digits: int | None = 10,
) -> dict[str, xr.DataArray]:
    """Require that all DataArrays have identical dims and coordinate values.

    This is the strictest alignment mode and the default. No reindexing or
    interpolation is performed; if any mismatch is detected after optional
    rounding, a ValueError is raised immediately.

    Args:
        arrays: Dict of DataArrays to validate.
        digits: Number of decimal places to round coordinate values to before
            comparison. Defaults to 10 to absorb floating-point representation
            errors. Pass None to disable rounding.

    Returns:
        The rounded dict (or original if digits is None), unchanged in shape.

    Raises:
        ValueError: If dims differ across any DataArrays.
        ValueError: If coordinate values differ for any dim after rounding.
    """
    arrays = apply_rounding(arrays, digits)
    validate_dims(arrays)

    items = list(arrays.items())
    ref_name, ref = items[0]

    for name, da in items[1:]:
        for dim in ref.dims:
            ref_coord = ref.coords[dim].values
            da_coord = da.coords[dim].values
            if ref_coord.shape != da_coord.shape or not np.array_equal(ref_coord, da_coord):
                raise ValueError(
                    f"Coordinate mismatch on dim '{dim}': "
                    f"'{ref_name}' has {ref_coord.tolist()}, "
                    f"but '{name}' has {da_coord.tolist()}. "
                    "Use alignment='inner' or alignment='outer' to handle "
                    "mismatched coordinates automatically."
                )

    return arrays