import numpy as np
import xarray as xr


def round_coords(da: xr.DataArray, digits: int) -> xr.DataArray:
    """Return a new DataArray with all coordinate values cast to float64 and
    rounded to *digits* decimal places.

    All coordinates are cast to ``np.float64`` before rounding. This is
    intentional: the primary use case is absorbing floating-point
    representation errors that arise when coordinate values are produced by
    different code paths (e.g. parsed from CSV vs computed via arithmetic).
    Two values that are semantically identical may differ in their binary
    representation (e.g. ``0.1 + 0.2 != 0.3`` in float64), and rounding after
    a uniform cast to float64 is the simplest way to canonicalize them.

    If a coordinate cannot be cast to float64 (e.g. string labels), an error
    is raised at the ``astype`` call. Coordinates that cannot be represented
    as float64 are not a supported use case for rounding.

    Args:
        da: The DataArray whose coordinates to round.
        digits: Number of decimal places to round to.

    Returns:
        A new DataArray with float64 rounded coordinates and identical data.
    """
    new_coords = {}
    for dim in da.dims:
        coord = da.coords[dim]
        new_coords[dim] = np.round(coord.values.astype(np.float64), digits)

    return xr.DataArray(da.values, dims=da.dims, coords=new_coords)