from typing import Protocol
import xarray as xr


class AlignmentStrategy(Protocol):
    """Protocol for alignment strategies.

    Each strategy receives a dict of DataArrays and returns a new dict with
    the same keys, where all DataArrays are aligned to a common set of
    coordinates.

    The strategy is responsible for:
    - Validating that dims are consistent across all DataArrays.
    - Aligning coordinate values according to its own policy.
    - Raising ValueError for any condition it considers invalid.

    Scalars and ndarrays are handled by the caller and never passed here.
    """

    def __call__(
        self,
        arrays: dict[str, xr.DataArray],
        digits: int | None = 10,
    ) -> dict[str, xr.DataArray]:
        ...