from typing import Protocol
import xarray as xr

class AlignmentStrategy(Protocol):
    def __call__(
        self,
        arrays: dict[str, xr.DataArray],
    ) -> dict[str, xr.DataArray]:
        ...