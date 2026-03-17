# funcexpr-xr

`xarray.DataArray` support layer for [funcexpr](https://github.com/yourname/funcexpr).

```
funcexpr-xr       xarray DataArray — axes alignment, interpolation
  ↓
funcexpr    callable registration, type normalization
  ↓
numexpr     fast array evaluation
```

## Installation

```bash
pip install funcexpr_xr
```

## Usage

### Basic

Pass `xarray.DataArray` objects directly in `ctx`. `funcexpr-xr` handles alignment and delegates to `funcexpr` with raw ndarrays. The result is returned as a `DataArray` with aligned coordinates.

```python
import numpy as np
import xarray as xr
import funcexpr_xr as fxr

da1 = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})
da2 = xr.DataArray([4.0, 5.0, 6.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})

result = fxr.evaluate("a + b * 2", ctx={"a": da1, "b": da2})
# <xarray.DataArray (x: 3)>
# array([ 9., 12., 15.])
# Coordinates:
#   * x  (x) float64 1.0 2.0 3.0
```

Scalars and ndarrays can be mixed in `ctx` alongside DataArrays.

```python
result = fxr.evaluate("a * scale + offset", ctx={"a": da1, "scale": 2.0, "offset": 1.0})
```

Custom callables work the same way as in `funcexpr`.

```python
def clip(x, lo, hi):
    return np.clip(x, lo, hi)

result = fxr.evaluate(
    "clip(a, 0.0, 2.5) + b",
    ctx={"a": da1, "b": da2},
    funcs={"clip": clip},
)
```

### Alignment strategies

`fxr.evaluate` requires all DataArrays to share the same dims and coordinate values. The `alignment` parameter controls what happens when coordinates don't match.

```python
fxr.evaluate("a + b", ctx={"a": da1, "b": da2}, alignment="exact")  # default
fxr.evaluate("a + b", ctx={"a": da1, "b": da2}, alignment="inner")
fxr.evaluate("a + b", ctx={"a": da1, "b": da2}, alignment="outer")
```

| Strategy | Behavior |
|---|---|
| `"exact"` | Requires identical coordinate values. Raises on any mismatch. |
| `"inner"` | Intersects coordinate values. Raises if intersection is empty. |
| `"outer"` | Unions coordinate values, fills gaps with NaN. |

### Floating-point coordinate errors

Coordinates loaded from CSV or Excel may differ in their binary float representation even when semantically identical (e.g. `0.1 + 0.2 != 0.3`). The `digits` parameter rounds all coordinates to a fixed number of decimal places before alignment.

```python
# coords from two different CSV files may not match exactly
result = fxr.evaluate("a + b", ctx={"a": da1, "b": da2}, digits=10)  # default
result = fxr.evaluate("a + b", ctx={"a": da1, "b": da2}, digits=None)  # disable
```

All coordinates are cast to `float64` before rounding. The returned DataArray carries the rounded coordinates.

### Interpolation

When DataArrays share the same dims but have different coordinate grids, use `evaluate_with_interp`. All DataArrays are interpolated onto the grid of `interp_ref`.

```python
da_fine   = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0, 4.0]})
da_coarse = xr.DataArray([10.0, 30.0],           dims=["x"], coords={"x": [1.0, 3.0]})

result = fxr.evaluate_with_interp(
    "a + b",
    ctx={"a": da_fine, "b": da_coarse},
    interp_ref="a",   # required — no default
    digits=10,
)
# result carries the (rounded) coordinates of da_fine
```

`interp_ref` is required. There is no default; the choice of reference grid is always explicit.

The returned DataArray carries the rounded coordinates of `interp_ref`, not the original pre-rounding values. Rounding defines the canonical grid.

### Reindex

When DataArrays have different coordinate grids and you want grid-aligned points taken as-is (rather than interpolated), use `evaluate_with_reindex`. Points that exist in both source and target are taken directly; points that exist only in the target are filled with NaN.

```python
da_fine   = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0, 4.0]})
da_coarse = xr.DataArray([10.0, 30.0],           dims=["x"], coords={"x": [1.0, 3.0]})

# reindex only: x=2.0 and x=4.0 have no match in da_coarse -> NaN
result = fxr.evaluate_with_reindex(
    "a + b",
    ctx={"a": da_fine, "b": da_coarse},
    reindex_ref="a",  # required — no default
)
```

When `interp=True`, interpolation is performed first and the reindexed values are used to fill any NaN that interpolation could not resolve. This guarantees that grid-aligned points are always taken from the source as-is, even when NaN values in the source would otherwise corrupt the interpolation result.

```python
# interp=True: non-matching points are interpolated;
# grid-aligned points are always taken from source exactly
result = fxr.evaluate_with_reindex(
    "a + b",
    ctx={"a": da_fine, "b": da_coarse},
    reindex_ref="a",
    interp=True,
)
```

This is particularly useful when source DataArrays contain NaN at some grid points — `interp_like` alone would propagate those NaN values into the interpolated result, but `evaluate_with_reindex(interp=True)` recovers any grid-aligned values that interpolation missed.

### Custom alignment strategies

Alignment strategies are stored in a registry and can be extended at runtime.

```python
from funcexpr_xr.alignment import register
import xarray as xr

def my_strategy(
    arrays: dict[str, xr.DataArray],
    digits: int | None = 10,
) -> dict[str, xr.DataArray]:
    ...

register("my_strategy", my_strategy)

fxr.evaluate("a + b", ctx={...}, alignment="my_strategy")
```

## Design

`funcexpr-xr` is intentionally a thin layer. It does not reimplement xarray's alignment or interpolation logic — it wraps `xr.align`, `xr.DataArray.interp_like`, and `xr.DataArray.reindex_like` directly. Custom alignment rules beyond what xarray provides are out of scope.

If your `ctx` contains no `DataArray` values, use `funcexpr.evaluate` directly.

## Dependencies

- `xarray`
- `funcexpr`
- `numexpr`
- `numpy`