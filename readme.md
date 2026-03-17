# xeval

`xarray.DataArray` support layer for [funcexpr](https://github.com/yourname/funcexpr).

```
xeval       xarray DataArray — axes alignment, interpolation
  ↓
funcexpr    callable registration, type normalization
  ↓
numexpr     fast array evaluation
```

## Installation

```bash
pip install xeval
```

## Usage

### Basic

Pass `xarray.DataArray` objects directly in `ctx`. `xeval` handles alignment and delegates to `funcexpr` with raw ndarrays. The result is returned as a `DataArray` with aligned coordinates.

```python
import numpy as np
import xarray as xr
import xeval

da1 = xr.DataArray([1.0, 2.0, 3.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})
da2 = xr.DataArray([4.0, 5.0, 6.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0]})

result = xeval.evaluate("a + b * 2", ctx={"a": da1, "b": da2})
# <xarray.DataArray (x: 3)>
# array([ 9., 12., 15.])
# Coordinates:
#   * x  (x) float64 1.0 2.0 3.0
```

Scalars and ndarrays can be mixed in `ctx` alongside DataArrays.

```python
result = xeval.evaluate("a * scale + offset", ctx={"a": da1, "scale": 2.0, "offset": 1.0})
```

Custom callables work the same way as in `funcexpr`.

```python
def clip(x, lo, hi):
    return np.clip(x, lo, hi)

result = xeval.evaluate(
    "clip(a, 0.0, 2.5) + b",
    ctx={"a": da1, "b": da2},
    funcs={"clip": clip},
)
```

### Alignment strategies

`xeval.evaluate` requires all DataArrays to share the same dims and coordinate values. The `alignment` parameter controls what happens when coordinates don't match.

```python
xeval.evaluate("a + b", ctx={"a": da1, "b": da2}, alignment="exact")  # default
xeval.evaluate("a + b", ctx={"a": da1, "b": da2}, alignment="inner")
xeval.evaluate("a + b", ctx={"a": da1, "b": da2}, alignment="outer")
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
result = xeval.evaluate("a + b", ctx={"a": da1, "b": da2}, digits=10)  # default
result = xeval.evaluate("a + b", ctx={"a": da1, "b": da2}, digits=None)  # disable
```

All coordinates are cast to `float64` before rounding. The returned DataArray carries the rounded coordinates.

### Interpolation

When DataArrays share the same dims but have different coordinate grids, use `evaluate_with_interp`. All DataArrays are interpolated onto the grid of `interp_ref`.

```python
da_fine   = xr.DataArray([1.0, 2.0, 3.0, 4.0], dims=["x"], coords={"x": [1.0, 2.0, 3.0, 4.0]})
da_coarse = xr.DataArray([10.0, 30.0],           dims=["x"], coords={"x": [1.0, 3.0]})

result = xeval.evaluate_with_interp(
    "a + b",
    ctx={"a": da_fine, "b": da_coarse},
    interp_ref="a",   # required — no default
    digits=10,
)
# result carries the (rounded) coordinates of da_fine
```

`interp_ref` is required. There is no default; the choice of reference grid is always explicit.

The returned DataArray carries the rounded coordinates of `interp_ref`, not the original pre-rounding values. Rounding defines the canonical grid.

### Custom alignment strategies

Alignment strategies are stored in a registry and can be extended at runtime.

```python
from xeval.alignment import register
import xarray as xr

def my_strategy(
    arrays: dict[str, xr.DataArray],
    digits: int | None = 10,
) -> dict[str, xr.DataArray]:
    ...

register("my_strategy", my_strategy)

xeval.evaluate("a + b", ctx={...}, alignment="my_strategy")
```

## Design

`xeval` is intentionally a thin layer. It does not reimplement xarray's alignment or interpolation logic — it wraps `xr.align` and `xr.DataArray.interp_like` directly. Custom alignment rules beyond what xarray provides are out of scope.

If your `ctx` contains no `DataArray` values, use `funcexpr.evaluate` directly.

## Dependencies

- `xarray`
- `funcexpr`
- `numexpr`
- `numpy`