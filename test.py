import xarray as xr
import numpy as np

# 基本構造の確認
da = xr.DataArray(
    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    dims=['time', 'x'],
    coords={'time': [0.0, 1.0], 'x': [10.0, 20.0, 30.0]}
)
print('=== DataArray基本構造 ===')
print(da)
print()
print('dims:', da.dims)
print('coords:', da.coords)
print('shape:', da.shape)
print('values type:', type(da.values))
print()

# 異なるcoordsのDataArray同士の演算
da2 = xr.DataArray(
    np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
    dims=['time', 'x'],
    coords={'time': [0.0, 1.0], 'x': [10.0, 20.0, 30.0]}
)
print('=== 同じcoords -> 演算 ===')
print(da + da2)
print()

# coordsがずれている場合
da3 = xr.DataArray(
    np.array([[10.0, 20.0, 30.0]]),
    dims=['time', 'x'],
    coords={'time': [0.5], 'x': [10.0, 20.0, 30.0]}  # timeがずれてる
)
print('=== coordsミスマッチ -> どうなる? ===')
try:
    result = da + da3
    print(result)
except Exception as e:
    print(f'Error: {e}')

# coordsの値がずれている場合
da4 = xr.DataArray(
    np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]),
    dims=['time', 'x'],
    coords={'time': [100, 200], 'x': [10.0, 20.0, 30.0]}  # timeがずれてる
)
print('=== coords値ミスマッチ -> どうなる? ===')
try:
    result = da + da3
    print(result)
except Exception as e:
    print(f'Error: {e}')

