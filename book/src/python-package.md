# Python package

The [`rwarp-py/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-py)
subdirectory of this repository is a Python package, also named **rwarp**,
that exposes the same planning and resampling pipeline via
[PyO3](https://pyo3.rs/).

> **Status**: the Python bindings are in early development. The directory
> structure is in place; function signatures and packaging are being
> finalised. Feedback on the API shape is welcome via
> [issues](https://github.com/hypertidy/rwarp/issues).

## Planned install

```sh
pip install rwarp
```

Or from source (requires a Rust toolchain >= 1.75 and maturin):

```sh
cd rwarp-py
pip install maturin
maturin develop
```

## Planned API

The Python API is intended to mirror the R package closely, with
NumPy arrays as the pixel buffer type.

```python
import rwarp
import numpy as np

# --- define source and destination grids ---
src_crs = "EPSG:4326"
src_gt  = (-180.0, 0.1, 0.0, 90.0, 0.0, -0.1)   # (xmin, xres, 0, ymax, 0, -yres)
src_dim = (3600, 1800)

dst_crs = "EPSG:3857"
dst_gt  = (-20037508.0, 78271.5, 0.0, 20037508.0, 0.0, -78271.5)
dst_dim = (512, 512)

# --- 1. Plan (no I/O) ---
plan = rwarp.warp_plan(
    src_crs=src_crs, src_gt=src_gt, src_dim=src_dim,
    dst_crs=dst_crs, dst_gt=dst_gt, dst_size=dst_dim,
)

# inspect fill ratio before any read
print(plan[0].fill_ratio)

# --- 2. Execute with a NumPy-returning read function ---
def read_fn(xoff, yoff, xsize, ysize):
    # return a 1-D int32 array of length xsize * ysize
    return my_source.read(xoff, yoff, xsize, ysize).ravel().astype(np.int32)

result = rwarp.warp_exec(
    plan,
    read_fn=read_fn,
    src_crs=src_crs, src_gt=src_gt,
    dst_crs=dst_crs, dst_gt=dst_gt,
    dst_size=dst_dim, nodata=-32768,
)

# result is a 1-D int32 ndarray, length dst_dim[0] * dst_dim[1]
grid = result.reshape(dst_dim[1], dst_dim[0])
```

## Core functions (planned)

| Function | What it does |
|---|---|
| `warp_plan()` | Compute source windows for a destination grid (pure geometry, no I/O) |
| `warp_exec()` | Execute a plan with a user-supplied read function |
| `warp_resample()` | Warp a pre-fetched pixel buffer directly |
| `pixel_transform()` | Map destination pixels to fractional source pixel coordinates |
| `crs_describe()` | Describe a CRS string in all PROJ output formats |

## Why PyO3

[PyO3](https://pyo3.rs/) compiles the same Rust core to a Python extension
module. The warp kernel and the planning geometry are identical to the R
package and the WebAssembly build; only the FFI layer differs.

NumPy arrays pass in and out without copying: the read function returns a
NumPy array and `warp_exec` returns one, so the data stays in Rust-managed
memory and Python sees a view.

## Integration ideas

Because the read function is user-supplied, any array source works:

```python
# rasterio
import rasterio
with rasterio.open("my.tif") as ds:
    def read_fn(xoff, yoff, xsize, ysize):
        window = rasterio.windows.Window(xoff, yoff, xsize, ysize)
        return ds.read(1, window=window).ravel().astype(np.int32)
    result = rwarp.warp_exec(plan, read_fn=read_fn, ...)

# xarray / zarr
import xarray as xr
da = xr.open_zarr("store.zarr")["sst"]
def read_fn(xoff, yoff, xsize, ysize):
    return da.isel(x=slice(xoff, xoff+xsize),
                   y=slice(yoff, yoff+ysize)).values.ravel().astype(np.int32)
result = rwarp.warp_exec(plan, read_fn=read_fn, ...)
```

## Source and issues

The Python package lives in
[`rwarp-py/`](https://github.com/hypertidy/rwarp/tree/main/rwarp-py).
Issues and contributions are welcome at
[github.com/hypertidy/rwarp](https://github.com/hypertidy/rwarp).
