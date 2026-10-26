"""Native HDF5 extraction and verified blackmarblepy request helpers."""
import calendar
from contextlib import contextmanager
from datetime import date, timedelta
import json
import math
import os
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault("TQDM_DISABLE", "1")
import h5py
import httpx
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from tenacity import stop_after_attempt
import blackmarble.download as bm_download

def plain(value):
    """Convert HDF5 attribute values to JSON-friendly Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.ndarray):
        return plain(value.item()) if value.size == 1 else [plain(x) for x in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    return value


@contextmanager
def bounded_verified_requests():
    """Keep TLS verification on and bound library manifest retries.

    blackmarblepy 2026.6.1 requests verify=False for manifests. Adapt the client
    at the call boundary; do not modify the installed package or disable TLS.
    """
    original = httpx.AsyncClient
    def client(*args, **kwargs):
        kwargs["verify"] = True
        kwargs.setdefault("timeout", 60.0)
        return original(*args, **kwargs)
    bounded_get = bm_download.get_url.retry_with(stop=stop_after_attempt(3))
    with patch.object(bm_download.httpx, "AsyncClient", client), patch.object(bm_download, "get_url", bounded_get):
        yield


def month_ranges(start, end):
    day = start
    while day <= end:
        last = min(end, date(day.year, day.month, calendar.monthrange(day.year, day.month)[1]))
        yield day, last
        day = last + timedelta(days=1)


def data_fields(handle):
    groups = []
    handle.visititems(lambda name, obj: groups.append(obj)
                     if isinstance(obj, h5py.Group) and name.endswith("/Data Fields") else None)
    if len(groups) != 1:
        raise ValueError("Expected a single data-fields grid")
    return groups[0]


def schema_for_file(path, geometry):
    with h5py.File(path) as handle:
        fields = data_fields(handle)
        bands = {name: ds for name, ds in fields.items() if isinstance(ds, h5py.Dataset) and ds.ndim == 2}
        if not bands:
            raise ValueError("No raster bands found")
        height, width = next(iter(bands.values())).shape
        west, east, south, north = [float(plain(handle.attrs[k])) for k in (
            "WestBoundingCoord", "EastBoundingCoord", "SouthBoundingCoord", "NorthBoundingCoord")]
        if not box(west, south, east, north).covers(geometry):
            raise ValueError("This acquisition currently supports an AOI wholly within one tile")
        dx, dy = (east-west)/width, (north-south)/height
        a,b,c,d = geometry.bounds
        c0,c1 = max(0,math.floor((a-west)/dx)), min(width,math.ceil((c-west)/dx))
        r0,r1 = max(0,math.floor((north-d)/dy)), min(height,math.ceil((north-b)/dy))
        return {"rows": [r0,r1], "columns": [c0,c1], "full_shape": [height,width],
            "bounds": [west,east,south,north],
            "x": (west+(np.arange(c0,c1)+0.5)*dx).tolist(),
            "y": (north-(np.arange(r0,r1)+0.5)*dy).tolist(),
            "bands": {name:{"dtype":str(ds.dtype),
                "attrs":{k:plain(v) for k,v in ds.attrs.items()}} for name,ds in bands.items()}}


def grid_signature(schema):
    return {**{k:v for k,v in schema.items() if k != "bands"},
            "bands":{name:spec["dtype"] for name,spec in schema["bands"].items()}}


def assemble_cube(paths, days, schema, geometry, product):
    r0,r1 = schema["rows"]
    c0,c1 = schema["columns"]
    variables = {}
    for name, spec in schema["bands"].items():
        dtype = np.dtype(spec["dtype"])
        fill = spec["attrs"].get("_FillValue")
        if fill is None:
            if np.issubdtype(dtype, np.floating):
                fill = np.nan
            else:
                raise ValueError(f"Integer band {name} has no fill value for missing dates")
        variables[name] = xr.DataArray(np.full((len(days),r1-r0,c1-c0),fill,dtype=dtype),
            dims=("time","y","x"), attrs=spec["attrs"].copy())
    available = np.zeros(len(days), dtype=np.uint8)
    source_names = []
    scene_metadata = []
    for i, day in enumerate(days):
        path = paths.get(day)
        source_names.append(path.name if path else "")
        if path is None:
            scene_metadata.append("")
            continue
        with h5py.File(path) as handle:
            if plain(handle.attrs["RangeBeginningDate"]) != day.isoformat():
                raise ValueError("Source date does not match target calendar date")
            if plain(handle.attrs["ShortName"]) != product or plain(handle.attrs["VersionID"]) != "002":
                raise ValueError("Product/collection mismatch")
            current = schema_for_file(path, geometry)
            if grid_signature(current) != grid_signature(schema):
                raise ValueError("Native grid/band metadata changed; separate schema revision required")
            scene_metadata.append(json.dumps({name:spec["attrs"] for name,spec in current["bands"].items()}))
            fields = data_fields(handle)
            for name in variables:
                variables[name].values[i] = fields[name][r0:r1,c0:c1]
        available[i] = 1
    ds = xr.Dataset(variables, coords={"time":pd.to_datetime(days),"x":schema["x"],"y":schema["y"]})
    ds["source_available"] = ("time",available)
    ds["source_available"].attrs["description"] = "1=source granule acquired; 0=not listed in NASA manifest"
    ds["source_file"] = ("time",np.asarray(source_names,dtype=str))
    ds["scene_band_metadata"] = ("time",np.asarray(scene_metadata,dtype=str))
    ds["scene_band_metadata"].attrs["description"] = "Exact NASA band attributes per acquired date; use for subsequent decoding"
    west,east,south,north = schema["bounds"]
    dx,dy = (east-west)/schema["full_shape"][1],(north-south)/schema["full_shape"][0]
    mask = np.array([[geometry.intersects(box(x-dx/2,y-dy/2,x+dx/2,y+dy/2))
                      for x in schema["x"]] for y in schema["y"]],dtype=np.uint8)
    ds["aoi_intersects_pixel"] = (("y","x"),mask)
    ds.attrs.update(product=product, collection="002", crs="EPSG:4326",
                    value_policy="Original packed NASA values; no scaling, QA masking or gap filling performed",
                    spatial_policy="Fixed pixel-aligned AOI bounding window; AOI mask stored separately",
                    boundary_geojson=json.dumps(geometry.__geo_interface__))
    ds.x.attrs.update(units="degrees_east",axis="X")
    ds.y.attrs.update(units="degrees_north",axis="Y")
    return ds


def open_history(root, product):
    """Open a product lazily; all raster variables have dimensions time,y,x."""
    files = sorted((Path(root)/product).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No {product} checkpoints in {root}")
    dataset = xr.open_mfdataset(files, combine="nested", concat_dim="time", mask_and_scale=False, decode_times=True,
                            data_vars="minimal", coords="minimal", compat="equals",
                            chunks={"time":31}, join="exact")
    if not dataset.indexes["time"].is_unique:
        dataset.close()
        raise ValueError("Overlapping checkpoint dates; use one checkpoint per product/month or a new output directory")
    return dataset

