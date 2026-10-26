"""Acquire uncleaned multi-band Black Marble AOI cubes with monthly checkpoints.

Downloads use blackmarblepy. Stored values retain NASA packing, fill values and
quality flags; no QA masking, band substitution, interpolation or statistics.
"""
import argparse
import asyncio
import calendar
from contextlib import contextmanager
from datetime import date, timedelta, datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import threading
import time
import importlib.metadata
from unittest.mock import patch

os.environ.setdefault("TQDM_DISABLE", "1")

import geopandas as gpd
import h5py
import httpx
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from tenacity import stop_after_attempt

from blackmarble.download import BlackMarbleDownloader
import blackmarble.download as bm_download
from blackmarble.types import Product
from read_nightlights import plain


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aoi",type=Path,default=Path("miami_smelter_boundary.geojson"))
    parser.add_argument("--start",type=date.fromisoformat,default=date(2016,1,1))
    parser.add_argument("--end",type=date.fromisoformat,required=True)
    parser.add_argument("--products",nargs="+",choices=["VNP46A1","VNP46A2"],default=["VNP46A1","VNP46A2"])
    parser.add_argument("--token-file",type=Path,default=Path(".earthdata_token.txt"))
    parser.add_argument("--output",type=Path,default=Path("data/miami_history"))
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--plan-only",action="store_true",help="Query manifests and save availability/size; no downloads")
    parser.add_argument("--reuse-manifest",action="store_true",help="Reuse an incomplete batch's saved NASA listing when resuming")
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("End precedes start")
    roi = gpd.read_file(args.aoi).to_crs(4326).dissolve()
    geometry = roi.geometry.iloc[0]
    if not geometry.is_valid:
        parser.error("Invalid AOI geometry")
    # Give the API an unambiguous query box; the library rounds bounds to 0.01 degrees.
    a,b,c,d=geometry.bounds
    query_roi=gpd.GeoDataFrame(geometry=[box(math.floor(a*100)/100,math.floor(b*100)/100,
                                          math.ceil(c*100)/100,math.ceil(d*100)/100)],crs=4326)
    token=args.token_file.read_text(encoding="utf-8").strip()
    args.output.mkdir(parents=True,exist_ok=True)
    staging=args.output/"staging"
    staging.mkdir(exist_ok=True)
    boundary_hash=hashlib.sha256(args.aoi.read_bytes()).hexdigest()
    run_path=args.output/"run_status.json"
    run={"requested_start":args.start.isoformat(),"requested_end":args.end.isoformat(),
         "products":args.products,"state":"running","boundary_sha256":boundary_hash,
         "pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
         "blackmarblepy_version":importlib.metadata.version("blackmarblepy"),
         "storage_policy":"Monthly uncleaned AOI cubes retained; task-created full-tile staging is temporary"}
    status_lock=threading.Lock()
    def update_status(**changes):
        with status_lock:
            if changes.pop("download_completed",False):
                run["batch_downloaded_files"]=run.get("batch_downloaded_files",0)+1
            run.update(changes,heartbeat_utc=datetime.now(timezone.utc).isoformat())
            temporary=run_path.with_suffix(".json.tmp")
            for attempt in range(10):
                try:
                    temporary.write_text(json.dumps(run,indent=2),encoding="utf-8")
                    temporary.replace(run_path)
                    return
                except PermissionError:
                    # OneDrive/readers can briefly lock the status file on Windows.
                    # A monitoring write must never terminate data acquisition.
                    time.sleep(0.1)
            print("WARNING: status update delayed by a file lock; acquisition continues",flush=True)
    update_status(phase="starting")
    for start,end in month_ranges(args.start,args.end):
        days=[start+timedelta(days=i) for i in range((end-start).days+1)]
        for product in args.products:
            update_status(current_product=product,current_start=start.isoformat(),current_end=end.isoformat(),phase="checking_checkpoint")
            folder=args.output/product
            folder.mkdir(exist_ok=True)
            stem=f"{start}_{end}"
            checkpoint=folder/f"{stem}.nc"
            metadata_path=folder/f"{stem}.json"
            if checkpoint.exists() and metadata_path.exists():
                record=json.loads(metadata_path.read_text())
                if record.get("boundary_sha256") != boundary_hash:
                    raise ValueError("AOI changed; choose a new output directory")
                if record.get("status") == "complete":
                    print(f"REUSE {product} {stem}",flush=True)
                    continue
            with tempfile.TemporaryDirectory(prefix=f"{product}_",dir=staging.resolve()) as cache:
                assert Path(cache).resolve().is_relative_to(staging.resolve())
                downloader=BlackMarbleDownloader(token=token,directory=Path(cache),collection="5200")
                update_status(phase="querying_manifest")
                cached=json.loads(metadata_path.read_text()) if args.reuse_manifest and metadata_path.exists() else None
                if cached and cached.get("boundary_sha256")==boundary_hash and cached.get("start")==start.isoformat() and cached.get("end")==end.isoformat() and cached.get("product")==product:
                    manifest=pd.DataFrame(cached["manifest"])
                    manifest["date"]=pd.to_datetime(manifest["date"]).dt.date
                    print(f"REUSE MANIFEST {product} {stem}",flush=True)
                else:
                    with bounded_verified_requests():
                        manifest=asyncio.run(downloader.get_manifest(query_roi,Product(product),days))
                # Retain latest production revision and exact requested dates.
                manifest=manifest[manifest["date"].isin(days)].sort_values("name").drop_duplicates(["date","TileID"],keep="last")
                tiles=manifest["TileID"].unique()
                if len(tiles)>1:
                    raise ValueError("AOI query touches multiple tiles; acquisition needs a mosaic adapter")
                records=manifest.drop(columns="geometry",errors="ignore").to_dict("records")
                metadata={"product":product,"start":start.isoformat(),"end":end.isoformat(),
                    "requested_days":len(days),"listed_days":len(set(manifest["date"])),
                    "listed_bytes":int(pd.to_numeric(manifest["size"]).sum()),
                    "boundary_sha256":boundary_hash,"manifest":records}
                metadata_path.write_text(json.dumps(metadata,default=str,indent=2),encoding="utf-8")
                print(f"MANIFEST {product} {stem}: {metadata['listed_days']}/{len(days)} days, {metadata['listed_bytes']/1e9:.2f} GB",flush=True)
                if args.plan_only:
                    continue
                found={}
                staged={p.name:p for p in staging.rglob("*.h5") if h5py.is_hdf5(p)}
                # Reuse existing local acquisitions; never remove user's data cache.
                for day in sorted(set(manifest["date"])):
                    names=manifest.loc[manifest["date"]==day,"name"].tolist()
                    matches=[Path("data")/n for n in names if (Path("data")/n).exists()]
                    matches += [staged[n] for n in names if n in staged]
                    if matches:
                        found[day]=matches[-1]
                missing_days=sorted(set(manifest["date"])-set(found))
                if missing_days:
                    update_status(phase="downloading",batch_reused_days=len(found),batch_download_days=len(missing_days),batch_downloaded_files=0)
                    original_download=BlackMarbleDownloader._download_file
                    async def selected_manifest(instance,gdf,product_id,date_range):
                        # The library queries min..max and otherwise downloads
                        # already cached days between sparse missing dates.
                        return manifest[manifest["date"].isin(date_range)].copy()
                    def tracked_download(instance,name,skip_if_exists=True):
                        path=original_download(instance,name,skip_if_exists)
                        update_status(download_completed=True,last_downloaded_file=Path(path).name)
                        print(f"DOWNLOADED {Path(path).name}",flush=True)
                        return path
                    with bounded_verified_requests(),patch.object(BlackMarbleDownloader,"get_manifest",selected_manifest),patch.object(BlackMarbleDownloader,"_download_file",tracked_download):
                        files=downloader.download(query_roi,Product(product),missing_days,n_jobs=args.workers)
                    for path in files:
                        path=Path(path)
                        day=pd.to_datetime(path.name.split(".")[1],format="A%Y%j").date()
                        found[day]=path
                schema_path=folder/"schema.json"
                if found:
                    schema=schema_for_file(next(iter(found.values())),geometry)
                    if schema_path.exists() and grid_signature(json.loads(schema_path.read_text()))!=grid_signature(schema):
                        raise ValueError("Schema differs from earlier checkpoint")
                    schema_path.write_text(json.dumps(schema,indent=2),encoding="utf-8")
                elif schema_path.exists():
                    schema=json.loads(schema_path.read_text())
                else:
                    raise ValueError("No source granule to establish schema for this product")
                update_status(phase="assembling_checkpoint")
                ds=assemble_cube(found,days,schema,geometry,product)
                temp=checkpoint.with_suffix(".nc.part")
                encoding={name:{"zlib":True,"complevel":4,"chunksizes":(min(len(days),31),len(ds.y),len(ds.x))}
                          for name in schema["bands"]}
                ds.to_netcdf(temp,engine="netcdf4",encoding=encoding)
                with xr.open_dataset(temp,decode_cf=False) as saved:
                    assert saved.sizes==ds.sizes
                    for name in schema["bands"]:
                        np.testing.assert_array_equal(saved[name].values,ds[name].values)
                temp.replace(checkpoint)
                metadata.update(status="complete",stored_bands=list(schema["bands"]),
                                stored_sizes=dict(ds.sizes),acquired_days=int(ds.source_available.sum()))
                metadata_path.write_text(json.dumps(metadata,default=str,indent=2),encoding="utf-8")
                update_status(phase="checkpoint_complete",last_completed_product=product,last_completed_end=end.isoformat())
                print(f"CHECKPOINT {product} {stem}: {len(schema['bands'])} bands {dict(ds.sizes)}",flush=True)
                ds.close()
            # Only the task-created transient download directory is cleaned up.
    update_status(state="complete",phase="complete")
    print("Acquisition finished. Open each product with acquire_history.open_history().",flush=True)


if __name__=="__main__":
    logging.basicConfig(level=logging.WARNING)
    try:
        main()
    except Exception as error:
        import sys
        output = Path("data/miami_history")
        if "--output" in sys.argv:
            output = Path(sys.argv[sys.argv.index("--output")+1])
        state_path = output/"run_status.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state.update(state="failed",error_type=type(error).__name__)
            state_path.write_text(json.dumps(state,indent=2),encoding="utf-8")
        raise
