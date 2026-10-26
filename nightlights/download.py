"""Incremental seven-band stores: local checkpoints first, blackmarblepy if needed."""
import argparse
import asyncio
from datetime import date, datetime, timezone, timedelta
import hashlib
import json
import math
import os
import time
from pathlib import Path
from unittest.mock import patch

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box
from blackmarble.download import BlackMarbleDownloader
from blackmarble.types import Product
import blackmarble.download as bm_download

from .acquisition import assemble_cube, schema_for_file, bounded_verified_requests
from .storage import BANDS, TIME_ENCODING, open_product, pending_dates, prepare, upsert


def persist_progress(path, state):
    """Progress is advisory: a transient Windows file lock must not stop acquisition."""
    path = Path(path)
    staged = path.with_suffix('.json.part')
    payload = json.dumps(state,indent=2)
    for attempt in range(20):
        try:
            staged.write_text(payload,encoding='utf-8')
            staged.replace(path)
            return True
        except PermissionError:
            if attempt<19:
                time.sleep(.25)
    print('WARNING: progress file is temporarily locked; acquisition continues and will retry the next update',flush=True)
    return False


def save_source_checkpoint(ds, history, product, manifest):
    """Keep all extracted bands and QA before the seven-band raw-store append."""
    folder = Path(history)/product
    folder.mkdir(parents=True, exist_ok=True)
    first, last = [pd.Timestamp(ds.time.values[i]).date().isoformat() for i in (0,-1)]
    path = folder/f'{first}_{last}.nc'
    staged = path.with_suffix('.nc.part')
    # A complete checkpoint is immutable; an interrupted staging file is replaceable.
    if path.exists() and path.with_suffix('.json').exists():
        with xr.open_dataset(path,engine='h5netcdf',mask_and_scale=False) as old:
            xr.testing.assert_identical(old.load(), ds)
        return
    encoding = {n:{'zlib':True,'complevel':4} for n in ds.data_vars if ds[n].ndim==3}
    for name in ds.variables:
        if '_FillValue' not in ds[name].attrs:
            encoding.setdefault(name,{})['_FillValue'] = None
    encoding.setdefault('time',{}).update(TIME_ENCODING)
    ds.to_netcdf(staged,engine='netcdf4',encoding=encoding)
    with xr.open_dataset(staged,engine='h5netcdf',mask_and_scale=False) as saved:
        xr.testing.assert_identical(ds,saved.load())
    staged.replace(path)
    record = {'status':'complete','product':product,'start':first,'end':last,
              'boundary_sha256':ds.attrs['boundary_sha256'],
              'requested_days':ds.sizes['time'],'acquired_days':int(ds.source_available.sum()),
              'stored_bands':list(ds.data_vars),'manifest':manifest.to_dict(orient='records')}
    metadata = path.with_suffix('.json')
    temporary = metadata.with_suffix('.json.part')
    temporary.write_text(json.dumps(record,default=str,indent=2),encoding='utf-8')
    temporary.replace(metadata)


def local_batches(history, product, pending, boundary_hash):
    """Open only checkpoints whose filename intervals intersect missing dates."""
    for path in sorted((Path(history) / product).glob("*.nc")):
        start, end = path.stem.split("_")
        wanted = pending[(pending >= start) & (pending <= end)]
        if not len(wanted):
            continue
        record = json.loads(path.with_suffix(".json").read_text())
        if record.get("status") != "complete":
            continue
        if record["boundary_sha256"] != boundary_hash:
            raise ValueError(f"AOI differs from checkpoint: {path}")
        with xr.open_dataset(path, mask_and_scale=False) as original:
            present = wanted.intersection(pd.DatetimeIndex(original.time.values))
            if not len(present):
                continue
            selected = original[BANDS[product] + ["source_available", "source_file",
                         "scene_band_metadata", "aoi_intersects_pixel"]].sel(time=present).load()
        selected.attrs["boundary_sha256"] = boundary_hash
        yield path, selected


async def manifest_allow_empty(downloader, roi, product, dates):
    """Work around blackmarblepy's empty-listing KeyError with response evidence."""
    original_get = bm_download.get_url
    responses = []
    async def checked_get(*args, **kwargs):
        response = await original_get(*args, **kwargs)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload,dict) or not isinstance(payload.get('content'),list):
            raise ValueError('NASA listing did not contain a valid content list')
        responses.append(payload['content'])
        return response
    with patch.object(bm_download,'get_url',checked_get):
        try:
            return await downloader.get_manifest(roi,product,dates)
        except KeyError as error:
            # Never infer absence from a generic exception or malformed/API-error payload.
            if error.args!=('name',) or not responses or any(responses):
                raise
            return pd.DataFrame(columns=['date','TileID','name'])


def nasa_batch(product, days, geometry, query_roi, token_file, cache, history, boundary_hash):
    """One H5 per product/day supplies every requested band; keep the cache."""
    cache = Path(cache) / product
    cache.mkdir(parents=True, exist_ok=True)
    downloader = BlackMarbleDownloader(token=Path(token_file).read_text().strip(),
                                      directory=cache, collection="5200")
    dates = [stamp.date() for stamp in days]
    with bounded_verified_requests():
        manifest = asyncio.run(manifest_allow_empty(downloader,query_roi,Product(product),dates))
    # An API exception propagates. Only a successful listing can mark a date absent.
    manifest = manifest[manifest["date"].isin(dates)].sort_values("name").drop_duplicates(["date", "TileID"], keep="last")
    if len(manifest["TileID"].unique()) > 1:
        raise ValueError("Multiple tiles require a mosaic adapter")
    found = {}
    for row in manifest.itertuples():
        candidates = [cache / row.name, Path("data") / row.name]
        candidates += list((Path(history) / "staging").rglob(row.name))
        for path in candidates:
            if path.exists() and h5py.is_hdf5(path):
                found[row.date] = path
                break
    needed = sorted(set(manifest["date"]) - set(found))
    if needed:
        async def selected_manifest(instance, roi, pid, requested):
            return manifest[manifest["date"].isin(requested)].copy()
        with bounded_verified_requests(), patch.object(BlackMarbleDownloader, "get_manifest", selected_manifest):
            files = downloader.download(query_roi, Product(product), needed, n_jobs=3)
        for path in files:
            path = Path(path)
            found[pd.to_datetime(path.name.split(".")[1], format="A%Y%j").date()] = path
    if set(found) != set(manifest["date"]):
        raise RuntimeError("Not all listed sources downloaded; refusing to record an incomplete batch")
    if found:
        schema = schema_for_file(next(iter(found.values())), geometry)
    else:
        schema = json.loads((Path(history) / product / "schema.json").read_text())
    ds = assemble_cube(found, dates, schema, geometry, product)
    ds.attrs["boundary_sha256"] = boundary_hash
    save_source_checkpoint(ds,history,product,manifest)
    return ds, len(needed)


def resume_acquisition(end=None, *, start=None, output='data/miami_zarr', history='data/miami_history',
                       aoi='aoi/miami_smelter_boundary.geojson',token_file='.earthdata_token.txt'):
    """Interleave products in six-day batches, with live status and a bounded cache."""
    import psutil
    xr.set_options(netcdf_engine_order=('h5netcdf','netcdf4','scipy'))
    end = pd.Timestamp(end or date.today()-timedelta(days=1))
    tails = []
    for product in BANDS:
        with open_product(Path(output)/(product+'.zarr')) as ds:
            tails.append(pd.Timestamp(ds.time.values[-1])+pd.Timedelta(days=1))
    start = pd.Timestamp(start) if start else min(tails)
    if end<start:
        print('No new dates to acquire',flush=True)
        return
    # Only files in this task-owned transient directory may be removed after verification.
    cache = Path('data/blackmarble_cache/resumed_acquisition').resolve()
    allowed = Path('data/blackmarble_cache').resolve()
    if cache.parent!=allowed:
        raise ValueError('Unexpected transient cache location')
    cache.mkdir(parents=True,exist_ok=True)
    (cache/'.managed_by_nightlights').write_text('Full tiles may be removed only after verified AOI persistence.\n')
    state_path = Path(history)/'resume_status.json'
    state = {'state':'running','pid':os.getpid(),'start':str(start.date()),'end':str(end.date()),
             'started_utc':datetime.now(timezone.utc).isoformat(),'downloaded_files_this_run':0,
             'completed_batches':0,'products':{}}
    def write_state(**changes):
        state.update(changes,updated_utc=datetime.now(timezone.utc).isoformat())
        persist_progress(state_path,state)
    write_state(phase='starting')
    try:
        for first in pd.date_range(start,end,freq='6D'):
            last = min(first+pd.Timedelta(days=5),end)
            for product in BANDS:
                if psutil.disk_usage(str(cache)).free<3*1024**3:
                    raise RuntimeError('Less than 3 GiB free; acquisition stopped before the next batch')
                write_state(phase='acquiring',current_product=product,
                            current_start=str(first.date()),current_end=str(last.date()))
                report = update(product,first.date(),last.date(),output=output,history=history,
                                aoi=aoi,token_file=token_file,cache=cache,allow_download=True)
                state['downloaded_files_this_run'] += report['downloaded_files']
                state['completed_batches'] += 1
                state['products'][product] = {key:report[key] for key in
                    ('stored_days','acquired_days','unavailable_days','first_date','last_date')}
                # upsert validated the seven-band rows. The all-band checkpoint was
                # independently round-tripped before upsert, so these tiles are expendable.
                with open_product(Path(output)/(product+'.zarr')) as saved:
                    names = [str(n) for n in saved.source_file.sel(time=slice(first,last)).values if n]
                for name in names:
                    tile = (cache/product/name).resolve()
                    if tile.parent!=(cache/product).resolve() or tile.suffix!='.h5':
                        raise ValueError('Unexpected managed tile path')
                    tile.unlink(missing_ok=True)
                write_state(phase='batch_complete')
        write_state(state='complete',phase='complete')
    except BaseException as error:
        # Status records no credentials or exception payloads containing request headers.
        write_state(state='failed',phase='stopped',error_type=type(error).__name__)
        raise


def update(product, start, end, *, output="data/miami_zarr", history="data/miami_history",
           aoi="aoi/miami_smelter_boundary.geojson", allow_download=False,
           token_file=".earthdata_token.txt", cache="data/blackmarble_cache", retry_missing=False):
    if pd.Timestamp(end) < pd.Timestamp(start):
        raise ValueError("End precedes start")
    store = Path(output) / f"{product}.zarr"
    boundary_hash = hashlib.sha256(Path(aoi).read_bytes()).hexdigest()
    pending = pending_dates(store, start, end, retry_missing)
    if store.exists():
        with open_product(store) as current:
            if current.attrs.get("boundary_sha256") != boundary_hash:
                raise ValueError("AOI differs from existing store; choose a new output")
    report = {"product": product, "requested_start": str(start), "requested_end": str(end),
              "pending_at_start": len(pending), "local_files_read": 0, "nasa_batches": 0,
              "downloaded_files": 0, "updates": []}
    # A retry of unavailable dates must actually check NASA rather than reuse
    # the same old absent-source checkpoint.
    if retry_missing and store.exists():
        with open_product(store) as current:
            local_unavailable = pd.DatetimeIndex(current.time.values[current.source_available.values == 0])
    else:
        local_unavailable = pd.DatetimeIndex([])
    batches = local_batches(history, product, pending.difference(local_unavailable), boundary_hash) if len(pending) else []
    for path, ds in batches:
        result = upsert(store, ds, product)
        report["local_files_read"] += 1
        report["updates"].append(result)
        pending = pending.difference(pd.DatetimeIndex(ds.time.values))
        print(f"LOCAL {product} {path.stem}: {result['action']}", flush=True)
    if len(pending) and not allow_download:
        report["unresolved_dates"] = [str(d.date()) for d in pending]
    elif len(pending):
        roi = gpd.read_file(aoi).to_crs(4326).dissolve()
        geometry = roi.geometry.iloc[0]
        if not geometry.is_valid:
            raise ValueError("Invalid AOI")
        a, b, c, d = geometry.bounds
        query_roi = gpd.GeoDataFrame(geometry=[box(math.floor(a*100)/100, math.floor(b*100)/100,
                                                math.ceil(c*100)/100, math.ceil(d*100)/100)], crs=4326)
        # Small requests also avoid the long-range manifest pagination failure.
        groups = []
        for stamp in pending:
            if not groups or len(groups[-1]) == 6 or (stamp - groups[-1][-1]).days > 1:
                groups.append([])
            groups[-1].append(stamp)
        for group in groups:
            days = pd.DatetimeIndex(group)
            ds, downloaded = nasa_batch(product, days, geometry, query_roi, token_file, cache, history, boundary_hash)
            report["updates"].append(upsert(store, ds, product))
            report["nasa_batches"] += 1
            report["downloaded_files"] += downloaded
            print(f"NASA {product} {days[0].date()}..{days[-1].date()}: {downloaded} files", flush=True)
    if store.exists():
        with open_product(store) as saved:
            report.update(stored_days=saved.sizes["time"], acquired_days=int(saved.source_available.sum()),
                          unavailable_days=int((saved.source_available == 0).sum()),
                          first_date=str(pd.Timestamp(saved.time.values[0]).date()),
                          last_date=str(pd.Timestamp(saved.time.values[-1]).date()))
    return report


def verify_history(product, output="data/miami_zarr", history="data/miami_history"):
    """Compare every migrated raw value and dtype with the retained originals."""
    comparisons = 0
    with open_product(Path(output) / f"{product}.zarr") as saved:
        saved_dates = pd.DatetimeIndex(saved.time.values)
        for path in sorted((Path(history) / product).glob("*.nc")):
            start, end = path.stem.split("_")
            if not len(saved_dates[(saved_dates >= start) & (saved_dates <= end)]):
                continue
            record = json.loads(path.with_suffix(".json").read_text())
            if record.get("status") != "complete":
                continue
            with xr.open_dataset(path, mask_and_scale=False) as source:
                dates = pd.DatetimeIndex(source.time.values).intersection(saved_dates)
                if not len(dates):
                    continue
                raw = prepare(source.sel(time=dates).load(), product)
                actual = saved.sel(time=dates)
                for name in raw.data_vars:
                    np.testing.assert_array_equal(actual[name].values, raw[name].values)
                    if name in BANDS[product]:
                        assert actual[name].dtype == raw[name].dtype
                comparisons += len(dates)
                print(f"VERIFIED {product} {path.stem}: {comparisons} days", flush=True)
    return comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--resume", action="store_true",help="Continue both products from their saved tails through yesterday; bounded cache")
    parser.add_argument("--output", default="data/miami_zarr")
    parser.add_argument("--history", default="data/miami_history")
    parser.add_argument("--aoi", default="aoi/miami_smelter_boundary.geojson")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--token-file", default=".earthdata_token.txt")
    parser.add_argument("--cache", default="data/blackmarble_cache")
    parser.add_argument("--verify", action="store_true")
    args = vars(parser.parse_args(argv))
    resume = args.pop('resume')
    if resume:
        if not args['allow_download']:
            parser.error('--resume requires --allow-download')
        if args['retry_missing'] or args['verify']:
            parser.error('--resume extends the tail; use explicit dates for retries or verification')
        return resume_acquisition(args['end'],start=args['start'],output=args['output'],history=args['history'],
                                  aoi=args['aoi'],token_file=args['token_file'])
    if args['start'] is None or args['end'] is None:
        parser.error('--start and --end are required without --resume')
    verify = args.pop("verify")
    reports = []
    for product in BANDS:
        report = update(product, **args)
        if verify:
            report["verified_days"] = verify_history(product, args["output"], args["history"])
        reports.append(report)
    output = Path(args["output"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "last_run.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
