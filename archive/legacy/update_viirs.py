"""Incremental seven-band stores: local checkpoints first, blackmarblepy if needed."""
import argparse
import asyncio
from datetime import date
import hashlib
import json
import math
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

from acquire_history import assemble_cube, schema_for_file, bounded_verified_requests
from viirs_zarr import BANDS, open_product, pending_dates, prepare, upsert


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


def nasa_batch(product, days, geometry, query_roi, token_file, cache, history, boundary_hash):
    """One H5 per product/day supplies every requested band; keep the cache."""
    cache = Path(cache) / product
    cache.mkdir(parents=True, exist_ok=True)
    downloader = BlackMarbleDownloader(token=Path(token_file).read_text().strip(),
                                      directory=cache, collection="5200")
    dates = [stamp.date() for stamp in days]
    with bounded_verified_requests():
        manifest = asyncio.run(downloader.get_manifest(query_roi, Product(product), dates))
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
    return ds, len(needed)


def update(product, start, end, *, output="data/miami_zarr", history="data/miami_history",
           aoi="miami_smelter_boundary.geojson", allow_download=False,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", default="data/miami_zarr")
    parser.add_argument("--history", default="data/miami_history")
    parser.add_argument("--aoi", default="miami_smelter_boundary.geojson")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--retry-missing", action="store_true")
    parser.add_argument("--token-file", default=".earthdata_token.txt")
    parser.add_argument("--cache", default="data/blackmarble_cache")
    parser.add_argument("--verify", action="store_true")
    args = vars(parser.parse_args())
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
