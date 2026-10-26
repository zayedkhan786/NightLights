"""Single-writer, raw-value-preserving Zarr storage for the seven project bands."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
import psutil
import rioxarray  # registers the geospatial accessor
import xarray as xr
import zarr
from numcodecs import Zstd

BANDS = {
    "VNP46A1": ["BrightnessTemperature_M13", "BrightnessTemperature_M15",
                "DNB_At_Sensor_Radiance", "QF_Cloud_Mask"],
    "VNP46A2": ["DNB_BRDF-Corrected_NTL", "Gap_Filled_DNB_BRDF-Corrected_NTL", "QF_Cloud_Mask"],
}
ANCILLARY = ["source_available", "source_file", "source_url", "scene_band_metadata", "aoi_intersects_pixel"]
TIME_ENCODING = {"dtype": "int64", "units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}


def open_product(path):
    """Decode dates and spatial CRS, while keeping packed bands and flags raw."""
    return xr.open_zarr(path, consolidated=False, chunks=None, mask_and_scale=False, decode_coords="all")


def prepare(ds, product):
    if ds.attrs.get("crs", "EPSG:4326") != "EPSG:4326" or (ds.rio.crs is not None and ds.rio.crs.to_epsg() != 4326):
        raise ValueError("Expected the native EPSG:4326 grid; reprojection needs a separate store")
    missing = set(BANDS[product]) - set(ds.data_vars)
    if missing:
        raise ValueError(f"Missing requested bands: {sorted(missing)}")
    if ds.attrs.get("product", product) != product or ds.attrs.get("collection", "002") != "002":
        raise ValueError("Wrong product or collection")
    keep = BANDS[product] + [v for v in ANCILLARY if v in ds]
    out = ds[keep].copy(deep=True).sortby("time")
    index = pd.DatetimeIndex(out.time.values)
    if not index.is_unique:
        raise ValueError("Incoming dates contain duplicates")
    if not len(index):
        raise ValueError("Empty update")
    if not (index == index.normalize()).all():
        raise ValueError("Expected daily dates at midnight")
    if "source_available" not in out:
        raise ValueError("source_available is required to distinguish missing sources")
    if "scene_band_metadata" in out:
        metadata = []
        for text in out.scene_band_metadata.values:
            attrs = json.loads(text) if text else {}
            metadata.append(json.dumps({n: attrs[n] for n in BANDS[product] if n in attrs}, separators=(",", ":")))
        out["scene_band_metadata"] = ("time", np.asarray(metadata, dtype=object))
    if "source_url" not in out:
        links = []
        for name in out.source_file.values:
            if name:
                stamp = str(name).split(".")[1][1:]
                links.append(f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/5200/{product}/{stamp[:4]}/{stamp[4:]}/{name}")
            else:
                links.append("")
        out["source_url"] = ("time", np.asarray(links, dtype=object))
    for name in ("source_file", "source_url", "scene_band_metadata"):
        if name in out:
            out[name] = out[name].astype(object)
    out.attrs.update(product=product, collection="002", crs="EPSG:4326",
                     value_policy="Raw NASA packing; no quality masking, scaling, cleaning or band merging",
                     storage_schema="viirs-seven-bands-v1")
    out = out.rio.set_spatial_dims(x_dim="x", y_dim="y").rio.write_crs("EPSG:4326").rio.write_transform()
    for name in out.variables:
        out[name].encoding = {}
    for name in BANDS[product]:
        out[name].attrs["grid_mapping"] = "spatial_ref"
    out.time.encoding = TIME_ENCODING.copy()
    return out


def _encoding(ds):
    encoding = {}
    for name, var in ds.variables.items():
        if name == "time":
            encoding[name] = {**TIME_ENCODING, "chunks": (31,), "compressors": [Zstd(level=3)]}
        elif var.ndim:
            chunks = tuple(31 if d == "time" else ds.sizes[d] for d in var.dims)
            encoding[name] = {"chunks": chunks, "compressors": [Zstd(level=3)]}
    return encoding


def _same(a, b):
    try:
        np.testing.assert_array_equal(a, b)
        return True
    except AssertionError:
        return False


def _check_grid(old, new, product):
    for key in ("product", "collection", "boundary_geojson", "boundary_sha256"):
        if old.attrs.get(key) != new.attrs.get(key):
            raise ValueError(f"Store identity changed: {key}")
    for axis in ("x", "y"):
        if not _same(old[axis].values, new[axis].values):
            raise ValueError("CRS/grid changed; choose a separate store")
    if not _same(old.aoi_intersects_pixel.values, new.aoi_intersects_pixel.values):
        raise ValueError("AOI pixel mask changed")
    if old.rio.crs != new.rio.crs:
        raise ValueError("CRS changed")
    for name in BANDS[product]:
        if old[name].dtype != new[name].dtype:
            raise ValueError(f"Packed dtype changed: {name}")


def _validate(path, expected=None):
    with open_product(path) as saved:
        idx = pd.DatetimeIndex(saved.time.values)
        if not idx.is_monotonic_increasing or not idx.is_unique:
            raise ValueError("Stored time must be sorted and unique")
        if saved.rio.crs.to_epsg() != 4326:
            raise ValueError("Missing or incorrect spatial CRS metadata")
        if expected is not None:
            actual = saved.sel(time=expected.time)
            for name, variable in expected.data_vars.items():
                if "time" in variable.dims:
                    np.testing.assert_array_equal(actual[name].values, variable.values)


def _clean_task_directory(path, store, prefix):
    path = Path(path).resolve()
    if path.parent != Path(store).resolve().parent or not path.name.startswith(prefix):
        raise ValueError("Refusing to clean an unverified staging path")
    shutil.rmtree(path)


@contextmanager
def _writer_lock(store):
    lock = store.parent / f".{store.name}.lock"
    if lock.exists():
        record = json.loads(lock.read_text())
        try:
            proc = psutil.Process(record["pid"])
            alive = abs(proc.create_time() - record["created_process"]) < 1
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError("Another writer owns this Zarr store")
        lock.unlink()
    record = {"pid": os.getpid(), "created_process": psutil.Process().create_time()}
    with lock.open("x", encoding="utf-8") as handle:
        json.dump(record, handle)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def _recover(store):
    journal = store.parent / f".{store.name}.transaction.json"
    if not journal.exists():
        return
    record = json.loads(journal.read_text())
    backup = Path(record["backup"]).resolve()
    prefix = f".{store.name}.txn-"
    if backup.parent != store.parent.resolve() or not backup.name.startswith(prefix):
        raise ValueError("Invalid recovery journal")
    before = set(record["files_before"])
    for file in store.rglob("*"):
        if file.is_file() and str(file.relative_to(store)) not in before:
            file.unlink()
    for file in backup.rglob("*"):
        if file.is_file():
            target = store / file.relative_to(backup)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    journal.unlink()
    _clean_task_directory(backup, store, prefix)


@contextmanager
def _transaction(store, touched_indexes):
    """Snapshot metadata and only chunks that may change, not the full history."""
    prefix = f".{store.name}.txn-"
    backup = Path(tempfile.mkdtemp(prefix=prefix, dir=store.parent))
    before = [str(p.relative_to(store)) for p in store.rglob("*") if p.is_file()]
    dirty = {key for key in before if Path(key).name.startswith(".z")}
    group = zarr.open_group(str(store), mode="r", zarr_format=2)
    for name, array in group.arrays():
        dims = array.attrs.get("_ARRAY_DIMENSIONS", [])
        if "time" not in dims:
            continue
        if dims[0] != "time":
            raise ValueError("Expected time as the first array dimension")
        for chunk in {int(i)//array.chunks[0] for i in touched_indexes}:
            suffix = ".".join([str(chunk)] + ["0"]*(len(dims)-1))
            key = str(Path(name)/suffix)
            if key in before:
                dirty.add(key)
    for key in dirty:
        target = backup/key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(store/key, target)
    journal = store.parent / f".{store.name}.transaction.json"
    journal.write_text(json.dumps({"backup": str(backup.resolve()), "files_before": before}), encoding="utf-8")
    try:
        yield
    except BaseException:
        _recover(store)
        raise
    else:
        journal.unlink()
        _clean_task_directory(backup, store, prefix)


def _write_raw_rows(store, ds, start, append=False):
    """Write packed buffers directly: xarray append would reapply CF scaling.

    xarray remains the public read/assembly interface. Direct Zarr assignment
    deliberately bypasses CF coders for raw integers and per-scene packing.
    """
    group = zarr.open_group(str(store), mode="r+", zarr_format=2)
    count = ds.sizes["time"]
    for name, variable in ds.variables.items():
        if "time" not in variable.dims:
            continue
        if variable.dims[0] != "time":
            raise ValueError("Expected time as first dimension")
        array = group[name]
        if append:
            array.resize((start + count, *array.shape[1:]))
        if name == "time":
            values = ds.time.values.astype("datetime64[D]").astype("int64")
        else:
            values = variable.values
        array[start:start+count] = values


def upsert(path, incoming, product):
    """Append newer dates; repair absent sources; rebuild only for date insertion."""
    store = Path(path).resolve()
    store.parent.mkdir(parents=True, exist_ok=True)
    new = prepare(incoming, product)
    with _writer_lock(store):
        _recover(store)
        if not store.exists():
            prefix = f".{store.name}.init-"
            staged = Path(tempfile.mkdtemp(prefix=prefix, dir=store.parent))
            try:
                new.to_zarr(staged, mode="w-", encoding=_encoding(new), zarr_format=2, consolidated=False)
                _validate(staged, new)
                staged.rename(store)
            finally:
                if staged.exists():
                    _clean_task_directory(staged, store, prefix)
            return {"action": "create", "added_dates": new.sizes["time"], "repaired_dates": 0}

        with open_product(store) as old:
            _check_grid(old, new, product)
            previous = pd.DatetimeIndex(old.time.values)
            requested = pd.DatetimeIndex(new.time.values)
            absent = requested.difference(previous)
            common = requested.intersection(previous)
            repair = []
            for stamp in common:
                a, b = old.sel(time=stamp), new.sel(time=stamp)
                if int(a.source_available) == 0 and int(b.source_available) == 1:
                    repair.append(stamp)
                elif int(a.source_available) == 1 and int(b.source_available) == 1:
                    if any(not _same(a[n].values, b[n].values) for n in BANDS[product]):
                        raise ValueError("Conflicting existing source values; explicit revision policy required")
            if not len(absent) and not repair:
                return {"action": "unchanged", "added_dates": 0, "repaired_dates": 0}
            needs_insertion = len(absent) and absent.min() < previous.max()
            if needs_insertion:
                # Dense sorted arrays must shift indices for earlier insertions.
                # This rare path publishes a new generation and retains the old one.
                base = old.load()
                replace = pd.DatetimeIndex(repair)
                base = base.sel(time=~base.time.isin(replace))
                extra = new.sel(time=absent.union(replace))
                combined = xr.concat([base, extra], dim="time", data_vars="minimal", coords="minimal", compat="equals").sortby("time")
            else:
                combined = None
        if combined is not None:
            prefix = f".{store.name}.init-"
            staged = Path(tempfile.mkdtemp(prefix=prefix, dir=store.parent))
            for variable in combined.variables:
                combined[variable].encoding = {}
            combined.to_zarr(staged, mode="w-", encoding=_encoding(combined), zarr_format=2, consolidated=False)
            _validate(staged, combined)
            retained = store.with_name(store.name + ".previous-" + uuid.uuid4().hex[:8])
            store.rename(retained)
            try:
                staged.rename(store)
            except BaseException:
                retained.rename(store)
                raise
            return {"action": "insert", "added_dates": len(absent), "repaired_dates": len(repair), "previous_generation": str(retained)}

        indexes = [previous.get_loc(stamp) for stamp in repair] + list(range(len(previous), len(previous)+len(absent)))
        with _transaction(store, indexes):
            for stamp in repair:
                position = previous.get_loc(stamp)
                row = new.sel(time=[stamp])
                _write_raw_rows(store, row, position)
            if len(absent):
                tail = new.sel(time=absent)
                _write_raw_rows(store, tail, len(previous), append=True)
            expected = new.sel(time=absent.union(pd.DatetimeIndex(repair)))
            _validate(store, expected)
        return {"action": "append" if len(absent) else "repair", "added_dates": len(absent), "repaired_dates": len(repair)}


def pending_dates(path, start, end, retry_missing=False):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _writer_lock(path):
        _recover(path)
    requested = pd.date_range(start, end, freq="D")
    if not Path(path).exists():
        return requested
    with open_product(path) as ds:
        local = pd.DatetimeIndex(ds.time.values)
        missing = requested.difference(local)
        if retry_missing:
            unavailable = pd.DatetimeIndex(ds.time.values[ds.source_available.values == 0])
            missing = missing.union(requested.intersection(unavailable))
    return missing.sort_values()

