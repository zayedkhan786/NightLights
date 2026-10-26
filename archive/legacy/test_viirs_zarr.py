"""Storage integrity tests using real acquired six-day AOI samples; no network."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

import viirs_zarr as vz
from update_viirs import update, nasa_batch


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob("*") if p.is_file()}


class StorageTests(unittest.TestCase):
    def setUp(self):
        Path("data/zarr_validation").mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir="data/zarr_validation")
        self.root = Path(self.temporary.name)
        self.product = "VNP46A2"
        self.store = self.root / "test.zarr"
        with xr.open_dataset("data/miami_history/VNP46A2/2016-01-01_2016-01-31.nc", mask_and_scale=False) as ds:
            self.ds = ds.isel(time=slice(0, 6)).load()

    def tearDown(self):
        self.temporary.cleanup()

    def test_append_repeat_raw_values_and_crs(self):
        for product in vz.BANDS:
            with xr.open_dataset(f"data/miami_history/{product}/2016-01-01_2016-01-31.nc", mask_and_scale=False) as original:
                ds = original.isel(time=slice(0, 6)).load()
            store = self.root / f"{product}.zarr"
            self.assertEqual(vz.upsert(store, ds.isel(time=slice(0, 3)), product)["added_dates"], 3)
            self.assertEqual(vz.upsert(store, ds, product)["added_dates"], 3)
            before = fingerprints(store)
            self.assertEqual(vz.upsert(store, ds, product)["action"], "unchanged")
            self.assertEqual(before, fingerprints(store))
            with vz.open_product(store) as actual:
                self.assertEqual(actual.rio.crs.to_epsg(), 4326)
                for band in vz.BANDS[product]:
                    np.testing.assert_array_equal(actual[band].values, ds[band].values)
                    self.assertEqual(actual[band].dtype, ds[band].dtype)

    def test_repair_missing_source(self):
        missing = self.ds.copy(deep=True)
        missing.source_available.values[1] = 0
        missing.source_file.values[1] = ""
        for name in vz.BANDS[self.product]:
            missing[name].values[1] = missing[name].attrs["_FillValue"]
        vz.upsert(self.store, missing, self.product)
        self.assertEqual(len(vz.pending_dates(self.store, "2016-01-01", "2016-01-06")), 0)
        self.assertEqual(len(vz.pending_dates(self.store, "2016-01-01", "2016-01-06", True)), 1)
        self.assertEqual(vz.upsert(self.store, self.ds.isel(time=[1]), self.product)["repaired_dates"], 1)
        vz._validate(self.store, vz.prepare(self.ds, self.product))

    def test_failure_rolls_back_tail_chunk_and_metadata(self):
        vz.upsert(self.store, self.ds.isel(time=slice(0, 3)), self.product)
        before = fingerprints(self.store)
        with patch.object(vz, "_validate", side_effect=RuntimeError("injected after write")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                vz.upsert(self.store, self.ds, self.product)
        self.assertEqual(before, fingerprints(self.store))
        self.assertEqual(vz.upsert(self.store, self.ds, self.product)["added_dates"], 3)

    def test_earlier_insertion_sorted_and_conflicting_revision_rejected(self):
        vz.upsert(self.store, self.ds.isel(time=[2, 3, 4, 5]), self.product)
        self.assertEqual(vz.upsert(self.store, self.ds, self.product)["action"], "insert")
        vz._validate(self.store, vz.prepare(self.ds, self.product))
        changed = self.ds.copy(deep=True)
        changed[vz.BANDS[self.product][0]].values[0, 0, 0] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            vz.upsert(self.store, changed, self.product)

    def test_duplicate_dates_wrong_crs_and_changed_grid_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            vz.upsert(self.store, self.ds.isel(time=[0, 0]), self.product)
        wrong = self.ds.copy(deep=True)
        wrong.attrs["crs"] = "EPSG:3857"
        with self.assertRaisesRegex(ValueError, "4326"):
            vz.upsert(self.store, wrong, self.product)
        vz.upsert(self.store, self.ds, self.product)
        with self.assertRaisesRegex(ValueError, "grid changed"):
            vz.upsert(self.store, self.ds.assign_coords(x=self.ds.x + 0.1), self.product)

    def test_future_provider_receives_only_new_days_and_repeat_skips_it(self):
        def provider(product, days, *args):
            ds = self.ds.sel(time=days).copy(deep=True)
            ds.attrs["boundary_sha256"] = args[-1]
            return ds, len(days)
        # No local checkpoints in this temporary directory: exercise the
        # acquisition branch with a stub, without any network or credentials.
        with patch("update_viirs.nasa_batch", side_effect=provider) as nasa:
            first = update(self.product, "2016-01-01", "2016-01-03", output=self.root,
                           history=self.root, allow_download=True)
            second = update(self.product, "2016-01-01", "2016-01-06", output=self.root,
                            history=self.root, allow_download=True)
            repeat = update(self.product, "2016-01-01", "2016-01-06", output=self.root,
                            history=self.root, allow_download=True)
            self.assertEqual(nasa.call_count, 2)
            self.assertEqual(list(nasa.call_args_list[1].args[1]), list(pd.date_range("2016-01-04", "2016-01-06")))
            self.assertEqual(repeat["nasa_batches"], 0)

    def test_successful_empty_listing_is_distinct_from_api_failure(self):
        roi = gpd.read_file("miami_smelter_boundary.geojson").to_crs(4326).dissolve()
        token = self.root / "fake_token"
        token.write_text("offline-test-placeholder")
        empty = pd.DataFrame(columns=["date", "TileID", "name"])
        async def manifest(*args):
            return empty
        with patch("update_viirs.BlackMarbleDownloader.get_manifest", manifest), patch("update_viirs.BlackMarbleDownloader.download", side_effect=AssertionError("no download for empty listing")):
            ds, count = nasa_batch(self.product, pd.date_range("2016-01-01", periods=2),
                roi.geometry.iloc[0], roi, token, self.root / "cache", "data/miami_history", "test-hash")
        self.assertEqual(count, 0)
        self.assertEqual(int(ds.source_available.sum()), 0)
        self.assertTrue(all(name in ds for name in vz.BANDS[self.product]))
        with patch("update_viirs.nasa_batch", side_effect=RuntimeError("API failure")):
            with self.assertRaisesRegex(RuntimeError, "API failure"):
                update(self.product, "2016-01-01", "2016-01-02", output=self.root,
                       history=self.root, allow_download=True)
        self.assertFalse((self.root / f"{self.product}.zarr").exists())

    def test_incremental_planner_skips_source_io_and_network_on_repeat(self):
        for product in vz.BANDS:
            first = update(product, "2016-01-01", "2016-01-03", output=self.root)
            second = update(product, "2016-01-01", "2016-01-06", output=self.root)
            self.assertEqual(first["pending_at_start"], 3)
            self.assertEqual(second["pending_at_start"], 3)
            store = self.root / f"{product}.zarr"
            before = fingerprints(store)
            with patch("update_viirs.local_batches", side_effect=AssertionError("no source I/O")), patch("update_viirs.nasa_batch", side_effect=AssertionError("no NASA requests")):
                repeat = update(product, "2016-01-01", "2016-01-06", output=self.root)
            self.assertEqual(repeat["pending_at_start"], 0)
            self.assertEqual(before, fingerprints(store))


if __name__ == "__main__":
    unittest.main()
