import json
from datetime import date
from pathlib import Path
import tempfile
import unittest
import asyncio
import httpx
import geopandas as gpd

import h5py
import numpy as np
from shapely.geometry import box
import xarray as xr

from nightlights.acquisition import assemble_cube, month_ranges, open_history, schema_for_file
from nightlights.download import save_source_checkpoint, persist_progress, manifest_allow_empty, BlackMarbleDownloader, Product
from unittest.mock import patch
import pandas as pd


class AcquisitionTests(unittest.TestCase):
    def manifest_response(self,payload,status=200):
        async def get_response(*args,**kwargs):
            return httpx.Response(status,json=payload,request=httpx.Request('GET','https://example.invalid/listing'))
        downloader = BlackMarbleDownloader(token='offline-placeholder',directory=Path(self.tmp.name))
        roi = gpd.GeoDataFrame(geometry=[self.geometry],crs=4326)
        with patch('nightlights.download.bm_download.get_url',get_response):
            return asyncio.run(manifest_allow_empty(downloader,roi,Product('VNP46A2'),[date(2024,7,11)]))

    def test_successful_empty_nasa_response_works_around_library_error(self):
        manifest = self.manifest_response({'content':[],'n_pages':1})
        self.assertTrue(manifest.empty)
        self.assertEqual(list(manifest.columns),['date','TileID','name'])

    def test_api_errors_and_malformed_content_never_become_missing_sources(self):
        for payload,status,error in [({'error':'unavailable'},200,ValueError),
                                     ({'content':[]},503,httpx.HTTPStatusError),
                                     ({'content':[{'unexpected':'field'}]},200,KeyError)]:
            with self.subTest(payload=payload,status=status),self.assertRaises(error):
                self.manifest_response(payload,status)

    def test_progress_retries_temporary_windows_lock(self):
        target = Path(self.tmp.name)/'progress.json'
        replace = Path.replace
        calls = []
        def locked_twice(source,destination):
            calls.append(source)
            if len(calls)<3:
                raise PermissionError('temporary file lock')
            return replace(source,destination)
        with patch.object(Path,'replace',locked_twice),patch('nightlights.download.time.sleep'):
            self.assertTrue(persist_progress(target,{'state':'running'}))
        self.assertEqual(len(calls),3)
        self.assertEqual(json.loads(target.read_text()),{'state':'running'})

    def test_locked_progress_file_does_not_abort_acquisition(self):
        target = Path(self.tmp.name)/'progress.json'
        with patch.object(Path,'replace',side_effect=PermissionError('locked')),patch('nightlights.download.time.sleep'):
            self.assertFalse(persist_progress(target,{'state':'running'}))

    def test_source_checkpoint_preserves_every_band_and_reuses_identical_data(self):
        ds = self.dataset()
        ds.attrs['boundary_sha256'] = 'fixture-boundary'
        manifest = pd.DataFrame([{'name':'source.h5','date':date(2016,1,1)}])
        root = Path(self.tmp.name)/'checkpoints'
        save_source_checkpoint(ds,root,'VNP46A2',manifest)
        path = root/'VNP46A2/2016-01-01_2016-01-02.nc'
        with xr.open_dataset(path,engine='h5netcdf',mask_and_scale=False) as saved:
            xr.testing.assert_identical(ds,saved.load())
        before = path.stat().st_mtime_ns
        save_source_checkpoint(ds,root,'VNP46A2',manifest)
        self.assertEqual(before,path.stat().st_mtime_ns)
        self.assertEqual(json.loads(path.with_suffix('.json').read_text())['status'],'complete')

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"source.h5"
        self.geometry=box(-110.99,33.01,-110.01,33.99)
        with h5py.File(self.path,"w") as f:
            for name,value in {"WestBoundingCoord":[-111.],"EastBoundingCoord":[-110.],
                "SouthBoundingCoord":[33.],"NorthBoundingCoord":[34.],
                "RangeBeginningDate":b"2016-01-01","ShortName":b"VNP46A2","VersionID":b"002"}.items():
                f.attrs[name]=value
            g=f.create_group("HDFEOS/GRIDS/test/Data Fields")
            d=g.create_dataset("radiance",data=np.array([[0.,-999.9],[2.,3.]],dtype=np.float32))
            d.attrs["_FillValue"]=np.float32(-999.9)
            d.attrs["scale_factor"]=0.1
            d.attrs["offset"]=2.
            q=g.create_dataset("quality",data=np.array([[0,255],[1,5]],dtype=np.uint8))
            q.attrs["_FillValue"]=np.uint8(255)

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self):
        schema=schema_for_file(self.path,self.geometry)
        ds=assemble_cube({date(2016,1,1):self.path},[date(2016,1,1),date(2016,1,2)],
                         schema,self.geometry,"VNP46A2")
        return ds

    def test_no_scaling_or_quality_cleaning(self):
        ds=self.dataset()
        np.testing.assert_array_equal(ds.radiance[0],np.array([[0.,-999.9],[2.,3.]],dtype=np.float32))
        np.testing.assert_array_equal(ds.quality[0],np.array([[0,255],[1,5]],dtype=np.uint8))
        self.assertEqual(ds.radiance.dims,("time","y","x"))
        self.assertEqual(ds.quality.dtype,np.uint8)

    def test_missing_date_and_exact_scene_metadata(self):
        ds=self.dataset()
        self.assertEqual(ds.source_available.values.tolist(),[1,0])
        self.assertTrue((ds.quality[1].values==255).all())
        meta=json.loads(ds.scene_band_metadata.values[0])
        self.assertEqual(meta["radiance"]["scale_factor"],0.1)
        np.testing.assert_allclose(ds.x,[-110.75,-110.25])
        np.testing.assert_allclose(ds.y,[33.75,33.25])

    def test_netcdf_and_multimonth_open_preserve_raw_values_and_dates(self):
        ds=self.dataset()
        folder=Path(self.tmp.name)/"VNP46A2"
        folder.mkdir()
        ds.to_netcdf(folder/"2016-01.nc",engine="netcdf4")
        later=ds.assign_coords(time=np.array(["2016-02-01","2016-02-02"],dtype="datetime64[ns]"))
        later.to_netcdf(folder/"2016-02.nc",engine="netcdf4")
        with open_history(self.tmp.name,"VNP46A2") as opened:
            self.assertEqual(opened.sizes["time"],4)
            self.assertEqual(str(opened.time.values[2])[:10],"2016-02-01")
            np.testing.assert_array_equal(opened.radiance[0],ds.radiance[0])
            self.assertEqual(opened.quality.dtype,np.uint8)

    def test_calendar_chunking_handles_leap_year(self):
        ranges=list(month_ranges(date(2016,1,30),date(2016,3,2)))
        self.assertEqual(ranges[1],(date(2016,2,1),date(2016,2,29)))


if __name__=="__main__":
    unittest.main()
