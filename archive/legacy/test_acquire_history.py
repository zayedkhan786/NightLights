import json
from datetime import date
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from shapely.geometry import box
import xarray as xr

from acquire_history import assemble_cube, month_ranges, open_history, schema_for_file


class AcquisitionTests(unittest.TestCase):
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
