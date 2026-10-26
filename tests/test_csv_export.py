import unittest
import numpy as np
from nightlights.csv_export import BASE_COLUMNS,SCIENCE,cloud_summary,spatial_stats,year_table
from tests.test_history import example


def source():
    ds = example()
    ds.attrs['boundary_sha256'] = 'fixture'
    for _,band in SCIENCE.values():
        suffix = '__reference' if 'Gap_Filled' in band else '__screened'
        ds[band+'__decoded'] = ds[band+suffix].copy(deep=True)
        if 'Gap_Filled' not in band:
            ds[band+'__decoded'].values[1,1,2] = 12
    ds['cloud_cloud_confidence'] = (('time','y','x'),np.zeros((3,2,4),dtype='int16'))
    ds.cloud_cloud_confidence.values[2] = -1
    ds['source_file'] = ('time',['one.h5','two.h5',''])
    return ds


class CsvExportTests(unittest.TestCase):
    def test_population_statistics_zero_and_incomplete_main(self):
        values = np.array([[10,30],[0,np.nan],[np.nan,np.nan]])
        main,count = spatial_stats(values,2)
        self.assertEqual(count.tolist(),[2,1,0])
        self.assertEqual(main['std'][0],10)
        self.assertTrue(np.isnan(main['mean'][1]))
        context,_ = spatial_stats(values,1)
        self.assertEqual(context['mean'][1],0)
        self.assertEqual(context['std'][1],0)
        self.assertTrue(np.isnan(context['max'][2]))

    def test_cloud_percentage_uses_only_known_flags(self):
        ds = source()
        mask = np.zeros((2,4),dtype=bool); mask[:,2] = True
        ds.cloud_cloud_confidence.values[0,0,2] = 2
        ds.cloud_cloud_confidence.values[1,1,2] = -1
        denominator,cloudy,percentage = cloud_summary(ds,mask)
        self.assertEqual(denominator.tolist(),[2,1,0])
        np.testing.assert_allclose(cloudy,[1,0,np.nan],equal_nan=True)
        np.testing.assert_allclose(percentage,[50,0,np.nan],equal_nan=True)

    def test_reference_schema_counts_aliases_age_and_primary_gate(self):
        ds = source()
        frame = year_table({'VNP46A1':ds,'VNP46A2':ds},'main_2pixels','fixture')
        self.assertEqual(len(BASE_COLUMNS),48)
        self.assertEqual(list(frame.columns[:48]),BASE_COLUMNS)
        self.assertEqual(frame.num_pixels.tolist(),[2,2,2])
        self.assertEqual(frame.valid_capture.tolist(),[1,0,0])
        self.assertEqual(frame.valid_pixels_processed_l2.tolist(),[2,1,0])
        self.assertEqual(frame.qa_rejected_pixels_l2.tolist(),[0,1,0])
        self.assertEqual(frame.mean_processed_l2.iloc[0],30)
        self.assertTrue(np.isnan(frame.mean_processed_l2.iloc[1]))
        np.testing.assert_allclose(frame.mean_count,frame.mean_processed_l2,equal_nan=True)
        np.testing.assert_allclose(frame.mean_l2_gapfilled,frame.mean_processed_l2_gapfilled,equal_nan=True)
        self.assertEqual(frame.gapfilled_max_age_days.iloc[0],5)
        self.assertTrue(np.isnan(frame.gapfilled_max_age_days.iloc[1]))
        self.assertEqual(frame.temperature_units.unique().tolist(),['K'])

    def test_original_context_includes_zero_and_does_not_change_mask(self):
        ds = source(); before = ds.aoi_intersects_pixel.values.copy()
        frame = year_table({'VNP46A1':ds,'VNP46A2':ds},'context_7pixels','fixture')
        self.assertEqual(frame.num_pixels.unique().tolist(),[7])
        self.assertEqual(frame.mean_processed_l2.iloc[1],0)
        self.assertEqual(frame.valid_capture.tolist(),[1,1,0])
        np.testing.assert_array_equal(ds.aoi_intersects_pixel,before)
        with self.assertRaisesRegex(ValueError,'original analysis AOI'):
            year_table({'VNP46A1':ds,'VNP46A2':ds},'context_7pixels','wrong-hash')

    def test_native_zarr_variable_length_strings_are_supported(self):
        ds = source()
        ds['source_file'] = ('time',np.array(['one.h5','two.h5',''],dtype=np.dtypes.StringDType()))
        frame = year_table({'VNP46A1':ds,'VNP46A2':ds},'main_2pixels','fixture')
        self.assertEqual(frame.a1_source_file.tolist(),['one.h5','two.h5',''])


if __name__=='__main__':
    unittest.main()
