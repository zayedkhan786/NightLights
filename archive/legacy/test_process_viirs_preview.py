import json
import unittest
import numpy as np
from process_viirs_preview import decode_values, cloud_fields, load_preview, process_sample, GAP


class DecodeTests(unittest.TestCase):
    def test_scale_offset_fill_and_native_range(self):
        raw = np.array([0,10000,65534,65535], dtype='uint16')
        physical, fill, outside, _, _ = decode_values(raw, {'_FillValue':65535,
            'valid_min':0, 'valid_max':65534, 'scale_factor':.0025, 'add_offset':203})
        np.testing.assert_allclose(physical[:3], [203,228,366.835])
        self.assertTrue(np.isnan(physical[3]))
        np.testing.assert_array_equal(fill,[False,False,False,True])
        self.assertFalse(outside.any())

    def test_float_radiance_identity_zero_negative_and_nan(self):
        values = np.array([0,12.5,-1,-999.9,np.nan],dtype='float32')
        physical, fill, outside, _, _ = decode_values(values, {'_FillValue':float(np.float32(-999.9)), 'valid_range':'>=0.0'})
        np.testing.assert_allclose(physical[:2], [0,12.5])
        self.assertTrue(np.isnan(physical[2:]).all())
        np.testing.assert_array_equal(fill,[False,False,False,True,True])
        np.testing.assert_array_equal(outside,[False,False,True,False,False])

    def test_cloud_bits_and_missing_flags(self):
        q = np.array([[50,50|1|256|512|1024|4096|8192,65535]],dtype='uint16')
        fields, valid = cloud_fields(q, {'_FillValue':65535,'valid_min':0,'valid_max':65534}, True)
        self.assertEqual(fields['cloud_quality'][0,0], 3)
        self.assertEqual(fields['cloud_confidence'][0,0], 0)
        for flag in ('day','shadow','cirrus','snow_ice','aurora','lunar_eclipse'):
            self.assertEqual(fields[flag][0,1], 1)
            self.assertEqual(fields[flag][0,2], -1)
        missing, valid = cloud_fields(q, {'_FillValue':65535}, False)
        self.assertFalse(valid.any())
        self.assertTrue(all((v == -1).all() for v in missing.values()))

    def test_unsupported_metadata_fails(self):
        with self.assertRaises(ValueError):
            decode_values(np.array([1]),{'valid_range':'unknown'})
        with self.assertRaises(ValueError):
            decode_values(np.array([1]),{'add_offset':2,'offset':3})


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples, cls.extras, cls.results, cls.audit, _, cls.summary = load_preview()

    def test_six_day_counts_and_cloudy_gaps(self):
        self.assertEqual(len(self.audit), 240)
        self.assertEqual(self.audit.inside_aoi.sum(), 210)
        self.assertEqual(len(self.summary), 30)
        a2 = self.summary[(self.summary['product']=='VNP46A2') & (self.summary.band=='DNB_BRDF-Corrected_NTL')]
        self.assertEqual(a2.retained_pixels.tolist(), [3,7,6,0,0,0])
        reference = self.summary[self.summary.band==GAP]
        self.assertEqual(reference.retained_pixels.tolist(), [7]*6)
        self.assertTrue(self.audit[self.audit.band==GAP].screened.isna().all())

    def test_per_date_packing_changes_and_input_immutability(self):
        sample = self.samples['VNP46A1'].copy(deep=True)
        sample.BrightnessTemperature_M13.values[:,0,1] = 10000
        metadata = [json.loads(s) for s in sample.scene_band_metadata.values]
        metadata[1]['BrightnessTemperature_M13'].update(scale_factor=.005, add_offset=210)
        sample['scene_band_metadata'] = ('time',np.array([json.dumps(m) for m in metadata],dtype=object))
        before = sample.BrightnessTemperature_M13.values.copy()
        output, _, _ = process_sample(sample, self.extras['VNP46A1'])
        actual = output['BrightnessTemperature_M13__decoded'].values[:,0,1]
        self.assertAlmostEqual(actual[0],228,places=4)
        self.assertAlmostEqual(actual[1],260,places=4)
        np.testing.assert_array_equal(before, sample.BrightnessTemperature_M13.values)

    def test_zero_radiance_retained_and_band_specific_qa(self):
        sample = self.samples['VNP46A1'].copy(deep=True)
        extra = self.extras['VNP46A1'].copy(deep=True)
        sample.QF_Cloud_Mask.values[:] = 50
        sample.DNB_At_Sensor_Radiance.values[:] = 0
        extra.QF_DNB.values[:] = 0
        extra.QF_VIIRS_M13.values[:] = 0
        extra.QF_VIIRS_M15.values[:] = 0
        extra.QF_VIIRS_M13.values[0,0,1] = 4
        result, audit, _ = process_sample(sample,extra)
        self.assertEqual(result.DNB_At_Sensor_Radiance__screened.values[0,0,1],0)
        self.assertTrue(np.isnan(result.BrightnessTemperature_M13__screened.values[0,0,1]))
        self.assertTrue(np.isfinite(result.BrightnessTemperature_M15__screened.values[0,0,1]))
        self.assertTrue(np.isnan(result.DNB_At_Sensor_Radiance__screened.values[:,0,0]).all())

    def test_missing_source_is_not_cloudy_observation(self):
        sample = self.samples['VNP46A1'].copy(deep=True)
        sample.source_available.values[0] = 0
        sample.scene_band_metadata.values[0] = ''
        result, audit, _ = process_sample(sample,self.extras['VNP46A1'])
        rows = audit[(audit.date=='2016-01-01') & audit.inside_aoi]
        self.assertTrue((rows.reason_flags == 1).all())
        self.assertTrue(rows.decoded.isna().all())
        self.assertTrue((rows.cloud_confidence == -1).all())

    def test_source_and_grid_mismatch_rejected(self):
        extra = self.extras['VNP46A1'].copy(deep=True)
        extra.source_file.values[0] = 'different-file.h5'
        with self.assertRaises(AssertionError):
            process_sample(self.samples['VNP46A1'],extra)


if __name__ == '__main__':
    unittest.main()
