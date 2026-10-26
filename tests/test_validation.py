import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
import xarray as xr
import rioxarray
from nightlights.validation import monthly_coverage, process_month, _checksums, _publish_zarr


class AnnualValidationTests(unittest.TestCase):
    def test_annual_zarr_preserves_cf_grid_mapping_with_explicit_chunk_encoding(self):
        with tempfile.TemporaryDirectory() as temp:
            ds = xr.Dataset({'value':(('time','y','x'),np.arange(8,dtype=float).reshape(2,2,2))},
                            coords={'time':pd.date_range('2016-02-28',periods=2),'y':[33.42,33.41],'x':[-110.86,-110.85]})
            path = Path(temp)/'derived.zarr'
            _publish_zarr(ds,path)
            with xr.open_zarr(path,consolidated=True,chunks=None,mask_and_scale=False,decode_coords='all') as saved:
                self.assertEqual(saved.rio.crs.to_epsg(),4326)
                self.assertIn('spatial_ref',saved.coords)
                np.testing.assert_array_equal(saved.value,ds.value)

    def test_leap_year_monthly_coverage_denominators(self):
        frame = pd.DataFrame({'date':pd.date_range('2016-01-01','2016-12-31').strftime('%Y-%m-%d')})
        frame = frame.assign(product='VNP46A2',band='test',source_available=True,
                             retained_pixels=3,aoi_pixels=7,meaning='current observation')
        result = monthly_coverage(frame)
        self.assertEqual(len(result),12)
        self.assertEqual(result.calendar_days.sum(),366)
        feb = result[result.month=='2016-02'].iloc[0]
        self.assertEqual(feb.calendar_days,29)
        self.assertEqual(feb.retained_pixel_days,87)
        self.assertEqual(feb.possible_pixel_days,203)
        self.assertAlmostEqual(feb.pixel_coverage_fraction,3/7)

    def test_checkpoint_reuse_and_corruption_or_input_change_invalidate(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'monthly/2016-01'
            folder.mkdir(parents=True)
            output = folder/'example.csv'
            output.write_text('checked-output')
            record = {'signature':'same-input','status':'complete','files':_checksums(folder)}
            (folder/'checkpoint.json').write_text(json.dumps(record))
            with patch('nightlights.validation.month_signature',return_value='same-input'), patch('nightlights.validation.load_window',side_effect=RuntimeError('reprocessing needed')) as load:
                changed, signature = process_month('2016-01-01','2016-01-31',temp,'raw','history')
                self.assertFalse(changed)
                load.assert_not_called()
                output.write_text('corrupted-output')
                with self.assertRaisesRegex(RuntimeError,'reprocessing needed'):
                    process_month('2016-01-01','2016-01-31',temp,'raw','history')
            output.write_text('checked-output')
            with patch('nightlights.validation.month_signature',return_value='changed-input'), patch('nightlights.validation.load_window',side_effect=RuntimeError('new input')):
                with self.assertRaisesRegex(RuntimeError,'new input'):
                    process_month('2016-01-01','2016-01-31',temp,'raw','history')


if __name__=='__main__':
    unittest.main()
