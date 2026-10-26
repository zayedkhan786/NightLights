import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import xarray as xr

from nightlights.history import current_table,reference_table,site_dataset,yearly_summary,_publish_series,acquired_end,plot_history
from nightlights.comparison import BANDS
from nightlights.processing import GAP
from nightlights.validation import validate_year


def example():
    values = np.full((3,2,4),np.nan)
    values[0] = 10
    values[0,0,2],values[0,1,2] = 20,40
    values[1,0,2] = 0
    ds = xr.Dataset(coords={'time':pd.date_range('2022-11-28',periods=3),'y':[33.414583333,33.410416667],
                            'x':[-110.864583333,-110.860416667,-110.85625,-110.852083333]})
    ds['aoi_intersects_pixel'] = (('y','x'),np.array([[0,1,1,1],[1,1,1,1]],dtype='uint8'))
    ds['source_available'] = ('time',np.array([1,1,0],dtype='uint8'))
    for _,band,_,_ in BANDS:
        ds[band+'__screened'] = (('time','y','x'),values.copy())
    ds[GAP+'__reference'] = (('time','y','x'),values.copy())
    age = np.zeros((3,2,4),dtype='uint8')
    age[0,0,2],age[0,1,2] = 1,5
    age[1,1,2] = 255
    ds['native_Latest_High_Quality_Retrieval'] = (('time','y','x'),age,{'_FillValue':255})
    return ds


class HistoryTests(unittest.TestCase):
    def test_default_end_uses_calendar_common_to_both_products(self):
        one = xr.Dataset(coords={'time':pd.date_range('2026-10-16','2026-10-18')})
        two = xr.Dataset(coords={'time':pd.date_range('2026-10-16','2026-10-17')})
        with patch('nightlights.history.open_product',side_effect=[one,two]):
            self.assertEqual(acquired_end(),'2026-10-17')

    def test_coverage_labels_actual_partial_year_end(self):
        import matplotlib
        matplotlib.use('Agg',force=True)
        import matplotlib.pyplot as plt
        ds = example().assign_coords(time=pd.date_range('2026-10-16',periods=3))
        daily = pd.concat([current_table(ds,p,b) for p,b,_,_ in BANDS],ignore_index=True)
        figures = plot_history(daily,yearly_summary(daily),reference_table(ds))
        try:
            labels = [tick.get_text() for tick in figures[1].axes[0].get_xticklabels()]
            self.assertEqual(labels,['2026\nthrough 18 Oct'])
        finally:
            for figure in figures:
                plt.close(figure)

    def test_primary_requires_two_and_counts_survive_missing_values(self):
        frame = current_table(example(),'VNP46A2','DNB_BRDF-Corrected_NTL')
        self.assertEqual(frame.main_value.iloc[0],30)
        self.assertTrue(np.isnan(frame.main_value.iloc[1]))
        self.assertEqual(frame.main_valid_pixels.tolist(),[2,1,0])
        self.assertEqual(frame.context_valid_pixels.tolist(),[7,1,0])
        self.assertEqual(frame.context_value.iloc[1],0)
        self.assertEqual(frame.r0c2_value.iloc[1],0)
        self.assertFalse(frame.source_available.iloc[2])

    def test_reference_age_requires_both_ages_and_is_not_current_measurement(self):
        ref = reference_table(example())
        self.assertEqual(ref.main_reference_value.iloc[0],30)
        self.assertEqual(ref.main_reference_max_age_days.iloc[0],5)
        self.assertTrue(np.isnan(ref.main_reference_max_age_days.iloc[1]))
        self.assertEqual(ref.main_reference_pixels_with_known_age.tolist(),[2,1,0])

    def test_summary_calendar_and_zarr_roundtrip(self):
        ds = example()
        daily = pd.concat([current_table(ds,p,b) for p,b,_,_ in BANDS],ignore_index=True)
        summary = yearly_summary(daily)
        self.assertEqual(summary.calendar_days.tolist(),[3]*4)
        self.assertEqual(summary.main_usable_days.tolist(),[1]*4)
        self.assertEqual(summary.context_usable_days.tolist(),[2]*4)
        combined = site_dataset(daily,reference_table(ds))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'site.zarr'
            _publish_series(combined,path)
            with xr.open_zarr(path,consolidated=True,chunks=None,mask_and_scale=False) as saved:
                np.testing.assert_array_equal(saved['DNB_BRDF-Corrected_NTL__main_valid_pixels'],[2,1,0])
                self.assertTrue(np.isnan(saved['DNB_BRDF-Corrected_NTL__main'].values[1]))

    def test_final_partial_year_has_no_december_query(self):
        with tempfile.TemporaryDirectory() as temp:
            report = {'input_signature':hashlib.sha256(('s'*11).encode()).hexdigest(),'annual_files':{}}
            (Path(temp)/'validation_report.json').write_text(json.dumps(report))
            with patch('nightlights.validation.process_month',return_value=(False,'s')) as month, patch('nightlights.validation.annual_hashes',return_value={}):
                validate_year(2022,output=temp,end='2022-11-30')
            self.assertEqual(month.call_count,11)
            self.assertEqual(str(month.call_args_list[-1].args[1]),'2022-11-30')
            self.assertFalse(any(str(c.args[0]).startswith('2022-12') for c in month.call_args_list))

    def test_october_final_year_stops_at_requested_day(self):
        with tempfile.TemporaryDirectory() as temp:
            report = {'input_signature':hashlib.sha256(('s'*10).encode()).hexdigest(),'annual_files':{}}
            (Path(temp)/'validation_report.json').write_text(json.dumps(report))
            with patch('nightlights.validation.process_month',return_value=(False,'s')) as month,patch('nightlights.validation.annual_hashes',return_value={}):
                validate_year(2026,output=temp,end='2026-10-18')
            self.assertEqual(month.call_count,10)
            self.assertEqual(str(month.call_args_list[-1].args[1]),'2026-10-18')


if __name__=='__main__':
    unittest.main()
