import unittest
import numpy as np
import pandas as pd
from nightlights.comparison import selection_series,paired_metrics,compare_arrays


class PixelComparisonTests(unittest.TestCase):
    def test_combined_completeness_zero_and_excluded_pixel(self):
        mask = np.array([[0,1,1,1],[1,1,1,1]],dtype=bool)
        values = np.full((3,2,4),np.nan)
        values[0,:,:] = 100
        values[0,0,0] = 99999
        values[0,0,2],values[0,1,2] = 10,20
        values[1,0,2] = 0
        values[2,1,1] = 5
        result = selection_series(values,mask)
        self.assertEqual(result.pair_both.iloc[0],15)
        self.assertAlmostEqual(result.aoi_any.iloc[0],530/7)
        self.assertEqual(result.pair_any.iloc[1],0)
        self.assertEqual(result.aoi_any.iloc[1],0)
        self.assertTrue(np.isnan(result.pair_both.iloc[1]))
        self.assertTrue(np.isnan(result.aoi_all.iloc[1]))
        self.assertTrue(np.isnan(result.pair_any.iloc[2]))
        self.assertEqual(result.aoi_any.iloc[2],5)
        self.assertEqual(result.pair_valid_pixels.tolist(),[2,1,0])
        self.assertEqual(result.aoi_valid_pixels.tolist(),[7,1,1])

    def frame(self,a,b):
        return pd.DataFrame({'date':pd.date_range('2016-01-01',periods=len(a)).strftime('%Y-%m-%d'),
            'product':'test','band':'test','r0c2':a,'r1c2':b,'aoi_valid_pixels':7,'pair_valid_pixels':2})

    def test_metrics_and_peaks_ignore_unshared_extreme_dates(self):
        frame = self.frame([1,2,3,999,np.nan],[2,4,6,np.nan,999])
        m,peaks = paired_metrics(frame,'r0c2','r1c2')
        self.assertEqual(m['common_days'],3)
        self.assertEqual(m['left_mean'],2)
        self.assertEqual(m['right_mean'],4)
        self.assertEqual(m['mean_difference_left_minus_right'],-2)
        self.assertAlmostEqual(m['rmse'],np.sqrt(14/3))
        self.assertAlmostEqual(m['pearson_r'],1)
        self.assertAlmostEqual(m['spearman_r'],1)
        self.assertEqual(m['shared_upper_decile_days'],1)
        self.assertEqual(m['upper_decile_jaccard'],1)
        self.assertEqual(peaks.date.tolist(),['2016-01-03'])

    def test_no_common_dates_and_constant_series(self):
        m,peaks = paired_metrics(self.frame([1,np.nan],[np.nan,1]),'r0c2','r1c2')
        self.assertEqual(m['common_days'],0)
        self.assertTrue(np.isnan(m['pearson_r']))
        self.assertTrue(peaks.empty)
        m,peaks = paired_metrics(self.frame([1,2,3],[5,5,5]),'r0c2','r1c2')
        self.assertTrue(np.isnan(m['pearson_r']))
        self.assertTrue(np.isnan(m['spearman_r']))
        self.assertEqual(m['right_upper_decile_days'],3)
        self.assertTrue(m['maximum_date_sets_overlap'])

    def test_annual_coverage_and_legacy_baseline_reconciliation(self):
        daily,summary,metrics,peaks,monthly = compare_arrays()
        self.assertEqual(len(daily),366*4)
        self.assertEqual(len(summary),24)
        self.assertEqual(len(metrics),24)
        self.assertEqual(len(monthly),12*24)
        for _,g in daily.groupby(['product','band']):
            self.assertTrue(g.pair_both.notna().sum()<=g.r0c2.notna().sum())
            self.assertTrue(g.pair_both.notna().sum()<=g.r1c2.notna().sum())
            self.assertTrue(g.pair_any.notna().sum()<=g.aoi_any.notna().sum())
            self.assertTrue(g.aoi_all.notna().sum()<=g.pair_both.notna().sum())


if __name__=='__main__':
    unittest.main()
