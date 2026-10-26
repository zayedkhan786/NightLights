"""Scientific handling checks using controlled geometry and pixel data."""
from datetime import date
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from shapely.geometry import box

from extract_month import extract_day, polygon_weights, weighted_mean
from read_nightlights import CORRECTED, GAP_FILLED


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/"synthetic.h5"
        with h5py.File(self.path, "w") as h:
            for k, v in {"WestBoundingCoord":-111., "EastBoundingCoord":-110.,
                         "SouthBoundingCoord":33., "NorthBoundingCoord":34.,
                         "ShortName":b"VNP46A2", "VersionID":b"002",
                         "RangeBeginningDate":b"2026-09-01"}.items():
                h.attrs[k] = v
            g = h.create_group("HDFEOS/GRIDS/test/Data Fields")
            for name in (CORRECTED, GAP_FILLED):
                ds = g.create_dataset(name, data=np.full((4,4), 10., dtype=np.float32))
                ds.attrs["_FillValue"] = np.array([-999.9], dtype=np.float32)
                ds.attrs["scale_factor"] = [1.]
                ds.attrs["offset"] = [0.]
            g.create_dataset("Mandatory_Quality_Flag", data=np.zeros((4,4), dtype=np.uint8))
            g.create_dataset("QF_Cloud_Mask", data=np.full((4,4), 48, dtype=np.uint16))
            g.create_dataset("Snow_Flag", data=np.zeros((4,4), dtype=np.uint8))
            g.create_dataset("Latest_High_Quality_Retrieval", data=np.full((4,4), 2, dtype=np.uint8))
        self.polygon = box(-110.75, 33.25, -110.25, 33.75)

    def tearDown(self):
        self.tmp.cleanup()

    def test_weighted_mean_keeps_zero_and_excludes_nan(self):
        mean, coverage, count = weighted_mean(np.array([0., 10., np.nan]),
            np.array([1., 3., 1.]), np.ones(3, dtype=bool))
        self.assertEqual(mean, 7.5)
        self.assertEqual(coverage, 0.8)
        self.assertEqual(count, 2)

    def test_no_valid_measurement_is_missing_not_zero(self):
        mean, coverage, count = weighted_mean(np.array([np.nan]), np.array([2.]), np.array([True]))
        self.assertIsNone(mean)
        self.assertEqual((coverage, count), (0., 0))

    def test_polygon_intersections_cover_boundary_including_hole(self):
        polygon = self.polygon.difference(box(-110.6,33.4,-110.4,33.6))
        with h5py.File(self.path) as h:
            _, w, pixels, epsg = polygon_weights(h, polygon)
        self.assertEqual(len(pixels), 4)
        self.assertEqual(epsg, 32612)
        self.assertTrue((w > 0).all())
        self.assertGreater(w.sum(), 0)

    def test_missing_retrieval_and_cloud_reduce_coverage_without_erasing_gapfill(self):
        with h5py.File(self.path,"r+") as h:
            g=h["HDFEOS/GRIDS/test/Data Fields"]
            g[CORRECTED][1,1] = -999.9
            g["Mandatory_Quality_Flag"][1,1] = 255
            g["QF_Cloud_Mask"][1,2] = 48 | (3 << 6)
        row, _, _ = extract_day(self.path, date(2026,9,1), self.polygon, 0.8)
        self.assertEqual(row["status"], "insufficient_coverage")
        self.assertIsNone(row["corrected_mean_nw_cm2_sr"])
        self.assertEqual(row["corrected_available_mean_nw_cm2_sr"], 10.)
        self.assertEqual(row["corrected_valid_pixel_count"], 2)
        self.assertEqual(row["gap_filled_mean_nw_cm2_sr"], 10.)
        self.assertAlmostEqual(row["gap_filled_coverage_fraction"], 1.)
        self.assertEqual(row["gap_retrieval_age_max_days"], 2.)

    def test_fully_screened_observation(self):
        row, _, _=extract_day(self.path, date(2026,9,1), self.polygon, 0.8)
        self.assertEqual(row["status"], "usable")
        self.assertEqual(row["corrected_mean_nw_cm2_sr"], 10.)
        self.assertEqual(row["corrected_valid_pixel_count"], 4)

    def test_wrong_observation_date_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"date"):
            extract_day(self.path, date(2026,9,2), self.polygon, 0.8)


if __name__ == "__main__":
    unittest.main()
