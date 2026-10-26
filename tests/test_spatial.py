import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from nightlights.spatial import pixel_geometry, fetch_aerial, attach_geometry


class SpatialReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells,cls.boundary,cls.building,cls.point = pixel_geometry()

    def test_native_hdf_grid_footprints_and_aoi_mask(self):
        schema = json.loads(Path('data/miami_history/VNP46A1/schema.json').read_text())
        west,east,south,north = schema['bounds']
        dx = (east-west)/schema['full_shape'][1]
        dy = (north-south)/schema['full_shape'][0]
        for row in self.cells.itertuples():
            rr,cc = schema['rows'][0]+row.row,schema['columns'][0]+row.column
            expected = (west+cc*dx,north-(rr+1)*dy,west+(cc+1)*dx,north-rr*dy)
            np.testing.assert_allclose(row.geometry.bounds,expected,rtol=0,atol=1e-11)
        self.assertEqual(int(self.cells.stored_aoi_mask.sum()),7)
        self.assertTrue(self.cells.pixel_area_m2.between(170000,190000).all())

    def test_building_partition_and_mixed_pixel_fractions(self):
        c = self.cells.set_index('pixel_id')
        self.assertEqual(set(c.index[c.building_overlap_m2>0]),{'r0c2','r1c2'})
        self.assertAlmostEqual(c.building_inside_pixel_pct.sum(),100,places=5)
        self.assertAlmostEqual(c.loc['r0c2','building_inside_pixel_pct'],85.7575,places=3)
        self.assertAlmostEqual(c.loc['r1c2','building_inside_pixel_pct'],14.2425,places=3)
        self.assertLess(c.loc['r0c2','pixel_covered_by_building_pct'],7)
        self.assertTrue(c.loc['r0c2','contains_reference_point'])

    def test_measurement_join_preserves_csv_rows_despite_float_rounding(self):
        original = pd.read_csv('outputs/validation_2016/pixel_coverage.csv')
        perturbed = original.copy()
        perturbed['longitude'] += 1e-13
        joined = attach_geometry(perturbed,self.cells.drop(columns='geometry'))
        self.assertEqual(len(joined),len(original))
        self.assertEqual(len(joined),35)
        shifted = original.copy()
        shifted['longitude'] += .001
        with self.assertRaises(AssertionError):
            attach_geometry(shifted,self.cells.drop(columns='geometry'))

    @unittest.skipUnless(Path('data/site_review/naip2017.json').exists(),'aerial cache not downloaded')
    def test_cached_aerial_uses_no_network_and_retains_crs(self):
        with patch('nightlights.spatial.httpx.get',side_effect=AssertionError('Unexpected network call')):
            image,record = fetch_aerial()
        self.assertTrue(image.is_file())
        self.assertEqual(record['nominal_imagery_year'],2017)
        extent = record['export']['extent']
        self.assertEqual(extent['spatialReference'].get('latestWkid',extent['spatialReference']['wkid']),32612)


if __name__=='__main__':
    unittest.main()
